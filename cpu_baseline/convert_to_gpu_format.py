"""
Unified converter: any supported real-world LP input format -> the
shared GPU-format COO file (gpu_format.py), one entry point instead of
a separate script per format.

Why one converter instead of "any format"
-------------------------------------------
Genuinely accepting *any* input format is an unbounded promise -- every
new format needs real parsing logic, and a hackathon demo is the worst
place to discover an unhandled edge case live. Instead this covers a
deliberately bounded, real-world set and fails loudly (never silently
guesses) on anything else:

    .mps   Netlib/MIPLIB-style MPS files (via mps_to_txt.py's
           highspy-based canonicalizer -- variable bounds, ranged rows,
           free variables, RANGES/MARKER sections, etc.)
    .lp    Free-format LP files -- highspy's readModel() auto-detects
           LP vs MPS from the file itself, so this reuses the exact
           same canonicalization path as .mps.
    .csv   The company-data schema used by the Streamlit UI:
           type,row,col,value,rhs,sense -- "constraint" rows define A/b,
           an optional "objective" rows subset defines c. If no
           objective rows are present, c is all zeros and this is
           reported explicitly (never invented) -- matching the UI's
           own honesty rule for the same input schema.

All three paths converge on gpu_format.py's export_to_gpu_format(), so
downstream (cpu_solve.py, gurobi_solve.py, cplex_solve.py, the GPU
solver) never has to know or care which format the problem started as.

Usage
-----
    python convert_to_gpu_format.py company_data.csv --out problem.txt
    python convert_to_gpu_format.py afiro.mps
    python convert_to_gpu_format.py model.lp --out model.txt
    python convert_to_gpu_format.py data.csv --format csv --out problem.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from gpu_format import export_to_gpu_format
from mps_to_txt import canonicalize_mps, write_meta

SUPPORTED_FORMATS = {".mps": "mps", ".lp": "lp", ".csv": "csv"}


def canonicalize_csv(path: str) -> dict:
    """Parse the company CSV schema (type,row,col,value,rhs,sense) into
    the same dict shape canonicalize_mps() returns, so both paths share
    one convert_file()/write_meta() implementation.

    Unlike MPS, this schema has no variable-bounds concept to
    canonicalize away (it's x >= 0 by construction) and no bound-shift
    objective constant -- objective_constant is always 0 here.
    """
    df = pd.read_csv(path)
    required = {"type", "row", "col", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: CSV is missing required column(s): {sorted(missing)}")

    constraint_df = df[df["type"].astype(str).str.lower() == "constraint"].copy()
    constraint_df["row"] = pd.to_numeric(constraint_df["row"], errors="coerce")
    constraint_df["col"] = pd.to_numeric(constraint_df["col"], errors="coerce")
    constraint_df["value"] = pd.to_numeric(constraint_df["value"], errors="coerce")
    constraint_df = constraint_df.dropna(subset=["row", "col", "value"])
    if constraint_df.empty:
        raise ValueError(f"{path}: no constraint rows found (type == 'constraint')")

    n_row = int(constraint_df["row"].nunique())
    # Row numbers may not be contiguous from 0 -- remap to 0..n_row-1 in
    # first-seen order so the output matrix has no phantom empty rows.
    row_ids = sorted(constraint_df["row"].unique())
    row_map = {orig: i for i, orig in enumerate(row_ids)}
    constraint_df["row_idx"] = constraint_df["row"].map(row_map)

    n_col = int(constraint_df["col"].max()) + 1

    objective_df = df[df["type"].astype(str).str.lower() == "objective"].copy()
    c = np.zeros(n_col)
    if not objective_df.empty:
        objective_df["col"] = pd.to_numeric(objective_df["col"], errors="coerce")
        objective_df["value"] = pd.to_numeric(objective_df["value"], errors="coerce")
        objective_df = objective_df.dropna(subset=["col", "value"])
        for _, r in objective_df.iterrows():
            c[int(r["col"])] = float(r["value"])
    else:
        print(
            "WARNING: no 'objective' rows in this CSV -- c is all zeros. "
            "This is not invented; the input genuinely has no objective vector."
        )

    ub_rows, ub_signs, ub_rhs = [], [], []
    eq_rows = []
    ub_entries = []  # (new_row_idx, col, value * sign)
    eq_entries = []  # (new_row_idx, col, value)
    eq_rhs = {}
    ub_rhs_map = {}
    ub_row_counter = 0

    for orig_row, group in constraint_df.groupby("row_idx", sort=True):
        rhs_vals = pd.to_numeric(group["rhs"], errors="coerce").dropna()
        rhs = float(rhs_vals.iloc[0]) if len(rhs_vals) else 0.0
        senses = group["sense"].dropna()
        sense = str(senses.iloc[0]).strip() if len(senses) else "<="

        if sense == "=":
            new_idx = len(eq_rows)
            eq_rows.append(orig_row)
            eq_rhs[new_idx] = rhs
            for _, r in group.iterrows():
                eq_entries.append((new_idx, int(r["col"]), float(r["value"])))
        elif sense == ">=":
            new_idx = ub_row_counter
            ub_row_counter += 1
            ub_rhs_map[new_idx] = -rhs
            for _, r in group.iterrows():
                ub_entries.append((new_idx, int(r["col"]), -float(r["value"])))
        else:  # "<=" (default)
            new_idx = ub_row_counter
            ub_row_counter += 1
            ub_rhs_map[new_idx] = rhs
            for _, r in group.iterrows():
                ub_entries.append((new_idx, int(r["col"]), float(r["value"])))

    def _build(entries, rhs_map, n_rows):
        if n_rows == 0:
            return None, None
        rows = np.array([e[0] for e in entries])
        cols = np.array([e[1] for e in entries])
        vals = np.array([e[2] for e in entries])
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_col)).tocsr()
        b = np.array([rhs_map[i] for i in range(n_rows)])
        return A, b

    A_ub, b_ub = _build(ub_entries, ub_rhs_map, ub_row_counter)
    A_eq, b_eq = _build(eq_entries, eq_rhs, len(eq_rows))

    return {
        "name": Path(path).stem.upper(),
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "c": c,
        "objective_constant": 0.0,
        "var_names": [f"x{j}" for j in range(n_col)],
        "n_row": ub_row_counter + len(eq_rows),
        "n_col": n_col,
    }


def detect_format(path: str, forced: str | None) -> str:
    if forced:
        return forced
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unrecognized input format '{ext}' for {path}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))} "
            f"(or pass --format explicitly)."
        )
    return SUPPORTED_FORMATS[ext]


def convert_file(input_file, output_file, fmt: str | None = None) -> dict:
    resolved = detect_format(str(input_file), fmt)
    print(f"\nConverting: {input_file}  (detected format: {resolved})")

    if resolved in ("mps", "lp"):
        model = canonicalize_mps(input_file)
    elif resolved == "csv":
        model = canonicalize_csv(input_file)
    else:
        raise ValueError(f"Unhandled format '{resolved}'")

    export_to_gpu_format(
        model["c"], model["A_ub"], model["b_ub"], model["A_eq"], model["b_eq"], output_file
    )
    meta_file = Path(output_file).with_suffix(".meta.json")
    write_meta(meta_file, model)

    nnz = (model["A_ub"].nnz if model["A_ub"] is not None else 0) + \
          (model["A_eq"].nnz if model["A_eq"] is not None else 0)
    size = model["n_row"] * model["n_col"]
    print(f"Output      : {output_file}")
    print(f"Metadata    : {meta_file}")
    print(f"Dimensions  : {model['n_row']} x {model['n_col']}")
    print(f"Nonzeros    : {nnz}")
    print(f"Density     : {nnz / size if size else 0:.8f}")
    print(f"Obj constant: {model['objective_constant']:.6f}")
    print("Conversion successful.")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="input file (.mps, .lp, or .csv)")
    p.add_argument("--out", help="output .txt file (default: same name, .txt extension)")
    p.add_argument(
        "--format", choices=["mps", "lp", "csv"], default=None,
        help="force the input format instead of detecting it from the file extension",
    )
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    output_path = Path(args.out) if args.out else input_path.with_suffix(".txt")
    convert_file(input_path, output_path, fmt=args.format)


if __name__ == "__main__":
    main()
