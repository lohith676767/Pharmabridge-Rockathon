"""
MPS -> TXT canonicalizer for the GPU dense LP solver, built on highspy.

HiGHS can natively represent equality rows, ranged rows, >= rows, variable
lower/upper bounds, and free variables. The GPU solver's 4-line TXT format
cannot represent any of that -- it only understands:

    minimize c^T x
    subject to:
        A x <= b
        x >= 0

So this script uses highspy (the official HiGHS Python bindings) to parse
the MPS file correctly -- offloading every MPS edge case (OBJSENSE,
RANGES, free-format quirks, MARKER sections) to a battle-tested library --
and then canonicalizes the resulting model into that exact restricted form
*without changing the underlying optimization problem*.

Canonicalization performed
---------------------------
1. Row bounds (row_lower <= A_row x <= row_upper) are split into one or
   two "<=" rows:
       - equality (lo == hi):        A x <= hi   AND   -A x <= -lo
       - only a lower bound (>=):    -A x <= -lo
       - only an upper bound (<=):    A x <= hi
       - ranged (both finite):        A x <= hi   AND   -A x <= -lo

2. Variable bounds are removed by substitution, so every new variable is
   simply y >= 0:
       - lo finite (most common case, lo == 0 already just passes through):
             x = lo + y             (y >= 0)
             if hi finite: add row   y <= hi - lo
       - lo = -inf, hi finite (MI/UP-only bound):
             x = hi - y             (y >= 0)
       - lo = -inf, hi = +inf (fully free variable):
             x = y_pos - y_neg      (y_pos, y_neg >= 0)  -- adds one extra column

   Substituting shifts the RHS (A x <= b  =>  A(shift + T y) <= b
   =>  (A T) y <= b - A(shift)) and adds a constant term to the objective
   (since c^T x = c^T shift + c^T T y). The 4-line TXT format has no slot
   for that constant, so it is written to a companion ``<name>.meta.json``
   file: the TRUE optimal objective of the original MPS problem is
   ``objective_from_txt_solve + objective_constant``.

Usage
-----
    python mps_to_txt.py afiro.mps
    python mps_to_txt.py afiro.mps --out afiro.txt
    python mps_to_txt.py --input-dir netlib --output-dir txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import highspy
import numpy as np


def _dense_columns(a_matrix, n_row: int, n_col: int) -> np.ndarray:
    """Expand HiGHS's column-wise sparse matrix into a dense n_row x n_col array."""
    A = np.zeros((n_row, n_col))
    start = a_matrix.start_
    index = a_matrix.index_
    value = a_matrix.value_
    for j in range(n_col):
        for k in range(start[j], start[j + 1]):
            A[index[k], j] = value[k]
    return A


def canonicalize_mps(path: str) -> dict:
    """Read an MPS file with highspy and canonicalize it to A x <= b, x >= 0.

    Returns a dict with keys: A, b, c, objective_constant, var_names, name.
    """
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(str(path))
    if status != highspy.HighsStatus.kOk:
        raise ValueError(f"highspy failed to read {path}: status={status}")

    lp = h.getLp()
    n_col = lp.num_col_
    n_row = lp.num_row_

    A_orig = _dense_columns(lp.a_matrix_, n_row, n_col)
    c = np.array(lp.col_cost_, dtype=float)
    offset = float(lp.offset_)
    if lp.sense_ == highspy.ObjSense.kMaximize:
        c = -c
        offset = -offset

    col_lower = np.array(lp.col_lower_, dtype=float)
    col_upper = np.array(lp.col_upper_, dtype=float)
    row_lower = np.array(lp.row_lower_, dtype=float)
    row_upper = np.array(lp.row_upper_, dtype=float)

    orig_names = list(lp.col_names_) if lp.col_names_ else [f"C{j}" for j in range(n_col)]

    # ---- 1. Canonicalize row bounds into A_can x <= b_can -----------------
    A_can_rows, b_can = [], []
    for i in range(n_row):
        lo, hi = row_lower[i], row_upper[i]
        row = A_orig[i, :]
        if lo == hi:
            A_can_rows.append(row); b_can.append(hi)
            A_can_rows.append(-row); b_can.append(-lo)
        elif np.isinf(hi):
            A_can_rows.append(-row); b_can.append(-lo)
        elif np.isinf(lo):
            A_can_rows.append(row); b_can.append(hi)
        else:
            A_can_rows.append(row); b_can.append(hi)
            A_can_rows.append(-row); b_can.append(-lo)
    A_can = np.array(A_can_rows) if A_can_rows else np.zeros((0, n_col))
    b_can = np.array(b_can)

    # ---- 2. Canonicalize variable bounds via substitution ------------------
    # Each original column j maps to one or two new nonnegative columns.
    # x_j = shift_j + sign_j * y_k   (normal / negated),  or
    # x_j = y_pos - y_neg            (free)
    new_names = []
    new_cols = []          # (orig_j, sign) per new column
    shift = np.zeros(n_col)
    extra_bound_rows = []  # (new_col_index, rhs) for finite-width vars

    for j in range(n_col):
        lo, hi = col_lower[j], col_upper[j]
        if np.isinf(lo) and np.isinf(hi):
            new_cols.append((j, 1.0)); new_names.append(orig_names[j] + "_pos")
            new_cols.append((j, -1.0)); new_names.append(orig_names[j] + "_neg")
        elif np.isinf(lo):  # hi finite, lo = -inf
            shift[j] = hi
            new_cols.append((j, -1.0)); new_names.append(orig_names[j])
        else:  # lo finite (the common case, including lo == 0)
            shift[j] = lo
            k = len(new_cols)
            new_cols.append((j, 1.0)); new_names.append(orig_names[j])
            if not np.isinf(hi):
                extra_bound_rows.append((k, hi - lo))

    # A_new[:, k] = sign * A_can[:, j] for each new column k -> old column j.
    # This is mathematically A_can @ T_signed for a signed selection matrix
    # T_signed, but T_signed has exactly one nonzero per column -- building
    # it densely (n_col x new_n) wastes huge amounts of memory for large
    # problems (e.g. ~17.7 GiB for a ~49k-variable problem) for no reason,
    # so select/flip columns directly instead of ever materializing it.
    new_n = len(new_cols)
    orig_cols = np.array([j for j, _ in new_cols])
    signs = np.array([sign for _, sign in new_cols])

    A_new = A_can[:, orig_cols] * signs
    b_new = b_can - A_can @ shift

    for k, rhs in extra_bound_rows:
        row = np.zeros(new_n)
        row[k] = 1.0
        A_new = np.vstack([A_new, row])
        b_new = np.append(b_new, rhs)

    c_new = np.zeros(new_n)
    for k, (j, sign) in enumerate(new_cols):
        c_new[k] = c[j] * sign
    objective_constant = offset + float(c @ shift)

    return {
        "name": lp.model_name_ or Path(path).stem.upper(),
        "A": A_new,
        "b": b_new,
        "c": c_new,
        "objective_constant": objective_constant,
        "var_names": new_names,
        "n_row": A_new.shape[0],
        "n_col": new_n,
    }


def write_txt(filename, model: dict) -> None:
    """Write the 4-line M N / c / A / b format expected by the GPU solver."""
    A, b, c = model["A"], model["b"], model["c"]
    M, N = A.shape
    with open(filename, "w") as f:
        f.write(f"{M} {N}\n")
        f.write(" ".join(f"{v:.17g}" for v in c) + "\n")
        f.write(" ".join(f"{v:.17g}" for v in A.flatten(order="C")) + "\n")
        f.write(" ".join(f"{v:.17g}" for v in b) + "\n")


def write_meta(filename, model: dict) -> None:
    """Write the sidecar metadata (objective constant, variable mapping)."""
    with open(filename, "w") as f:
        json.dump({
            "name": model["name"],
            "objective_constant": model["objective_constant"],
            "note": "true_optimal_objective = objective_value_from_txt_solve + objective_constant",
            "var_names": model["var_names"],
            "n_row": model["n_row"],
            "n_col": model["n_col"],
        }, f, indent=2)


def convert_file(input_file, output_file) -> dict:
    print(f"\nConverting: {input_file}")
    model = canonicalize_mps(input_file)
    write_txt(output_file, model)
    meta_file = Path(output_file).with_suffix(".meta.json")
    write_meta(meta_file, model)

    nonzeros = int(np.count_nonzero(model["A"]))
    size = model["A"].size
    print(f"Output      : {output_file}")
    print(f"Metadata    : {meta_file}")
    print(f"Dimensions  : {model['n_row']} x {model['n_col']}")
    print(f"Nonzeros    : {nonzeros}")
    print(f"Density     : {nonzeros / size if size else 0:.8f}")
    print(f"Obj constant: {model['objective_constant']:.6f}")
    print("Conversion successful.")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="input .mps file")
    p.add_argument("--out", help="output .txt file")
    p.add_argument("--input-dir", help="directory containing MPS files")
    p.add_argument("--output-dir", help="directory for converted TXT files")
    args = p.parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        output_path = Path(args.out) if args.out else input_path.with_suffix(".txt")
        convert_file(input_path, output_path)
        return

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        output_dir = Path(args.output_dir if args.output_dir else "converted_txt")
        output_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_dir.glob("*.mps"))
        if not files:
            print("No .mps files found.")
            return

        print(f"Found {len(files)} MPS files.")
        for input_file in files:
            output_file = output_dir / input_file.with_suffix(".txt").name
            try:
                convert_file(input_file, output_file)
            except Exception as e:
                print(f"FAILED: {input_file}")
                print(f"Reason: {e}")
        print("\nBatch conversion completed.")
        return

    p.print_help()


if __name__ == "__main__":
    main()
