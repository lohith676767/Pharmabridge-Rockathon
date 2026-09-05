"""
CPU baseline solver for the shared GPU-format (sparse COO) LP input.

Reads the format defined in gpu_format.py:

Line 1: M N nnz
Line 2: c            (N values)
Line 3: b            (M values -- b_ub rows then b_eq rows)
Line 4: row_type     (M values -- 0 = "<=", 1 = "=")
Remaining nnz lines: row col value  (0-indexed)

Solves the LP using scipy.optimize.linprog with the HiGHS solver on the
CPU, passing the constraint matrices in as sparse (scipy.sparse) --
linprog accepts sparse A_ub/A_eq directly, no dense conversion needed
anywhere in this pipeline.

If a companion "<input>.meta.json" file exists (written by mps_to_txt.py
for problems converted from real Netlib .mps files), its
"objective_constant" is added to the solved objective. That constant
comes from the variable-bound substitution mps_to_txt.py performs to fit
the model into this format's x >= 0 -only shape -- without adding it
back, the reported objective would be wrong for any Netlib problem with
non-zero variable bounds. Synthetic problems (no meta file) are
unaffected -- the constant defaults to 0.

Usage:
    python cpu_solve.py --input matrix_input.txt --out cpu_result.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scipy.optimize import linprog

from gpu_format import read_gpu_format


def read_objective_constant(input_file: str) -> float:
    """Read the objective_constant from a companion <input>.meta.json, if present."""
    meta_path = Path(input_file).with_suffix(".meta.json")
    if not meta_path.exists():
        return 0.0
    with open(meta_path) as f:
        meta = json.load(f)
    return float(meta.get("objective_constant", 0.0))


def solve_on_cpu(c, A_ub, b_ub, A_eq, b_eq):
    """
    Solve the LP using SciPy HiGHS on the CPU.

    The problem is:

        minimize     c^T x

        subject to   A_ub x <= b_ub
                     A_eq x  = b_eq
                     x >= 0

    Returns:
        result, solve_time
    """

    print("Starting CPU solver...")

    start_time = time.perf_counter()

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs"
    )

    end_time = time.perf_counter()

    solve_time = end_time - start_time

    return result, solve_time


def save_result(result, solve_time, objective_constant, output_file):
    """
    Save CPU solver results as JSON.
    """

    if result.success:

        data = {
            "success": True,
            "status": int(result.status),
            "message": result.message,
            "objective_value": float(result.fun) + objective_constant,
            "objective_constant": objective_constant,
            "solve_time_seconds": float(solve_time),
            "solution_vector": result.x.tolist()
        }

    else:

        data = {
            "success": False,
            "status": int(result.status),
            "message": result.message,
            "objective_value": None,
            "objective_constant": objective_constant,
            "solve_time_seconds": float(solve_time),
            "solution_vector": None
        }

    with open(output_file, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        type=str,
        default="matrix_input.txt",
        help="input matrix file"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="cpu_result.json",
        help="output JSON result file"
    )

    args = parser.parse_args()

    print()
    print("Reading input...")
    print("----------------")

    c, A_ub, b_ub, A_eq, b_eq = read_gpu_format(args.input)

    n_col = len(c)
    n_ub = A_ub.shape[0] if A_ub is not None else 0
    n_eq = A_eq.shape[0] if A_eq is not None else 0

    objective_constant = read_objective_constant(args.input)

    print(f"Input file   : {args.input}")
    print(f"Variables    : {n_col}")
    print(f"<= rows      : {n_ub}")
    print(f"=  rows      : {n_eq}")
    if objective_constant:
        print(f"Obj constant : {objective_constant} (from companion .meta.json)")

    result, solve_time = solve_on_cpu(
        c,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
    )

    print()
    print("CPU RESULT")
    print("----------")

    if result.success:

        print("Status       : SUCCESS")
        print(
            f"Objective    : {result.fun + objective_constant:.10f}"
        )
        print(
            f"Solve time   : {solve_time:.6f} seconds"
        )

    else:

        print("Status       : FAILED")
        print("Message      :", result.message)
        print(
            f"Solve time   : {solve_time:.6f} seconds"
        )

    save_result(
        result,
        solve_time,
        objective_constant,
        args.out
    )

    print()
    print(
        f"Result saved : {args.out}"
    )
    print()


if __name__ == "__main__":
    main()
