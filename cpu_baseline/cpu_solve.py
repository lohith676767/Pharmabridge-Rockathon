"""
CPU baseline solver for the common solver input format.

Reads the common solver input format:

Line 1: M N
Line 2: C (N values)
Line 3: A (M*N values, flattened row-major)
Line 4: B (M values)

Solves the LP using scipy.optimize.linprog with the HiGHS
solver on the CPU.

If a companion "<input>.meta.json" file exists (written by mps_to_txt.py
for problems converted from real Netlib .mps files), its
"objective_constant" is added to the solved objective. That constant
comes from the variable-bound substitution mps_to_txt.py performs to fit
the model into this format's restricted A x <= b, x >= 0 shape -- without
adding it back, the reported objective would be wrong for any Netlib
problem with non-zero variable bounds. Synthetic problems (no meta file)
are unaffected -- the constant defaults to 0.

Usage:
    python cpu_solve.py --input matrix_input.txt --out cpu_result.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def read_matrix_input(filename: str):
    """
    Read the common matrix_input.txt format.

    Format:

        Line 1:
            M N

        Line 2:
            C

        Line 3:
            A flattened in row-major order

        Line 4:
            B

    Returns:
        A, b, c
    """

    with open(filename, "r") as f:

        # -------------------------------------------------
        # Line 1: M N
        # -------------------------------------------------

        first_line = f.readline().strip()

        if not first_line:
            raise ValueError(
                "Input file is empty."
            )

        dimensions = first_line.split()

        if len(dimensions) != 2:
            raise ValueError(
                "First line must contain exactly two values: M N"
            )

        M, N = map(int, dimensions)

        if M <= 0 or N <= 0:
            raise ValueError(
                "M and N must be greater than zero."
            )

        # -------------------------------------------------
        # Line 2: C
        # -------------------------------------------------

        c_line = f.readline().strip()

        if not c_line:
            raise ValueError(
                "Missing C vector."
            )

        c = np.array(
            list(map(float, c_line.split())),
            dtype=np.float64
        )

        # -------------------------------------------------
        # Line 3: A flattened
        # -------------------------------------------------

        A_line = f.readline().strip()

        if not A_line:
            raise ValueError(
                "Missing A matrix."
            )

        A_flat = np.array(
            list(map(float, A_line.split())),
            dtype=np.float64
        )

        # -------------------------------------------------
        # Line 4: B
        # -------------------------------------------------

        b_line = f.readline().strip()

        if not b_line:
            raise ValueError(
                "Missing B vector."
            )

        b = np.array(
            list(map(float, b_line.split())),
            dtype=np.float64
        )

    # -----------------------------------------------------
    # Validate dimensions
    # -----------------------------------------------------

    expected_c = N
    expected_A = M * N
    expected_b = M

    if len(c) != expected_c:
        raise ValueError(
            f"C contains {len(c)} values, "
            f"but expected {expected_c}."
        )

    if len(A_flat) != expected_A:
        raise ValueError(
            f"A contains {len(A_flat)} values, "
            f"but expected {expected_A} "
            f"({M} × {N})."
        )

    if len(b) != expected_b:
        raise ValueError(
            f"B contains {len(b)} values, "
            f"but expected {expected_b}."
        )

    # -----------------------------------------------------
    # Convert flattened A back to M × N
    # -----------------------------------------------------

    A = A_flat.reshape(
        (M, N),
        order="C"
    )

    return A, b, c


def read_objective_constant(input_file: str) -> float:
    """Read the objective_constant from a companion <input>.meta.json, if present."""
    meta_path = Path(input_file).with_suffix(".meta.json")
    if not meta_path.exists():
        return 0.0
    with open(meta_path) as f:
        meta = json.load(f)
    return float(meta.get("objective_constant", 0.0))


def solve_on_cpu(A, b, c):
    """
    Solve the LP using SciPy HiGHS on the CPU.

    The problem is:

        minimize     c^T x

        subject to   A x <= b
                     x >= 0

    Returns:
        result, solve_time
    """

    print("Starting CPU solver...")

    # -----------------------------------------------------
    # Start timing ONLY around the solver
    # -----------------------------------------------------

    start_time = time.perf_counter()

    result = linprog(
        c,
        A_ub=A,
        b_ub=b,
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

    # -----------------------------------------------------
    # Command-line arguments
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Read matrix
    # -----------------------------------------------------

    print()
    print("Reading input...")
    print("----------------")

    A, b, c = read_matrix_input(
        args.input
    )

    M, N = A.shape

    objective_constant = read_objective_constant(args.input)

    print(f"Input file   : {args.input}")
    print(f"Constraints  : {M}")
    print(f"Variables    : {N}")
    print(f"Matrix shape : {A.shape}")
    if objective_constant:
        print(f"Obj constant : {objective_constant} (from companion .meta.json)")

    # -----------------------------------------------------
    # Solve
    # -----------------------------------------------------

    result, solve_time = solve_on_cpu(
        A,
        b,
        c
    )

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

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
