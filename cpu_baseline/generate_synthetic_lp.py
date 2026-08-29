"""
Generate synthetic DENSE LP problems that mimic petroleum-blending
"what-if" scenarios, for benchmarking cuSOLVER (GPU) against the
scipy/HiGHS CPU baseline on matrices Netlib doesn't provide.

Output format:

Line 1: M N
Line 2: C (N space-separated floats)
Line 3: A flattened row-major matrix (M*N space-separated floats)
Line 4: B (M space-separated floats)

Usage:
    python synthetic_matrix_generator.py --vars 500 --constraints 300 --seed 42 --out matrix_input.txt
"""

from __future__ import annotations

import argparse
import numpy as np


def generate_dense_blending_lp(
    n_vars: int,
    n_constraints: int,
    seed: int = 0,
    density: float = 1.0,
) -> dict:
    """
    Build a random dense maximize-profit LP, cast to scipy's minimize form:

        min (-c)^T x

        subject to:
            A_ub x <= b_ub
            x >= 0

    Modeled loosely on petroleum blending:

    - Variables represent blend-stock volumes.
    - Each variable has a profit value.
    - Rows represent capacity constraints.
    - Coefficients are positive.
    - Capacities are tight enough to produce a non-trivial optimum.
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Generate profit for each variable
    # ---------------------------------------------------------

    profit = rng.uniform(10, 100, size=n_vars)

    # scipy.optimize.linprog performs MINIMIZATION.
    # To maximize profit:
    #
    #       maximize profit^T x
    #
    # becomes:
    #
    #       minimize (-profit)^T x
    #
    c = -profit

    # ---------------------------------------------------------
    # 2. Generate constraint matrix A
    # ---------------------------------------------------------

    A_ub = rng.uniform(
        0.1,
        5.0,
        size=(n_constraints, n_vars)
    )

    # Optional sparsity control
    if density < 1.0:

        mask = rng.random(
            size=(n_constraints, n_vars)
        ) < density

        A_ub *= mask

        # Make sure every row has at least one
        # non-zero coefficient.
        for row in range(n_constraints):
            if A_ub[row].sum() == 0:
                A_ub[row, 0] = rng.uniform(0.1, 5.0)

    # ---------------------------------------------------------
    # 3. Generate constraint bounds B
    # ---------------------------------------------------------

    # Temporary per-variable upper-cap estimate
    per_var_cap = rng.uniform(
        5,
        20,
        size=n_vars
    )

    # Generate reasonably tight capacities.
    #
    # This prevents the trivial x = 0 solution
    # from being the only interesting result.
    b_ub = (
        A_ub @ per_var_cap
    ) * rng.uniform(
        0.3,
        0.6,
        size=n_constraints
    )

    # ---------------------------------------------------------
    # 4. Variable bounds
    # ---------------------------------------------------------

    bounds = [(0, None)] * n_vars

    return {
        "name": (
            f"synthetic_{n_vars}x"
            f"{n_constraints}_seed{seed}"
        ),
        "c": c,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": None,
        "b_eq": None,
        "bounds": bounds,
    }


def export_to_solver_format(
    A,
    b,
    c,
    filename
):
    """
    Export the LP into the common solver input format.

    File structure:

        Line 1:
            M N

        Line 2:
            C
            N space-separated values

        Line 3:
            A
            M*N space-separated values
            stored in row-major order

        Line 4:
            B
            M space-separated values
    """

    # Number of constraints and variables
    M, N = A.shape

    # ---------------------------------------------------------
    # Validate dimensions before writing
    # ---------------------------------------------------------

    if len(c) != N:
        raise ValueError(
            f"Cost vector has {len(c)} values, "
            f"but expected {N}."
        )

    if len(b) != M:
        raise ValueError(
            f"RHS vector has {len(b)} values, "
            f"but expected {M}."
        )

    if A.size != M * N:
        raise ValueError(
            f"Matrix A contains {A.size} values, "
            f"but expected {M * N}."
        )

    # ---------------------------------------------------------
    # Write the four-line solver format
    # ---------------------------------------------------------

    with open(filename, "w") as f:

        # Line 1: M N
        f.write(f"{M} {N}\n")

        # Line 2: C
        f.write(
            " ".join(map(str, c))
            + "\n"
        )

        # Line 3: A flattened row-major
        #
        # A.flatten() uses row-major order by default.
        f.write(
            " ".join(
                map(str, A.flatten())
            )
            + "\n"
        )

        # Line 4: B
        f.write(
            " ".join(map(str, b))
            + "\n"
        )


def main():

    # ---------------------------------------------------------
    # Command-line arguments
    # ---------------------------------------------------------

    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument(
        "--vars",
        type=int,
        default=500,
        help="number of variables (columns)"
    )

    p.add_argument(
        "--constraints",
        type=int,
        default=300,
        help="number of constraints (rows)"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed"
    )

    p.add_argument(
        "--density",
        type=float,
        default=1.0,
        help=(
            "fraction of nonzero A entries "
            "(1.0 = fully dense)"
        )
    )

    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="output solver input file"
    )

    args = p.parse_args()

    # ---------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------

    if args.vars <= 0:
        raise ValueError(
            "--vars must be greater than 0"
        )

    if args.constraints <= 0:
        raise ValueError(
            "--constraints must be greater than 0"
        )

    if not 0 < args.density <= 1:
        raise ValueError(
            "--density must be greater than 0 "
            "and less than or equal to 1"
        )

    # ---------------------------------------------------------
    # Generate LP
    # ---------------------------------------------------------

    problem = generate_dense_blending_lp(
        n_vars=args.vars,
        n_constraints=args.constraints,
        seed=args.seed,
        density=args.density,
    )

    # ---------------------------------------------------------
    # Export using the common solver format
    # ---------------------------------------------------------

    export_to_solver_format(
        problem["A_ub"],
        problem["b_ub"],
        problem["c"],
        args.out
    )

    # ---------------------------------------------------------
    # Display information
    # ---------------------------------------------------------

    print()
    print("Synthetic LP generated successfully.")
    print("--------------------------------------")
    print(f"Problem      : {problem['name']}")
    print(f"Constraints  : {args.constraints}")
    print(f"Variables    : {args.vars}")
    print(f"Matrix shape : {problem['A_ub'].shape}")
    print(f"Density      : {args.density}")
    print(f"Output file  : {args.out}")
    print()
    print("Format:")
    print("Line 1 -> M N")
    print("Line 2 -> C")
    print("Line 3 -> A (flattened row-major)")
    print("Line 4 -> B")
    print()


if __name__ == "__main__":
    main()