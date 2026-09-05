"""
Generate synthetic BLOCK-DIAGONAL LP problems that mimic petroleum-blending
"what-if" scenarios, for benchmarking the GPU solver against the
scipy/HiGHS CPU baseline on matrices Netlib doesn't provide.

Domain-specific sparsity modeling
----------------------------------
Real refineries don't have every chemical interacting with every catalyst
-- a given catalyst bed only processes a specific subset of feedstocks.
So instead of scattering random nonzeros across the whole constraint
matrix, variables and constraints are partitioned into blocks (each
block = one process unit's chemicals + the catalyst/capacity constraints
that apply to it). Coefficients are only generated *within* a block --
everywhere else is structurally zero, not just randomly dropped.

A small number of "coupling" rows span every block (shared upstream
resources like total crude intake, or shared utilities) so the problem
stays one connected LP instead of N independent sub-problems, matching
how real refineries share some resources across otherwise-separate units.

The whole matrix is assembled with scipy.sparse.block_diag directly --
never as a dense array -- so this scales to very large variable counts
without the memory blowup a dense matrix would need.

Output: the shared GPU-format sparse COO file (see gpu_format.py).

Usage:
    python generate_synthetic_lp.py --vars 500 --constraints 300 --seed 42 --out matrix_input.txt
    python generate_synthetic_lp.py --vars 500 --constraints 300 --blocks 10 --coupling 0.05 --out matrix_input.txt
"""

from __future__ import annotations

import argparse
import numpy as np
from scipy import sparse

from gpu_format import export_to_gpu_format


def generate_block_diagonal_blending_lp(
    n_vars: int,
    n_constraints: int,
    seed: int = 0,
    density: float = 1.0,
    n_blocks: int | None = None,
    coupling_fraction: float = 0.05,
) -> dict:
    """
    Build a random BLOCK-DIAGONAL maximize-profit LP, cast to scipy's
    minimize form:

        min (-c)^T x

        subject to:
            A_ub x <= b_ub
            x >= 0

    Modeled on petroleum blending, with realistic sparsity:

    - Variables represent blend-stock volumes, grouped into n_blocks
      "process units" (e.g. a catalyst bed processing a specific set
      of feedstocks).
    - Each block's constraint rows only reference that block's own
      variables -- other blocks are structurally zero, not randomly
      sparse. This is the block-diagonal pattern, assembled directly
      with scipy.sparse.block_diag (no dense intermediate).
    - A small number of "coupling" rows span every block, representing
      shared resources (total crude intake, shared utilities), keeping
      the problem one connected LP instead of independent sub-problems.
      Built directly as a sparse matrix (scipy.sparse.random), never
      dense, so this stays scalable even for very large n_vars.
    - Coefficients are positive; capacities are tight enough to produce
      a non-trivial optimum.

    Parameters
    ----------
    n_blocks : number of process-unit blocks. Defaults to roughly one
        block per 25 variables (a plausible unit size), clamped so
        every block gets at least one variable and one constraint row.
    coupling_fraction : fraction of n_constraints reserved as
        cross-block coupling rows (default 5%).
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Decide block layout
    # ---------------------------------------------------------

    if n_blocks is None:
        n_blocks = max(1, round(n_vars / 25))

    n_blocks = max(1, min(n_blocks, n_vars))

    n_coupling = (
        max(0, round(n_constraints * coupling_fraction))
        if n_blocks > 1
        else 0
    )
    n_block_constraints = n_constraints - n_coupling

    # Every block needs at least one constraint row; if there aren't
    # enough rows to go around, give up on coupling rows instead of
    # starving a block down to zero constraints.
    if n_block_constraints < n_blocks:
        n_block_constraints = n_constraints
        n_coupling = 0

    var_groups = np.array_split(np.arange(n_vars), n_blocks)
    row_groups = np.array_split(np.arange(n_block_constraints), n_blocks)

    # ---------------------------------------------------------
    # 2. Generate profit for each variable
    # ---------------------------------------------------------

    profit = rng.uniform(10, 100, size=n_vars)

    # scipy.optimize.linprog performs MINIMIZATION.
    # To maximize profit:  minimize (-profit)^T x
    c = -profit

    # ---------------------------------------------------------
    # 3. Build each block (small, so a dense intermediate per block is
    #    fine even for huge overall problems) and assemble them into
    #    one sparse block-diagonal matrix.
    # ---------------------------------------------------------

    blocks = []
    b_parts = []

    for vars_in_block, rows_in_block in zip(var_groups, row_groups):

        if len(vars_in_block) == 0 or len(rows_in_block) == 0:
            continue

        block = rng.uniform(
            0.1, 5.0, size=(len(rows_in_block), len(vars_in_block))
        )

        # Optional intra-block sparsity control (on top of the
        # structural block-diagonal sparsity from the layout itself).
        if density < 1.0:
            mask = rng.random(size=block.shape) < density
            block *= mask
            for r in range(block.shape[0]):
                if block[r].sum() == 0:
                    block[r, 0] = rng.uniform(0.1, 5.0)

        # Tight-ish per-block capacities: cap each row well below what
        # running every variable at a generous per-variable cap would
        # need, so the block's own constraints actually bind.
        per_var_cap = rng.uniform(5, 20, size=len(vars_in_block))
        b_block = (block @ per_var_cap) * rng.uniform(
            0.3, 0.6, size=len(rows_in_block)
        )

        blocks.append(sparse.csr_matrix(block))
        b_parts.append(b_block)

    A_blocks = sparse.block_diag(blocks, format="csr") if blocks else sparse.csr_matrix((0, n_vars))
    b_blocks = np.concatenate(b_parts) if b_parts else np.zeros(0)

    # ---------------------------------------------------------
    # 4. Coupling rows -- shared resources spanning every block.
    #    Built directly as a sparse random matrix (never dense), so
    #    this stays memory-safe even for very large n_vars.
    # ---------------------------------------------------------

    if n_coupling > 0:
        coupling_A = sparse.random(
            n_coupling,
            n_vars,
            density=density,
            random_state=rng,
            data_rvs=lambda size: rng.uniform(0.05, 1.0, size=size),
            format="csr",
        )
        per_var_cap_all = rng.uniform(5, 20, size=n_vars)
        coupling_b = (coupling_A @ per_var_cap_all) * rng.uniform(
            0.3, 0.6, size=n_coupling
        )
    else:
        coupling_A = sparse.csr_matrix((0, n_vars))
        coupling_b = np.zeros(0)

    A_ub = sparse.vstack([A_blocks, coupling_A], format="csr")
    b_ub = np.concatenate([b_blocks, coupling_b])

    return {
        "name": (
            f"synthetic_block_{n_vars}x{n_constraints}"
            f"_blocks{n_blocks}_seed{seed}"
        ),
        "c": c,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "n_blocks": n_blocks,
        "n_coupling": n_coupling,
    }


def main():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--vars", type=int, default=500, help="number of variables (columns)")
    p.add_argument("--constraints", type=int, default=300, help="number of constraints (rows)")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument(
        "--density", type=float, default=1.0,
        help="fraction of nonzero entries WITHIN each block (1.0 = fully dense inside a block)"
    )
    p.add_argument(
        "--blocks", type=int, default=0,
        help="number of block-diagonal process-unit groups (0 = auto, roughly one block per 25 variables)"
    )
    p.add_argument(
        "--coupling", type=float, default=0.05,
        help="fraction of --constraints reserved as cross-block coupling rows"
    )
    p.add_argument("--out", type=str, required=True, help="output GPU-format .txt file")

    args = p.parse_args()

    if args.vars <= 0:
        raise ValueError("--vars must be greater than 0")
    if args.constraints <= 0:
        raise ValueError("--constraints must be greater than 0")
    if not 0 < args.density <= 1:
        raise ValueError("--density must be greater than 0 and less than or equal to 1")
    if args.blocks < 0:
        raise ValueError("--blocks must be 0 (auto) or greater")
    if not 0 <= args.coupling < 1:
        raise ValueError("--coupling must be in [0, 1)")

    problem = generate_block_diagonal_blending_lp(
        n_vars=args.vars,
        n_constraints=args.constraints,
        seed=args.seed,
        density=args.density,
        n_blocks=(args.blocks or None),
        coupling_fraction=args.coupling,
    )

    export_to_gpu_format(
        problem["c"], problem["A_ub"], problem["b_ub"], None, None, args.out
    )

    nnz = problem["A_ub"].nnz
    size = problem["A_ub"].shape[0] * problem["A_ub"].shape[1]

    print()
    print("Synthetic block-diagonal LP generated successfully.")
    print("-----------------------------------------------------")
    print(f"Problem      : {problem['name']}")
    print(f"Constraints  : {args.constraints}")
    print(f"Variables    : {args.vars}")
    print(f"Matrix shape : {problem['A_ub'].shape}")
    print(f"Blocks       : {problem['n_blocks']}")
    print(f"Coupling rows: {problem['n_coupling']}")
    print(f"Overall density : {nnz / size:.6f}  ({nnz} / {size} nonzero)")
    print(f"Output file  : {args.out}")
    print()


if __name__ == "__main__":
    main()
