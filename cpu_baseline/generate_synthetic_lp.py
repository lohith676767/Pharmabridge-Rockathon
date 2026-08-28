"""
Generate synthetic DENSE LP problems that mimic petroleum-blending
"what-if" scenarios, for benchmarking cuSOLVER (GPU) against the
scipy/HiGHS CPU baseline on matrices Netlib doesn't provide (Netlib's
real-world sets are mostly sparse).

Usage:
    python generate_synthetic_lp.py --vars 500 --constraints 300 --seed 42 \
        --out synthetic_500x300.npz
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
    """Build a random dense maximize-profit LP, cast to scipy's minimize form:
    min (-c)^T x  s.t. A_ub x <= b_ub, x >= 0.

    Modeled loosely on petroleum blending: variables are blend-stock volumes
    with a per-unit profit, rows are capacity constraints on shared
    resources (crude intake, refinery throughput, etc). All coefficients are
    positive and capacities are tight enough that the optimum sits on the
    constraint boundary rather than trivially at x=0, so the problem
    actually exercises the solver.
    """
    rng = np.random.default_rng(seed)

    profit = rng.uniform(10, 100, size=n_vars)
    c = -profit  # linprog minimizes, so maximize profit == minimize -profit

    A_ub = rng.uniform(0.1, 5.0, size=(n_constraints, n_vars))
    if density < 1.0:
        mask = rng.random(size=(n_constraints, n_vars)) < density
        A_ub *= mask
        A_ub[A_ub.sum(axis=1) == 0, 0] = rng.uniform(0.1, 5.0)  # keep rows non-degenerate

    # Tight-ish capacities: a fraction of what it'd take to run every
    # variable at a generous per-variable cap, so several constraints bind.
    per_var_cap = rng.uniform(5, 20, size=n_vars)
    b_ub = (A_ub @ per_var_cap) * rng.uniform(0.3, 0.6, size=n_constraints)

    bounds = [(0, None)] * n_vars

    return {
        "name": f"synthetic_{n_vars}x{n_constraints}_seed{seed}",
        "c": c,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": None,
        "b_eq": None,
        "bounds": bounds,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vars", type=int, default=500, help="number of variables (columns)")
    p.add_argument("--constraints", type=int, default=300, help="number of <= constraints (rows)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--density", type=float, default=1.0, help="fraction of nonzero A entries (1.0 = fully dense)")
    p.add_argument("--out", type=str, required=True, help="output .npz path")
    args = p.parse_args()

    problem = generate_dense_blending_lp(
        n_vars=args.vars,
        n_constraints=args.constraints,
        seed=args.seed,
        density=args.density,
    )

    np.savez(
        args.out,
        name=problem["name"],
        c=problem["c"],
        A_ub=problem["A_ub"],
        b_ub=problem["b_ub"],
        lower=np.array([b[0] if b[0] is not None else -np.inf for b in problem["bounds"]]),
        upper=np.array([b[1] if b[1] is not None else np.inf for b in problem["bounds"]]),
    )
    print(f"Saved {problem['name']} -> {args.out}  "
          f"(A_ub shape={problem['A_ub'].shape}, density={args.density})")


if __name__ == "__main__":
    main()
