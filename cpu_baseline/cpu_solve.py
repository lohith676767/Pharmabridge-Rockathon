"""
CPU baseline LP solver for SIH26119.

Solves an LP with scipy.optimize.linprog(method="highs") — CPU-only,
matching whatever matrix (Netlib .mps or a synthetic .npz) the GPU
(cuSOLVER) team is timing — and writes a verification package
(objective value + solution vector + wall-clock time) so the UI
dashboard can check the GPU result's numerical correctness against it.

Usage:
    # Netlib .mps problem
    python cpu_solve.py --mps path/to/problem.mps --out results/afiro_cpu.json

    # Synthetic .npz problem (from generate_synthetic_lp.py)
    python cpu_solve.py --npz synthetic_500x300.npz --out results/synth_cpu.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.optimize import linprog

from mps_reader import read_mps


def load_problem(args: argparse.Namespace) -> dict:
    if args.mps:
        return read_mps(args.mps)

    if args.npz:
        data = np.load(args.npz, allow_pickle=True)
        lower = data["lower"]
        upper = data["upper"]
        bounds = [
            (None if lo == -np.inf else lo, None if hi == np.inf else hi)
            for lo, hi in zip(lower, upper)
        ]
        return {
            "name": str(data["name"]),
            "c": data["c"],
            "A_ub": data["A_ub"],
            "b_ub": data["b_ub"],
            "A_eq": None,
            "b_eq": None,
            "bounds": bounds,
        }

    raise ValueError("Provide either --mps or --npz")


def solve_cpu_baseline(problem: dict) -> dict:
    """Solve with HiGHS on CPU and time it. Returns a verification package."""
    start = time.perf_counter()
    result = linprog(
        c=problem["c"],
        A_ub=problem["A_ub"],
        b_ub=problem["b_ub"],
        A_eq=problem["A_eq"],
        b_eq=problem["b_eq"],
        bounds=problem["bounds"],
        method="highs",
    )
    elapsed = time.perf_counter() - start

    return {
        "problem_name": problem["name"],
        "status": result.status,          # 0 = optimal
        "success": bool(result.success),
        "message": result.message,
        "objective_value": float(result.fun) if result.success else None,
        "solution_vector": result.x.tolist() if result.success else None,
        "n_vars": int(problem["c"].shape[0]),
        "n_constraints_ub": int(problem["A_ub"].shape[0]) if problem["A_ub"] is not None else 0,
        "n_constraints_eq": int(problem["A_eq"].shape[0]) if problem["A_eq"] is not None else 0,
        "solve_time_seconds": elapsed,
        "solver": "scipy.optimize.linprog(method='highs')",
        "device": "CPU",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mps", type=str, help="path to a Netlib .mps file")
    src.add_argument("--npz", type=str, help="path to a synthetic .npz file")
    p.add_argument("--out", type=str, required=True, help="output .json verification package")
    args = p.parse_args()

    problem = load_problem(args)
    verification = solve_cpu_baseline(problem)

    with open(args.out, "w") as f:
        json.dump(verification, f, indent=2)

    if verification["success"]:
        print(f"[{verification['problem_name']}] optimal objective = "
              f"{verification['objective_value']:.6f}  "
              f"in {verification['solve_time_seconds']*1000:.2f} ms "
              f"({verification['n_vars']} vars, "
              f"{verification['n_constraints_ub'] + verification['n_constraints_eq']} constraints)")
    else:
        print(f"[{verification['problem_name']}] FAILED: {verification['message']}")

    print(f"Verification package written to {args.out}")


if __name__ == "__main__":
    main()
