"""
Commercial-solver baseline: solves the shared GPU-format (sparse COO) LP
input with Gurobi, mirroring cpu_solve.py's interface and output shape
exactly, so it's a drop-in third comparison point alongside the CPU
(HiGHS) and GPU (PDHG) results -- same JSON fields, same input format,
no changes needed anywhere else in the pipeline.

Why this exists: the SIH problem statement's background specifically
names commercial solvers (CPLEX, Gurobi, Xpress) as the incumbents being
compared against -- an open-source-only comparison (HiGHS) satisfies the
letter of "at least one established commercial or open-source solver"
but not the actual point being made. This gives the real number.

Requires a Gurobi license (a free academic license is enough --
see https://www.gurobi.com/academia/academic-program-and-licenses/).
Without an activated license, Gurobi's default install still works for
small problems (its size-limited evaluation license), which is enough
to sanity-check this script, but not for realistic benchmark sizes.

Usage:
    python gurobi_solve.py --input matrix_input.txt --out gurobi_result.json
"""

from __future__ import annotations

import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from gpu_format import read_gpu_format
from cpu_solve import read_objective_constant


def solve_with_gurobi(c, A_ub, b_ub, A_eq, b_eq):
    """
    Solve the LP with Gurobi.

    The problem is:

        minimize     c^T x

        subject to   A_ub x <= b_ub
                     A_eq x  = b_eq
                     x >= 0

    Returns:
        (success, status_name, objective, x, solve_time)
    """
    n = len(c)

    print("Starting Gurobi solver...")

    start_time = time.perf_counter()

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)  # quiet -- match cpu_solve.py's clean console output
        env.start()
        with gp.Model(env=env) as model:
            x = model.addMVar(shape=n, lb=0.0, name="x")
            model.setObjective(c @ x, GRB.MINIMIZE)

            if A_ub is not None:
                model.addConstr(A_ub @ x <= b_ub, name="ub")
            if A_eq is not None:
                model.addConstr(A_eq @ x == b_eq, name="eq")

            model.optimize()

            solve_time = time.perf_counter() - start_time

            if model.Status == GRB.OPTIMAL:
                return True, "OPTIMAL", model.ObjVal, x.X.tolist(), solve_time
            else:
                # GRB status codes: https://docs.gurobi.com/current/refman/optimization_status_codes.html
                status_name = {
                    GRB.INFEASIBLE: "INFEASIBLE",
                    GRB.UNBOUNDED: "UNBOUNDED",
                    GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
                    GRB.TIME_LIMIT: "TIME_LIMIT",
                }.get(model.Status, f"STATUS_{model.Status}")
                return False, status_name, None, None, solve_time


def save_result(success, status_name, objective, x, solve_time, objective_constant, output_file):
    if success:
        data = {
            "success": True,
            "status": status_name,
            "message": "Optimal solution found.",
            "objective_value": float(objective) + objective_constant,
            "objective_constant": objective_constant,
            "solve_time_seconds": float(solve_time),
            "solution_vector": x,
        }
    else:
        data = {
            "success": False,
            "status": status_name,
            "message": f"Gurobi did not find an optimal solution ({status_name}).",
            "objective_value": None,
            "objective_constant": objective_constant,
            "solve_time_seconds": float(solve_time),
            "solution_vector": None,
        }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default="matrix_input.txt", help="input matrix file (GPU COO format)")
    parser.add_argument("--out", type=str, default="gurobi_result.json", help="output JSON result file")
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

    success, status_name, objective, x, solve_time = solve_with_gurobi(c, A_ub, b_ub, A_eq, b_eq)

    print()
    print("GUROBI RESULT")
    print("-------------")

    if success:
        print("Status       : SUCCESS")
        print(f"Objective    : {objective + objective_constant:.10f}")
        print(f"Solve time   : {solve_time:.6f} seconds")
    else:
        print("Status       : FAILED")
        print("Message      :", status_name)
        print(f"Solve time   : {solve_time:.6f} seconds")

    save_result(success, status_name, objective, x, solve_time, objective_constant, args.out)

    print()
    print(f"Result saved : {args.out}")
    print()


if __name__ == "__main__":
    main()
