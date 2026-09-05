"""
Commercial-solver baseline: solves the shared GPU-format (sparse COO) LP
input with IBM CPLEX, mirroring cpu_solve.py's/gurobi_solve.py's
interface and output shape exactly, so it's a drop-in comparison point
alongside CPU (HiGHS), GPU (PDHG), and Gurobi results -- same input
format, same JSON output shape, no other tooling needs to change.

Why this exists: the SIH problem statement's background names CPLEX,
Gurobi, and Xpress specifically as the commercial-solver dependency this
project is positioned against. This gives that comparison directly.

Requires the `cplex` Python package. `pip install cplex` gives you IBM's
Community Edition for free with no license file needed at all -- it's
limited to 1000 variables / 1000 constraints, which covers small-to-
medium benchmark problems (e.g. AFIRO, BLEND, SHARE1B) but not large
ones (e.g. ken-07, pds-100). For those, IBM's free Academic Initiative
license removes the size cap (verified by institutional email, not
network IP, unlike Gurobi's academic program) --
see https://www.ibm.com/academic/home

Usage:
    python cplex_solve.py --input matrix_input.txt --out cplex_result.json
"""

from __future__ import annotations

import argparse
import json
import time

import cplex
from cplex import SparsePair

from gpu_format import read_gpu_format
from cpu_solve import read_objective_constant


def _rows_to_sparse_pairs(A_csr):
    """Convert a CSR matrix's rows into a list of cplex.SparsePair, one per row."""
    pairs = []
    indptr, indices, data = A_csr.indptr, A_csr.indices, A_csr.data
    for i in range(A_csr.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        pairs.append(SparsePair(ind=indices[start:end].tolist(), val=data[start:end].tolist()))
    return pairs


def solve_with_cplex(c, A_ub, b_ub, A_eq, b_eq):
    """
    Solve the LP with CPLEX.

    The problem is:

        minimize     c^T x

        subject to   A_ub x <= b_ub
                     A_eq x  = b_eq
                     x >= 0

    Returns:
        (success, status_name, objective, x, solve_time)
    """
    n = len(c)

    print("Starting CPLEX solver...")

    prob = cplex.Cplex()
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)

    prob.objective.set_sense(prob.objective.sense.minimize)
    prob.variables.add(obj=c.tolist(), lb=[0.0] * n, names=[f"x{i}" for i in range(n)])

    if A_ub is not None:
        pairs = _rows_to_sparse_pairs(A_ub.tocsr())
        prob.linear_constraints.add(lin_expr=pairs, senses=["L"] * len(pairs), rhs=b_ub.tolist())

    if A_eq is not None:
        pairs = _rows_to_sparse_pairs(A_eq.tocsr())
        prob.linear_constraints.add(lin_expr=pairs, senses=["E"] * len(pairs), rhs=b_eq.tolist())

    start_time = time.perf_counter()
    prob.solve()
    solve_time = time.perf_counter() - start_time

    status = prob.solution.get_status()
    status_name = prob.solution.status[status]

    is_optimal = status in (
        prob.solution.status.optimal,
        prob.solution.status.optimal_tolerance,
    )

    if is_optimal:
        objective = prob.solution.get_objective_value()
        x = prob.solution.get_values()
        return True, status_name, objective, x, solve_time
    else:
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
            "message": f"CPLEX did not find an optimal solution ({status_name}).",
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
    parser.add_argument("--out", type=str, default="cplex_result.json", help="output JSON result file")
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

    success, status_name, objective, x, solve_time = solve_with_cplex(c, A_ub, b_ub, A_eq, b_eq)

    print()
    print("CPLEX RESULT")
    print("------------")

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
