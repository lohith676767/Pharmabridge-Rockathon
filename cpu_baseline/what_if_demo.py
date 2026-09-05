"""
Live "what-if" re-solve demo for the petroleum-blending CPU baseline.

Loads (or generates) a blending LP, solves it once as the baseline, then
lets you change one coefficient -- a feedstock's profit/cost, or a
capacity/resource limit -- and re-solves instantly, showing exactly how
the optimal blend plan shifts in response. This is the CPU half of the
"change a business assumption, watch the plan update live" demo; a GPU
call can be added alongside solve_on_cpu() later without changing this
script's structure (see the RUN_GPU placeholder below).

Usage
-----
Interactive mode (recommended for a live demo):
    python what_if_demo.py --vars 40 --constraints 25 --blocks 4 --seed 7

    Then at each prompt, either:
      cost <var_index> <new_value>      -- change a feedstock's profit
      cap <row_index> <new_value>       -- change a capacity/resource limit
      show                              -- reprint the current top variables
      quit

One-shot scenario mode (for scripted/repeatable demos):
    python what_if_demo.py --vars 40 --constraints 25 --blocks 4 --seed 7 \
        --scenario cost:3:150 --scenario cap:0:50
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy import sparse

from generate_synthetic_lp import generate_block_diagonal_blending_lp
from cpu_solve import solve_on_cpu
from gpu_format import read_gpu_format


def load_or_generate(args) -> dict:
    """Build the base problem, either from an existing GPU-format file or freshly generated.

    objective_mode is "profit" only for the synthetic generator's own
    convention (c = -profit, so linprog's minimization maximizes profit).
    An arbitrary --input file (e.g. from mps_to_txt.py) minimizes its own
    cost directly and should NOT be negated/labeled as profit.
    """
    if args.input:
        c, A_ub, b_ub, A_eq, b_eq = read_gpu_format(args.input)
        if A_eq is not None:
            raise ValueError(
                "what_if_demo.py only supports pure-inequality (A_ub only) problems for now "
                "-- generate_synthetic_lp.py output works directly."
            )
        var_names = [f"Feedstock_{j}" for j in range(len(c))]
        return {
            "name": args.input, "c": c, "A_ub": A_ub, "b_ub": b_ub,
            "var_names": var_names, "objective_mode": "cost",
        }

    problem = generate_block_diagonal_blending_lp(
        n_vars=args.vars,
        n_constraints=args.constraints,
        seed=args.seed,
        n_blocks=(args.blocks or None),
        coupling_fraction=args.coupling,
    )
    var_names = [f"Feedstock_{j}" for j in range(args.vars)]
    return {
        "name": problem["name"],
        "c": problem["c"],
        "A_ub": problem["A_ub"],
        "b_ub": problem["b_ub"],
        "var_names": var_names,
        "objective_mode": "profit",
    }


def solve_and_report(label: str, c, A_ub, b_ub, var_names, objective_mode: str = "profit", top_n: int = 8):
    """Solve on CPU and print a readable summary of the blend plan.

    objective_mode: "profit" negates linprog's minimized value for display
    (matching generate_synthetic_lp.py's c = -profit convention); "cost"
    reports linprog's value as-is (for an arbitrary --input problem that
    minimizes its own cost directly, not necessarily "profit").
    """
    result, solve_time = solve_on_cpu(c, A_ub, b_ub, None, None)

    print(f"\n=== {label} ===")
    if not result.success:
        print(f"FAILED: {result.message}")
        return None

    displayed = -result.fun if objective_mode == "profit" else result.fun
    metric_name = "Optimal profit" if objective_mode == "profit" else "Optimal cost  "
    print(f"{metric_name} : {displayed:.2f}")
    print(f"Solve time     : {solve_time * 1000:.2f} ms")

    order = np.argsort(-result.x)[:top_n]
    print(f"Top {top_n} feedstocks in the blend:")
    for idx in order:
        if result.x[idx] > 1e-6:
            print(f"  {var_names[idx]:15s} = {result.x[idx]:10.3f}")

    return {"objective": displayed, "x": result.x, "solve_time": solve_time}


def apply_cost_change(c, var_idx: int, new_value: float, objective_mode: str = "profit"):
    """Change one feedstock's objective coefficient.

    In "profit" mode c stores -profit (linprog minimizes), so the stored
    coefficient is negated relative to the displayed value. In "cost" mode
    the stored coefficient IS the displayed value, no negation.
    """
    c = c.copy()
    c[var_idx] = -new_value if objective_mode == "profit" else new_value
    return c


def apply_capacity_change(b_ub, row_idx: int, new_limit: float):
    """Change one row's capacity/resource limit."""
    b_ub = b_ub.copy()
    b_ub[row_idx] = new_limit
    return b_ub


def diff_report(before, after, var_names, top_n: int = 5):
    """Print how much the plan and profit changed between two solves."""
    if before is None or after is None:
        print("\n(Cannot compare -- one of the solves failed.)")
        return

    delta_profit = after["objective"] - before["objective"]
    pct = (delta_profit / before["objective"] * 100) if before["objective"] else float("nan")
    print(f"\n--- Impact ---")
    print(f"Profit change: {delta_profit:+.2f} ({pct:+.2f}%)")

    delta_x = after["x"] - before["x"]
    order = np.argsort(-np.abs(delta_x))[:top_n]
    print(f"Biggest plan changes:")
    for idx in order:
        if abs(delta_x[idx]) > 1e-6:
            print(f"  {var_names[idx]:15s}: {before['x'][idx]:8.3f} -> {after['x'][idx]:8.3f}  ({delta_x[idx]:+.3f})")


def run_scenario(spec: str, c, A_ub, b_ub, var_names, objective_mode: str = "profit"):
    """Apply one 'cost:idx:value' or 'cap:idx:value' scenario and return the modified (c, A_ub, b_ub)."""
    kind, idx_str, value_str = spec.split(":")
    idx, value = int(idx_str), float(value_str)
    if kind == "cost":
        label = "profit" if objective_mode == "profit" else "cost"
        print(f"\n>>> Scenario: change {var_names[idx]}'s {label} to {value}")
        return apply_cost_change(c, idx, value, objective_mode), A_ub, b_ub
    elif kind == "cap":
        print(f"\n>>> Scenario: change capacity row {idx}'s limit to {value}")
        return c, A_ub, apply_capacity_change(b_ub, idx, value)
    else:
        raise ValueError(f"Unknown scenario kind '{kind}' (use 'cost' or 'cap')")


def interactive_loop(c, A_ub, b_ub, var_names, baseline, objective_mode: str = "profit"):
    print("\nInteractive what-if mode. Commands:")
    print("  cost <var_index> <new_value>   -- change a feedstock's profit")
    print("  cap <row_index> <new_value>    -- change a capacity/resource limit")
    print("  show                            -- reprint current top variables")
    print("  quit")

    current = baseline
    cur_c, cur_A_ub, cur_b_ub = c, A_ub, b_ub

    while True:
        try:
            line = input("\nwhat-if> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == "quit":
            break
        if line == "show":
            solve_and_report("Current plan", cur_c, cur_A_ub, cur_b_ub, var_names, objective_mode)
            continue

        parts = line.split()
        if len(parts) != 3 or parts[0] not in ("cost", "cap"):
            print("Unrecognized command. Use: cost <idx> <value>  |  cap <idx> <value>  |  show  |  quit")
            continue

        kind, idx_str, value_str = parts
        try:
            idx, value = int(idx_str), float(value_str)
        except ValueError:
            print("idx must be an integer, value must be a number.")
            continue

        if kind == "cost":
            if not (0 <= idx < len(cur_c)):
                print(f"var index out of range [0, {len(cur_c) - 1}]")
                continue
            cur_c = apply_cost_change(cur_c, idx, value, objective_mode)
            metric = "profit" if objective_mode == "profit" else "cost"
            label = f"After: {var_names[idx]} {metric} -> {value}"
        else:
            if not (0 <= idx < cur_A_ub.shape[0]):
                print(f"row index out of range [0, {cur_A_ub.shape[0] - 1}]")
                continue
            cur_b_ub = apply_capacity_change(cur_b_ub, idx, value)
            label = f"After: capacity row {idx} -> {value}"

        new_result = solve_and_report(label, cur_c, cur_A_ub, cur_b_ub, var_names, objective_mode)
        diff_report(current, new_result, var_names)
        current = new_result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=str, help="load an existing GPU-format .txt problem instead of generating one")
    p.add_argument("--vars", type=int, default=40, help="number of feedstocks (if generating)")
    p.add_argument("--constraints", type=int, default=25, help="number of constraint rows (if generating)")
    p.add_argument("--blocks", type=int, default=4, help="number of process-unit blocks (if generating)")
    p.add_argument("--coupling", type=float, default=0.08, help="coupling row fraction (if generating)")
    p.add_argument("--seed", type=int, default=7, help="random seed (if generating)")
    p.add_argument(
        "--scenario", action="append", default=[],
        help="one-shot scenario as 'cost:idx:value' or 'cap:idx:value'; repeatable, applied in order"
    )
    args = p.parse_args()

    problem = load_or_generate(args)
    print(f"Problem: {problem['name']}  ({len(problem['c'])} feedstocks, {problem['A_ub'].shape[0]} constraints)")

    c, A_ub, b_ub, var_names = problem["c"], problem["A_ub"], problem["b_ub"], problem["var_names"]
    objective_mode = problem["objective_mode"]

    baseline = solve_and_report("Baseline plan", c, A_ub, b_ub, var_names, objective_mode)

    if args.scenario:
        cur_c, cur_A_ub, cur_b_ub = c, A_ub, b_ub
        for spec in args.scenario:
            cur_c, cur_A_ub, cur_b_ub = run_scenario(spec, cur_c, cur_A_ub, cur_b_ub, var_names, objective_mode)
        new_result = solve_and_report("Re-solved plan", cur_c, cur_A_ub, cur_b_ub, var_names, objective_mode)
        diff_report(baseline, new_result, var_names)
    else:
        interactive_loop(c, A_ub, b_ub, var_names, baseline, objective_mode)


if __name__ == "__main__":
    main()
