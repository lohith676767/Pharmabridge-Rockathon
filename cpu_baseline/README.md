# CPU Baseline LP Solver (SIH26119 — Data & Problem Formulation)

Solves the same LP matrices the GPU (cuSOLVER) side times, using
`scipy.optimize.linprog(method="highs")` on CPU only. Produces a
verification package (objective value + solution vector + timing) that
the dashboard uses to check GPU numerical correctness against this
baseline.

## Setup

```bash
cd cpu_baseline
pip install -r requirements.txt
```

## 1. Get problem data

**Option A — Netlib LP benchmark (.mps)**
Download a small/dense Netlib set (e.g. `afiro`, `share1b`) from
https://www.netlib.org/lp/data/ and pass the `.mps` file straight to
`cpu_solve.py`. Netlib problems are mostly *sparse* — pick smaller, denser
ones if you're also benchmarking cuSOLVER's dense routines, and say so
explicitly in the pitch.

**Option B — synthetic dense "what-if" matrix**

```bash
python generate_synthetic_lp.py --vars 500 --constraints 300 --seed 42 \
    --out synthetic_500x300.npz
```

Generates a random dense maximize-profit LP (petroleum-blending style:
per-unit profits, capacity constraints) with a nontrivial optimum, so it
actually exercises the solver instead of trivially returning zero.

## 2. Solve on CPU and save the verification package

```bash
# Netlib
python cpu_solve.py --mps afiro.mps --out results/afiro_cpu.json

# Synthetic
python cpu_solve.py --npz synthetic_500x300.npz --out results/synth_cpu.json
```

Each run writes a JSON file with:

- `objective_value` — final `c^T x`
- `solution_vector` — the optimal `x`
- `solve_time_seconds` — wall-clock CPU solve time
- `status` / `success` / `message` — HiGHS termination info

Hand this JSON to the GPU side (or the dashboard) to diff against the
cuSOLVER result — same objective value and solution vector (within
numerical tolerance) is the correctness check; `solve_time_seconds` is
the CPU number to beat.

## Files

- `mps_reader.py` — minimal free-format Netlib `.mps` parser (ROWS,
  COLUMNS, RHS, RANGES, BOUNDS) producing `linprog`-ready arrays.
- `generate_synthetic_lp.py` — dense synthetic LP generator.
- `cpu_solve.py` — loads a problem (`--mps` or `--npz`), solves with
  `linprog(method="highs")`, times it, writes the verification JSON.
