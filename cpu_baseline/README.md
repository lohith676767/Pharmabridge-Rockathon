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

## 3. Convert a problem to the GPU solver's flat TXT format

The GPU (cuSOLVER) side needs a plain 4-line dense format it can load
directly, since it can't parse `.mps`. `mps_to_txt.py` uses `highspy`
(the official HiGHS Python bindings) to read the MPS file correctly and
canonicalizes it — equality rows, `>=` rows, ranged rows, variable
bounds, and free variables — into the restricted form the GPU solver
understands:

```
minimize c^T x
subject to:  A x <= b
             x >= 0
```

```bash
python mps_to_txt.py afiro.mps --out afiro.txt
# or convert a whole directory of .mps files at once:
python mps_to_txt.py --input-dir netlib --output-dir txt
```

Each conversion writes two files:
- `<name>.txt` — the 4-line format (`M N` / `c` / flattened `A` / `b`)
- `<name>.meta.json` — the objective constant introduced by variable
  substitution (bounds shifting adds a constant term the TXT format has
  no room for) plus the new variable name mapping. **The true optimal
  objective of the original problem = the objective the GPU solver
  reports from the TXT file + `objective_constant` from the meta file.**

This was validated against `highspy`'s own direct solve on AFIRO and on
synthetic MPS files exercising every bound/row type (equality, `>=`,
ranged rows, `MI`/`UP`/`FR` bounds) — canonicalized and original problems
produce identical optimal objective values.

## Files

- `mps_reader.py` — `highspy`-based `.mps` reader producing `linprog`-ready
  arrays in the problem's *original* variable space (bounds and equality
  passed through natively, not canonicalized), used by `cpu_solve.py`.
  Originally a hand-rolled parser; switched to `highspy` after it silently
  mis-parsed a real Netlib file (BLEND) that omits the optional RHS vector
  name — same class of edge case `mps_to_txt.py` was built to avoid.
- `mps_to_txt.py` — `highspy`-based MPS reader + canonicalizer producing
  the GPU solver's flat `A x <= b, x >= 0` TXT format (see above).
- `generate_synthetic_lp.py` — dense synthetic LP generator.
- `cpu_solve.py` — loads a problem (`--mps` or `--npz`), solves with
  `linprog(method="highs")`, times it, writes the verification JSON.
