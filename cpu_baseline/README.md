# CPU Baseline LP Solver (SIH26119 — Data & Problem Formulation)

One shared flat text format, two producers, one solver — this is the
CPU-side counterpart to the GPU (cuSOLVER) solver, built so both sides
load the **exact same input file**.

```
                    ┌─ generate_synthetic_lp.py ──┐
                    │                              ├──► <name>.txt ──► cpu_solve.py ──► result.json
Netlib .mps ──► mps_to_txt.py ──► <name>.meta.json ┘         ▲
                                                        (also read by GPU solver)
```

## Setup

```bash
cd cpu_baseline
pip install -r requirements.txt
```

## The shared format

Every problem — synthetic or Netlib-derived — ends up as a plain 4-line
text file:

```
Line 1: M N
Line 2: C            (N space-separated floats — objective coefficients)
Line 3: A             (M*N floats, flattened row-major)
Line 4: B             (M space-separated floats — RHS)
```

representing `minimize c^T x  subject to  A x <= b, x >= 0`. Both the CPU
(`cpu_solve.py`) and GPU (cuSOLVER) solvers read this same file, so there's
no risk of the two sides silently solving different problems.

## 1. Get problem data

**Option A — synthetic dense "what-if" matrix**

```bash
python generate_synthetic_lp.py --vars 500 --constraints 300 --seed 42 \
    --out matrix_input.txt
```

Generates a random dense maximize-profit LP (petroleum-blending style:
per-unit profits, capacity constraints) with a nontrivial optimum, and
writes it straight to the shared `.txt` format.

**Option B — real Netlib LP benchmark (.mps)**

Netlib's own files (https://www.netlib.org/lp/data/) are compressed —
decompress with Netlib's `emps` tool first:
```bash
gcc -O2 -o emps emps.c
./emps afiro > afiro.mps
```
(A plain-text copy can also be pulled from HiGHS's own test suite,
`ERGO-Code/HiGHS/check/instances/`, which needs no decompression.)

Then convert to the shared format:
```bash
python mps_to_txt.py afiro.mps --out afiro.txt
```

`mps_to_txt.py` uses `highspy` (the official HiGHS Python bindings) to
parse the MPS file correctly, then canonicalizes equality rows, `>=`
rows, ranged rows, variable bounds, and free variables into the
restricted `A x <= b, x >= 0` form. This substitution introduces a
constant term the flat format has no room for, so it's written to a
companion `<name>.meta.json` file — `cpu_solve.py` picks this up
automatically (see below).

## 2. Solve on CPU

```bash
python cpu_solve.py --input matrix_input.txt --out cpu_result.json
```

If a companion `<input-basename>.meta.json` exists next to the input file
(i.e. the problem came from `mps_to_txt.py`), its `objective_constant` is
automatically added to the solved objective, so the reported value is
always the true optimum of the original problem — synthetic problems
(no meta file) are unaffected, the constant defaults to 0.

Each run writes a JSON file with:

- `objective_value` — the true optimal `c^T x` (constant already applied)
- `objective_constant` — the constant that was added (0 for synthetic problems)
- `solution_vector` — the optimal `x`
- `solve_time_seconds` — wall-clock CPU solve time
- `status` / `success` / `message` — HiGHS termination info

Hand this JSON to the GPU side (or the dashboard) to diff against the
cuSOLVER result — same objective value and solution vector (within
numerical tolerance) is the correctness check; `solve_time_seconds` is
the CPU number to beat.

## Verified correctness

Tested end-to-end against known results:
- **AFIRO** (Netlib) → -464.75314285714285 ✓
- **BLEND** (Netlib, exercises a non-standard MPS quirk — an omitted RHS
  vector name) → -30.812149845828237 ✓ (matches `highspy`'s own direct solve)
- Synthetic edge-case files covering equality/`>=`/ranged rows and
  `MI`/`UP`/`FX`/`FR` variable bounds → all match `highspy`'s direct solve,
  including correct `objective_constant` recovery.

## Files

- `generate_synthetic_lp.py` — dense synthetic LP generator, writes
  directly to the shared `.txt` format.
- `mps_to_txt.py` — `highspy`-based MPS reader + canonicalizer, converts
  real Netlib `.mps` files to the shared `.txt` format (+ `.meta.json`).
- `cpu_solve.py` — solves any `.txt` file in the shared format with
  `scipy.optimize.linprog(method="highs")`, applies the `.meta.json`
  constant if present, times the solve, writes the verification JSON.
