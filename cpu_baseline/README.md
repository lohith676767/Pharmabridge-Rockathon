# CPU Baseline LP Solver (SIH26119 — Data & Problem Formulation)

One shared sparse input format, two producers, one solver — this is the
CPU-side counterpart to the GPU (CUDA PDHG) solver, built so both sides
load the **exact same input file**.

```
                    ┌─ generate_synthetic_lp.py ──┐
                    │                              ├──► <name>.txt ──► cpu_solve.py ──► result.json
Netlib .mps ──► mps_to_txt.py ──► <name>.meta.json ┘         ▲
                                                        (also read by the GPU solver)
```

## Setup

```bash
cd cpu_baseline
pip install -r requirements.txt
```

## The shared format (sparse COO)

Every problem — synthetic or Netlib-derived — ends up as a sparse
Coordinate-format (COO) text file, defined and shared by
`gpu_format.py`:

```
Line 1: M N nnz          (total rows, total columns, nonzero count)
Line 2: c                 (N floats — objective coefficients)
Line 3: b                 (M floats — b_ub rows, then b_eq rows)
Line 4: row_type          (M ints — 0 = "<=" row, 1 = "=" row)
Remaining nnz lines: row col value   (one nonzero per line, 0-indexed)
```

representing `minimize c^T x` subject to `A_ub x <= b_ub`, `A_eq x = b_eq`,
`x >= 0`. Only the nonzero entries are ever written or held in memory —
this is what makes a real large, sparse Netlib benchmark (tens of
thousands of rows/columns, but a tiny fraction nonzero) tractable at
all: as a dense matrix it would need hundreds of GB; as COO it's a few
hundred KB. Both the CPU (`cpu_solve.py`) and GPU solvers read this same
file, so there's no risk of the two sides silently solving different
problems.

## 1. Get problem data

**Option A — synthetic block-diagonal "what-if" matrix**

```bash
python generate_synthetic_lp.py --vars 500 --constraints 300 --seed 42 \
    --out matrix_input.txt
```

Generates a random **block-diagonal** maximize-profit LP
(petroleum-blending style: per-unit profits, capacity constraints) with
a nontrivial optimum, assembled directly as a sparse matrix
(`scipy.sparse.block_diag`, never a dense array — scales to very large
variable counts) and written straight to the shared COO format.
Variables and constraints are partitioned into blocks (each
representing a process unit — e.g. a catalyst bed processing a specific
set of feedstocks); coefficients are only generated *within* a block,
everywhere else is structurally zero. A small number of coupling rows
(`--coupling`, default 5%) span every block to represent shared
resources (total crude intake, shared utilities), keeping the problem
one connected LP instead of independent sub-problems. This models real
refinery sparsity (specific chemicals only interact with specific
catalysts) instead of academic random scattering.

- `--blocks N` — number of process-unit blocks (default: auto, ~1 per 25 variables)
- `--coupling F` — fraction of constraints reserved as cross-block coupling rows (default 0.05)
- `--density F` — fraction of nonzero entries *within* each block (default 1.0 = fully dense inside a block)
- `--blocks 1 --coupling 0` recovers the old fully-dense-single-block behavior, for comparison.

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
parse the MPS file correctly, then canonicalizes variable bounds and
free variables into the `x >= 0`-only form the COO format requires.
Equality rows, `>=` rows, and ranged rows are kept/converted directly
into the format's native `<=`/`=` row types — equality rows are **not**
artificially split into two `<=` rows the way a purely-`<=`-only format
would require, so real MPS problems produce fewer output rows than a
naive conversion would. Everything stays sparse throughout (never a
dense `n_row x n_col` array), which is what makes converting genuinely
large Netlib benchmarks possible in the first place.

Bound elimination introduces a constant term the COO format has no
field for, so it's written to a companion `<name>.meta.json` file —
`cpu_solve.py` picks this up automatically (see below).

## 2. Solve on CPU

```bash
python cpu_solve.py --input matrix_input.txt --out cpu_result.json
```

Reads the COO file into `scipy.sparse` matrices and solves with
`scipy.optimize.linprog(method="highs")` — `linprog` accepts sparse
`A_ub`/`A_eq` directly, so no dense conversion happens anywhere in this
pipeline.

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
GPU result — same objective value and solution vector (within numerical
tolerance) is the correctness check; `solve_time_seconds` is the CPU
number to beat.

## Verified correctness

Tested end-to-end against known results, and cross-checked structurally
byte-for-byte against the GPU team's own reference file-writing code:
- **AFIRO** (Netlib) → -464.75314285714285 ✓
- **BLEND** (Netlib, exercises a non-standard MPS quirk — an omitted RHS
  vector name) → -30.812149845828237 ✓ (matches `highspy`'s own direct solve)
- **SHARE1B** (Netlib, larger/sparser real benchmark) → -76589.3185791857 ✓
- Synthetic edge-case files covering equality/`>=`/ranged rows and
  `MI`/`UP`/`FX`/`FR` variable bounds, including problems with **zero**
  `<=` rows and problems with **zero** `=` rows → all match `highspy`'s
  direct solve, including correct `objective_constant` recovery.
- A hand-built toy problem run through both `gpu_format.py`'s writer and
  the GPU team's own reference COO-writing code produced byte-identical
  file structure (same dimensions, values, row order, and triple order).

## Files

- `gpu_format.py` — the single shared reader/writer for the sparse COO
  format, used by every other file here (and matching the GPU team's
  own file-format contract) so the format can't drift into two
  different versions across the CPU and GPU sides.
- `generate_synthetic_lp.py` — block-diagonal synthetic LP generator
  (realistic sparsity: chemicals only interact with their own process
  unit's constraints, plus a few cross-block coupling rows), built and
  exported entirely as sparse matrices.
- `mps_to_txt.py` — `highspy`-based MPS reader + canonicalizer, converts
  real Netlib `.mps` files to the shared COO format (+ `.meta.json`),
  staying sparse throughout.
- `cpu_solve.py` — solves any COO-format file with
  `scipy.optimize.linprog(method="highs")`, applies the `.meta.json`
  constant if present, times the solve, writes the verification JSON.
