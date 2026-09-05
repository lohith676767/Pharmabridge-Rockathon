# CPU Baseline Pipeline — File-by-File Explanation

This document walks through every file in `cpu_baseline/`, what it does,
and why the code is written the way it is.

---

## 1. `gpu_format.py`

**Purpose:** The single shared reader/writer for the sparse COO format
that both the CPU (`cpu_solve.py`) and GPU (CUDA PDHG) solvers read.
Every other file here calls into this module rather than writing/reading
the file format itself — so there's exactly one place the format lives,
and it can't drift into two different versions on the CPU and GPU sides
(which is exactly what happened once already in this project's history:
an earlier hand-rolled dense format on each side wasn't kept in sync).

### Why COO (sparse) instead of a dense flattened matrix

An earlier version of this pipeline wrote every cell of the constraint
matrix, including zeros, into a flat text file. That's fine for a small
problem, but for a real large Netlib benchmark (tens of thousands of
rows/columns, a tiny fraction nonzero) it means writing out hundreds of
billions of numbers — physically impossible to generate or read on a
laptop (a genuine attempt at converting `pds-100` needed **588 GiB**
just for the matrix). COO (Coordinate format) stores only the nonzero
entries as `(row, col, value)` triples, so the file size tracks the
*actual* amount of information in the problem, not its nominal
dimensions.

### File structure

```
Line 1: M N nnz
Line 2: c            (N floats)
Line 3: b            (M floats — b_ub rows, then b_eq rows)
Line 4: row_type     (M ints — 0 = "<=", 1 = "=")
Remaining nnz lines: row col value   (0-indexed)
```

### `export_to_gpu_format(c, A_ub, b_ub, A_eq, b_eq, filename)`

- `_as_sparse_or_empty` coerces `A_ub`/`A_eq` into a sparse matrix even
  if the caller passed `None` (meaning "no rows of this type") or a
  dense array — callers shouldn't need to special-case an empty
  constraint block themselves.
- `sparse.vstack([A_ub, A_eq])` — stacks inequality rows on top of
  equality rows into one matrix, matching the GPU team's own reference
  file-writing code exactly (verified byte-for-byte structurally
  identical against their snippet on a toy example).
- `row_type` is built as a block of `0`s (length = number of `A_ub`
  rows) followed by a block of `1`s (length = number of `A_eq` rows) —
  same order as the stacked matrix, so row `i`'s type always matches
  row `i`'s data.
- The final loop writes one `row col value` line per nonzero directly
  from the COO matrix's own `.row`/`.col`/`.data` arrays — no Python-side
  matrix math needed to extract them, this is what COO format is
  designed for.

### `read_gpu_format(filename)`

- Reads the first 4 lines normally (they're only `O(N)`/`O(M)` in size,
  not `O(nnz)`), but reads the (potentially huge) triple list with
  `np.loadtxt(f, max_rows=nnz)` — a vectorized, fast bulk-read — rather
  than looping line-by-line in Python, which would be far too slow for
  a large sparse problem's nonzero list.
- Rebuilds a `scipy.sparse.coo_matrix` from the triples, converts to CSR
  (fast row-slicing), then splits it back into `A_ub`/`A_eq` using
  `row_type == 0` / `row_type == 1` boolean masks.
- Returns `None` for `A_ub`/`b_ub` (or `A_eq`/`b_eq`) if there are no
  rows of that type — matching exactly what `scipy.optimize.linprog`
  expects when a constraint type is absent (verified: a pure-equality
  problem and a pure-inequality problem both solve correctly).

---

## 2. `generate_synthetic_lp.py`

**Purpose:** Creates a random **block-diagonal** LP problem from scratch
(no external data needed), built and exported entirely as sparse
matrices — no dense array ever exists, even temporarily, for the full
problem (only small individual blocks are briefly dense, which is safe
regardless of overall problem size).

### Why block-diagonal instead of purely random

Earlier versions filled the entire constraint matrix with random values
and then randomly dropped some to control sparsity — meaning any
variable could touch any constraint, which doesn't reflect a real
refinery: a given catalyst bed only interacts with a specific subset of
feedstocks, not all of them. Real refinery constraint matrices have
**structural** sparsity from this unit/catalyst separation, not just a
uniform random scattering.

### `generate_block_diagonal_blending_lp(n_vars, n_constraints, seed, density, n_blocks, coupling_fraction)`

- **Block layout**: variables and constraint rows are each split into
  `n_blocks` groups (`np.array_split`) — each group represents one
  process unit's chemicals and the constraints that apply only to it.
  `n_blocks` defaults to roughly one block per 25 variables if not
  specified, clamped so no block ends up empty.
- **Coupling rows**: `coupling_fraction` (default 5%) of the constraint
  rows are set aside as "coupling" rows that reference every block's
  variables — representing a shared resource like total crude intake or
  a shared utility. Without these, the blocks would be `n_blocks`
  completely independent LPs, which isn't realistic and would be a less
  interesting benchmark. Built directly with `scipy.sparse.random`
  (never a dense `(n_coupling, n_vars)` array), so this stays
  memory-safe even for very large `n_vars`.
- **Per-block generation loop**: each block is built as a small dense
  array first (safe regardless of overall scale, since an individual
  block is only ~25 variables wide by construction), then wrapped as
  `sparse.csr_matrix(block)`.
- `sparse.block_diag(blocks, format="csr")` assembles all the blocks
  into one sparse matrix in their correct diagonal positions — this is
  what replaced the old `A_ub[np.ix_(rows_in_block, vars_in_block)] = block`
  dense-array placement; `block_diag` never materializes the full dense
  matrix at any point.
- `profit = rng.uniform(10, 100, size=n_vars)` then `c = -profit` —
  unchanged: `scipy.optimize.linprog` only *minimizes*, so to
  **maximize** profit the objective is negated.
- `per_var_cap` + `b_block = (block @ per_var_cap) * rng.uniform(0.3, 0.6, ...)`
  — unchanged non-trivial-optimum trick, computed per block using only
  that block's own variables.
- Returns a dict bundling `c`, `A_ub` (sparse), `b_ub`, `n_blocks`,
  `n_coupling`, plus a descriptive `name`. No `A_eq`/bounds fields —
  this generator only ever produces `<=` constraints with `x >= 0`.
- `--blocks 1 --coupling 0` collapses this back to the old fully-dense
  single-block behavior, useful for direct before/after comparison.

### `main()`

CLI wrapper: parses `--vars`, `--constraints`, `--seed`, `--density`,
`--blocks`, `--coupling`, `--out`; validates arguments; calls the
generator; writes the result via `gpu_format.export_to_gpu_format`
(passing `None` for `A_eq`/`b_eq`, since there are none); prints a
summary including the actual sparsity achieved.

---

## 3. `mps_to_txt.py`

**Purpose:** Converts a real Netlib `.mps` benchmark file into the
shared sparse COO format, staying sparse throughout — this is the part
that makes converting genuinely large Netlib benchmarks possible at all
(a dense intermediate anywhere in this pipeline reintroduces the exact
memory wall the COO format exists to avoid).

### `_sparse_from_highspy(a_matrix, n_row, n_col)`

HiGHS stores its constraint matrix in column-compressed sparse form
(`start`/`index`/`value` arrays — standard CSC format). This function
wraps those arrays directly as a `scipy.sparse.csc_matrix` — `O(nnz)`,
no dense copy, unlike the earlier version of this file which expanded
them into a full dense array (`np.zeros((n_row, n_col))`) via a
Python loop — that approach silently failed for anything beyond a
modest problem size (crashed with a 588 GiB allocation attempt on a
real ~156k x ~505k Netlib benchmark).

### `canonicalize_mps(path)` — the core logic, in two phases, entirely sparse

**Phase 1 — split rows into `<=` (`A_ub`) and `=` (`A_eq`) groups.**
HiGHS represents every constraint as `row_lower ≤ A·row ≤ row_upper`.
This loop walks each row and classifies it:
- `lo == hi` (equality) → kept as **one** `A_eq` row (the COO format
  supports equality natively via `row_type`, so — unlike the old
  dense-only format — this no longer needs to be artificially split
  into two `<=` rows; real MPS files produce noticeably fewer output
  rows as a result, e.g. AFIRO: 35→27, BLEND: 117→74)
- only `hi` finite (a `≤` row) → kept as-is
- only `lo` finite (a `≥` row) → negated: `-A·x ≤ -lo`
- both finite, unequal (a ranged row) → split into two `<=` rows (upper,
  and negated lower), same as before
- Row selection and sign-flipping is done via sparse row indexing +
  `.multiply()` (broadcasting a per-row sign vector) — never densifying.

**Phase 2 — eliminate variable bounds by substitution**, so every
remaining variable is a plain `y ≥ 0` (unchanged logic from before, just
applied to sparse matrices now):
- If a variable has a finite lower bound `lo` (the common case,
  including the default `lo=0`): substitute `x = lo + y`. If it also
  has a finite upper bound, add an extra `A_ub` row `y ≤ hi - lo`.
- If a variable is unbounded below but bounded above (`MI`/`UP`
  combo): substitute `x = hi - y` (sign flip).
- If a variable is **fully free** (`FR`, no bounds at all): split it
  into two nonnegative variables, `x = y_pos - y_neg`.
- Column selection/sign-flip is applied via direct sparse column
  indexing (`A_can[:, orig_cols].multiply(signs)`) rather than building
  a selection matrix and multiplying it out — an earlier version of
  this built a **dense** `n_col x new_n` selection matrix for this step,
  which needed **17.7 GiB** for a ~49k-variable problem despite the
  operation being mathematically sparse (one nonzero per column);
  direct indexing needs no such intermediate at all.
- Extra bound rows (one `y ≤ hi - lo` row per finitely-upper-bounded
  variable) are built as a single sparse block via `sparse.coo_matrix`
  and appended with one `vstack` call, rather than one `vstack` per row
  (which would be slow for many bounded variables).

Because shifting variables changes the objective by a constant
(`c^T x = c^T shift + c^T(T y)`), that constant (`objective_constant`)
is tracked separately — the COO format has no field for it, so it can't
just be silently dropped without corrupting the true optimal value.

### `write_meta` / `convert_file` / `main()`

- `write_meta` writes the sidecar `.meta.json`: `objective_constant`,
  the new variable name mapping, and dimensions.
- `convert_file` calls `canonicalize_mps`, exports via
  `gpu_format.export_to_gpu_format`, writes the meta file, and prints
  dimensions/nonzero count/density computed from the sparse matrices'
  own `.nnz` (never from a dense array that no longer exists).
- `main()` supports both single-file (`input --out`) and batch
  (`--input-dir --output-dir`) conversion, with per-file error handling
  in batch mode so one bad file doesn't kill the whole run.

---

## 4. `cpu_solve.py`

**Purpose:** The actual CPU baseline solver — reads any file in the
shared COO format, solves it with `scipy.optimize.linprog(method="highs")`
using sparse matrices directly, times it, and writes a verification JSON.

### Reading

Delegates entirely to `gpu_format.read_gpu_format`, which returns
`(c, A_ub, b_ub, A_eq, b_eq)` as `scipy.sparse` matrices (or `None` for
an absent constraint type) — no matrix parsing logic lives in this file
at all, by design, so there's only one place the format's structure is
understood.

### `read_objective_constant(input_file)`

Looks for a file with the same name but `.meta.json` extension sitting
next to the input (`Path.with_suffix`). If found, reads
`objective_constant` from it; otherwise defaults to `0.0`. This is what
keeps Netlib-derived results correct without requiring any extra flag
or manual step from you.

### `solve_on_cpu(c, A_ub, b_ub, A_eq, b_eq)`

- Calls `scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
  b_eq=b_eq, bounds=(0, None), method="highs")` — `linprog` accepts
  sparse matrices (or `None`) for both constraint types directly, so
  there's no dense conversion anywhere in the CPU solve path either.
- Timing uses `time.perf_counter()` (a monotonic, high-resolution
  clock — the correct choice for benchmarking, unlike `time.time()`
  which can jump if the system clock adjusts) placed tightly around
  just the solve call, not the file I/O.

### `save_result(...)`

- Builds the result dict. Critically:
  `"objective_value": float(result.fun) + objective_constant` — this
  is where the constant from `mps_to_txt.py` gets added back in, so
  the JSON always reports the *true* optimal value of the original
  problem, not the shifted/canonicalized one.
- On failure (`result.success == False`), writes `null`s for the
  objective/solution instead of crashing.

### `main()`

CLI wrapper: `--input`/`--out`; reads the matrices, reads the constant,
solves, prints a human-readable summary to the console, writes the JSON.

---

## 5. `requirements.txt`

```
numpy>=1.24
scipy>=1.11
highspy>=1.15.1
```

- `numpy` — vector operations (`c`, `b`, shift/sign arrays).
- `scipy` — `scipy.sparse` (the COO/CSR/CSC matrices everything is built
  from) and `scipy.optimize.linprog` (the solver itself).
- `highspy` — official HiGHS Python bindings, used only by
  `mps_to_txt.py` to correctly parse MPS files (offloading edge cases
  like `RANGES`, `OBJSENSE`, and quirky optional fields to the library
  instead of hand-parsing).

---

## The full pipeline, end to end

```
                    ┌─ generate_synthetic_lp.py ──┐
                    │                              ├──► <name>.txt ──► cpu_solve.py ──► result.json
Netlib .mps ──► mps_to_txt.py ──► <name>.meta.json ┘         ▲
                                                        (also read by the GPU solver)
```

Verified end-to-end against known/HiGHS-confirmed results, and
byte-for-byte structural compatibility with the GPU team's own
reference file-writing code:
- **AFIRO** (Netlib) → objective **-464.75314285714285**
- **BLEND** (Netlib, exposed and fixed a real parsing bug around an
  omitted RHS vector name) → objective **-30.812149845828237**
- **SHARE1B** (Netlib, larger/sparser real benchmark) → objective
  **-76589.3185791857**
- Synthetic edge-case files covering equality/`>=`/ranged rows,
  `MI`/`UP`/`FX`/`FR` variable bounds, problems with **zero** `<=` rows,
  and problems with **zero** `=` rows — all match `highspy`'s direct
  solve, including correct `objective_constant` recovery.
- A toy problem run through both `gpu_format.py`'s writer and the GPU
  team's own reference COO-writing snippet produced byte-identical file
  structure (same dimensions, values, row order, and triple order).
