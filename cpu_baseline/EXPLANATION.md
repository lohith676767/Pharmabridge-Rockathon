# CPU Baseline Pipeline — File-by-File Explanation

This document walks through every file in `cpu_baseline/`, what it does,
and why the code is written the way it is.

---

## 1. `generate_synthetic_lp.py`

**Purpose:** Creates a random **block-diagonal** LP problem from scratch
(no external data needed) and writes it directly in the shared 4-line
format both the CPU and GPU solvers read.

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
  rows are set aside as "coupling" rows that reference *every* block's
  variables — representing a shared resource like total crude intake
  or a shared utility. Without these, the blocks would be `n_blocks`
  completely independent LPs, which isn't realistic (refineries do
  share some upstream/downstream resources across units) and would be
  a less interesting benchmark (embarrassingly parallel rather than
  needing real cross-block reasoning).
- **Per-block generation loop**: for each block, `rng.uniform(0.1, 5.0, ...)`
  fills only that block's own `(rows, vars)` submatrix — every other
  cell in the full `A_ub` stays exactly `0.0` (structural zero, not a
  randomly-dropped one). `A_ub[np.ix_(rows_in_block, vars_in_block)] = block`
  places it into the full matrix at the right block-diagonal position.
- `profit = rng.uniform(10, 100, size=n_vars)` then `c = -profit` — same
  as before: `scipy.optimize.linprog` only *minimizes*, so to
  **maximize** profit the objective is negated.
- The `density < 1.0` branch still applies *within* each block (an
  additional, optional sparsity control on top of the structural
  block-diagonal sparsity), and still guarantees no all-zero row.
- `per_var_cap` + `b_ub[rows] = (block @ per_var_cap) * rng.uniform(0.3, 0.6, ...)`
  — same non-trivial-optimum trick as before, computed per block using
  only that block's own variables.
- Coupling rows get their own (lighter — `rng.uniform(0.05, 1.0)`
  instead of `0.1, 5.0`) coefficients and the same tight-capacity trick,
  applied across *all* variables at once.
- `bounds = [(0, None)] * n_vars` — unchanged: standard non-negativity.
- Returns a dict bundling `c`, `A_ub`, `b_ub`, `bounds`, `n_blocks`,
  `n_coupling`, plus a descriptive `name`.
- `--blocks 1 --coupling 0` collapses this back to the old fully-dense
  single-block behavior, useful for direct before/after comparison.

### `export_to_solver_format(A, b, c, filename)`

- Validates that `c`'s length matches `N`, `b`'s length matches `M`,
  and `A`'s size matches `M*N` — catches dimension bugs before writing
  garbage to disk.
- Writes the 4 lines: `M N`, then `c` space-separated, then
  `A.flatten()` (row-major, i.e. row 0's values, then row 1's, etc.),
  then `b`.

### `main()`

CLI wrapper: parses `--vars`, `--constraints`, `--seed`, `--density`,
`--out`; validates arguments are sane (positive counts, density in
`(0,1]`); calls the two functions above; prints a summary.

---

## 2. `mps_to_txt.py`

**Purpose:** Converts a real Netlib `.mps` benchmark file into that same
shared 4-line format, since Netlib problems use general LP constraints
(equality, `>=`, ranged, bounded variables) that the flat format can't
natively express.

### `_dense_columns(a_matrix, n_row, n_col)`

HiGHS stores its constraint matrix in **column-compressed sparse** form
(`start`/`index`/`value` arrays — standard CSC format). This function
expands it into a plain dense NumPy array, since the flat format needs
every value written out, not just the nonzero ones.

### `canonicalize_mps(path)` — the core logic, in two phases

**Phase 1 — flatten row types into `<=` only.** HiGHS represents every
constraint as `row_lower ≤ A·row ≤ row_upper`. This loop walks each row
and rewrites it:
- `lo == hi` (equality) → two rows: `A·x ≤ hi` and `-A·x ≤ -lo`
- only `hi` finite (a `≤` row) → one row as-is
- only `lo` finite (a `≥` row) → negate: `-A·x ≤ -lo`
- both finite, unequal (a ranged row) → same as equality's two-row split

**Phase 2 — eliminate variable bounds by substitution**, so every
remaining variable is a plain `y ≥ 0`:
- If a variable has a finite lower bound `lo` (the common case,
  including the default `lo=0`): substitute `x = lo + y`. If it also
  has a finite upper bound, add an extra row `y ≤ hi - lo`.
- If a variable is unbounded below but bounded above (`MI`/`UP`
  combo): substitute `x = hi - y` (sign flip).
- If a variable is **fully free** (`FR`, no bounds at all): split it
  into two nonnegative variables, `x = y_pos - y_neg`.

This substitution is applied algebraically via a signed selection
matrix `T_signed` (`A_new = A_can @ T_signed`), and the RHS is shifted
accordingly (`b_new = b_can - A_can @ shift`) — this is standard
"eliminate bounds by substitution" LP theory, just written out
explicitly with matrices instead of per-variable loops for
clarity/vectorization.

Because shifting variables changes the objective by a constant
(`c^T x = c^T shift + c^T(T y)`), that constant (`objective_constant`)
is tracked separately — the flat text format has no field for it, so
it can't just be silently dropped without corrupting the true optimal
value.

### `write_txt` / `write_meta`

Write the 4-line file, and a sidecar `.meta.json` carrying
`objective_constant`, the new variable name mapping, and dimensions.

### `convert_file` / `main()`

CLI wrapper supporting both single-file (`input --out`) and batch
(`--input-dir --output-dir`) conversion, with per-file error handling
in batch mode so one bad file doesn't kill the whole run.

---

## 3. `cpu_solve.py`

**Purpose:** The actual CPU baseline solver — reads any file in the
shared format, solves it with `scipy.optimize.linprog(method="highs")`,
times it, and writes a verification JSON.

### `read_matrix_input(filename)`

- Reads the 4 lines one at a time, validating at each step (empty
  file, wrong field count, non-positive dimensions).
- Converts line 2/4 into NumPy float arrays (`c`, `b`) via `split()` +
  `map(float, ...)`.
- Converts line 3 into a flat array, then checks its length equals
  `M*N` exactly (catches a mismatched/corrupted file) before reshaping
  it into the `(M, N)` matrix with `order="C"` (row-major — matches
  how the generator/converter wrote it, so no transpose bugs).

### `read_objective_constant(input_file)`

Looks for a file with the same name but `.meta.json` extension sitting
next to the input (`Path.with_suffix`). If found, reads
`objective_constant` from it; otherwise defaults to `0.0`. This is
what keeps Netlib-derived results correct without requiring any extra
flag or manual step from you.

### `solve_on_cpu(A, b, c)`

- Wraps `scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=(0, None),
  method="highs")`.
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

CLI wrapper: `--input`/`--out`; reads the matrix, reads the constant,
solves, prints a human-readable summary to the console, writes the
JSON.

---

## 4. `requirements.txt`

```
numpy>=1.24
scipy>=1.11
highspy>=1.15.1
```

- `numpy` — array/matrix operations.
- `scipy` — the `linprog` solver (`cpu_solve.py`).
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
                                                        (also read by GPU solver)
```

Verified end-to-end against known/HiGHS-confirmed results on:
- **AFIRO** (Netlib) → objective **-464.75314285714285**
- **BLEND** (Netlib, exposed and fixed a real parsing bug around an
  omitted RHS vector name) → objective **-30.812149845828237**
- **SHARE1B** (Netlib) → objective **-76589.3185791857**
- Synthetic edge-case files covering equality/`>=`/ranged rows and
  `MI`/`UP`/`FX`/`FR` variable bounds, including correct
  `objective_constant` recovery for problems with non-zero bounds.
