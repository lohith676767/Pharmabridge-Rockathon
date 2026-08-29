# CPU Baseline Pipeline — File-by-File Explanation

This document walks through every file in `cpu_baseline/`, what it does,
and why the code is written the way it is.

---

## 1. `generate_synthetic_lp.py`

**Purpose:** Creates a random dense LP problem from scratch (no external
data needed) and writes it directly in the shared 4-line format both the
CPU and GPU solvers read.

### `generate_dense_blending_lp(n_vars, n_constraints, seed, density)`

- `profit = rng.uniform(10, 100, size=n_vars)` — assigns each variable
  (think: a blend-stock) a random per-unit profit between 10 and 100.
- `c = -profit` — `scipy.optimize.linprog` only *minimizes*, so to
  **maximize** profit, the objective is negated
  (`min(-profit) ≡ max(profit)`).
- `A_ub = rng.uniform(0.1, 5.0, size=(n_constraints, n_vars))` — builds
  the constraint matrix with random positive coefficients (every
  variable consumes some amount of every resource).
- The `density < 1.0` branch randomly zeroes out entries to make `A`
  sparse instead of fully dense, but guarantees every row keeps at
  least one nonzero (an all-zero row would be a meaningless/degenerate
  constraint).
- `per_var_cap` + `b_ub = (A_ub @ per_var_cap) * rng.uniform(0.3, 0.6, ...)`
  — this is the key trick that makes the problem non-trivial. It
  computes what each constraint's right-hand side *would* be if every
  variable ran at a generous cap, then shrinks it to 30-60% of that.
  This makes capacities tight enough that the optimal solution isn't
  just "produce nothing" (`x=0`), which would be a useless benchmark.
- `bounds = [(0, None)] * n_vars` — standard non-negativity (`x ≥ 0`),
  no upper bound on any single variable — the tightness only comes
  from the shared constraint rows.
- Returns a dict bundling `c`, `A_ub`, `b_ub`, `bounds` plus a
  descriptive `name`.

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
