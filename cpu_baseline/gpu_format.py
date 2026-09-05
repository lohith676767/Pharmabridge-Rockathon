"""
Shared reader/writer for the GPU (CUDA PDHG) solver's sparse COO input
format -- the single source of truth for this format, used by both
mps_to_txt.py and generate_synthetic_lp.py (writers) and cpu_solve.py
(reader), so the CPU and GPU sides can never drift onto two different
file layouts.

Why COO instead of the old dense 4-line format
------------------------------------------------
The old format wrote out every cell of the M x N constraint matrix,
including zeros. For a small dense problem that's fine, but for a real
large sparse benchmark (e.g. a ~156k x ~505k Netlib problem) that means
writing hundreds of billions of numbers to a single-precision-per-word
text file -- physically impossible to generate, store, or read on a
laptop, regardless of how efficient the code is. COO (Coordinate
format) only stores the nonzero entries, so a problem with a handful of
thousand real values stays a handful of thousand values, no matter how
large the matrix's nominal dimensions are.

File structure
--------------
    Line 1: M N nnz          (total rows, total columns, nonzero count)
    Line 2: c                 (N floats, space-separated)
    Line 3: b                 (M floats -- b_ub rows, then b_eq rows)
    Line 4: row_type          (M ints -- 0 = "<=" row, 1 = "=" row)
    Remaining nnz lines: "row col value", one nonzero per line, 0-indexed

Row order in lines 3/4 and in the "row" column of the COO triples all
match: every A_ub row comes before every A_eq row (scipy.sparse.vstack
order), and row_type marks which is which.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _as_sparse_or_empty(A, n_col):
    """Coerce A (dense array, sparse matrix, or None) into a COO matrix with n_col columns."""
    if A is None:
        return sparse.coo_matrix((0, n_col))
    A = sparse.coo_matrix(A)
    if A.shape[0] == 0:
        return sparse.coo_matrix((0, n_col))
    return A


def export_to_gpu_format(c, A_ub, b_ub, A_eq, b_eq, filename):
    """Write c, (A_ub x <= b_ub), (A_eq x = b_eq), x >= 0 to the GPU solver's COO format.

    A_ub / A_eq may be dense arrays, any scipy.sparse matrix type, or
    None (meaning "no rows of this type"). b_ub / b_eq may be None to
    match.
    """
    n_col = len(c)

    A_ub = _as_sparse_or_empty(A_ub, n_col)
    A_eq = _as_sparse_or_empty(A_eq, n_col)

    A = sparse.vstack([A_ub, A_eq], format="coo")
    b = np.concatenate([
        np.asarray(b_ub, dtype=float).reshape(-1) if b_ub is not None else np.zeros(0),
        np.asarray(b_eq, dtype=float).reshape(-1) if b_eq is not None else np.zeros(0),
    ])
    row_type = np.concatenate([
        np.zeros(A_ub.shape[0], dtype=int),
        np.ones(A_eq.shape[0], dtype=int),
    ])

    M, N = A.shape
    nnz = A.nnz

    if len(b) != M:
        raise ValueError(f"b has {len(b)} values but A has {M} rows")

    with open(filename, "w") as f:
        f.write(f"{M} {N} {nnz}\n")
        f.write(" ".join(f"{v:.17g}" for v in c) + "\n")
        f.write(" ".join(f"{v:.17g}" for v in b) + "\n")
        f.write(" ".join(str(int(t)) for t in row_type) + "\n")
        for r, cc, v in zip(A.row, A.col, A.data):
            f.write(f"{r} {cc} {v:.17g}\n")


def read_gpu_format(filename):
    """Read the COO format back into (c, A_ub, b_ub, A_eq, b_eq).

    A_ub / A_eq are returned as scipy.sparse CSR matrices, or None if
    there are no rows of that type -- matching what
    scipy.optimize.linprog expects directly.
    """
    with open(filename, "r") as f:
        M, N, nnz = map(int, f.readline().split())
        c = np.array(list(map(float, f.readline().split())))
        b = np.array(list(map(float, f.readline().split())))
        row_type = np.array(list(map(int, f.readline().split())))

        # Vectorized read of the (potentially huge) triple list -- a
        # plain Python per-line loop here would be far too slow for a
        # large sparse problem.
        triples = np.loadtxt(f, max_rows=nnz, ndmin=2) if nnz > 0 else np.zeros((0, 3))

    if len(c) != N:
        raise ValueError(f"c has {len(c)} values, expected {N}")
    if len(b) != M:
        raise ValueError(f"b has {len(b)} values, expected {M}")
    if len(row_type) != M:
        raise ValueError(f"row_type has {len(row_type)} values, expected {M}")
    if triples.shape[0] != nnz:
        raise ValueError(f"found {triples.shape[0]} nonzero entries, expected {nnz}")

    rows = triples[:, 0].astype(np.int64)
    cols = triples[:, 1].astype(np.int64)
    vals = triples[:, 2]

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(M, N)).tocsr()

    ub_mask = row_type == 0
    eq_mask = row_type == 1

    A_ub = A[ub_mask].tocsr() if ub_mask.any() else None
    b_ub = b[ub_mask] if ub_mask.any() else None
    A_eq = A[eq_mask].tocsr() if eq_mask.any() else None
    b_eq = b[eq_mask] if eq_mask.any() else None

    return c, A_ub, b_ub, A_eq, b_eq
