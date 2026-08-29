"""
MPS reader for Netlib LP benchmark problems, built on highspy.

Reads an MPS file via highspy (the official HiGHS Python bindings) --
offloading every MPS parsing edge case (OBJSENSE, RANGES, optional
RHS/BOUNDS vector names, free- vs fixed-format quirks, MARKER sections)
to a battle-tested library instead of a hand-rolled parser -- and returns
plain NumPy arrays in the model's *original* variable space (bounds and
equality constraints passed through natively), ready for
scipy.optimize.linprog.

Unlike mps_to_txt.py, this does NOT canonicalize into A x <= b, x >= 0 --
scipy's linprog accepts bounds and equality constraints directly, so
keeping the original variables means the returned solution_vector maps
straight back to the problem's real decision variables.
"""

from __future__ import annotations

import highspy
import numpy as np


def _dense_columns(a_matrix, n_row: int, n_col: int) -> np.ndarray:
    """Expand HiGHS's column-wise sparse matrix into a dense n_row x n_col array."""
    A = np.zeros((n_row, n_col))
    start = a_matrix.start_
    index = a_matrix.index_
    value = a_matrix.value_
    for j in range(n_col):
        for k in range(start[j], start[j + 1]):
            A[index[k], j] = value[k]
    return A


def read_mps(path: str) -> dict:
    """Parse an MPS file and return a dict with linprog-ready arrays.

    Returns
    -------
    dict with keys:
        name        : problem name (str)
        c           : objective coefficients, shape (n,)
        A_ub, b_ub  : inequality constraints (<=), from L/G/ranged rows
        A_eq, b_eq  : equality constraints, from E rows
        bounds      : list of (lo, hi) tuples, length n
        var_names   : list of column names, length n
        row_names   : list of constraint row names (ub then eq, in that order)
    """
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(str(path))
    if status != highspy.HighsStatus.kOk:
        raise ValueError(f"highspy failed to read {path}: status={status}")

    lp = h.getLp()
    n_col = lp.num_col_
    n_row = lp.num_row_

    A = _dense_columns(lp.a_matrix_, n_row, n_col)
    c = np.array(lp.col_cost_, dtype=float)
    if lp.sense_ == highspy.ObjSense.kMaximize:
        c = -c

    row_lower = np.array(lp.row_lower_, dtype=float)
    row_upper = np.array(lp.row_upper_, dtype=float)
    col_lower = np.array(lp.col_lower_, dtype=float)
    col_upper = np.array(lp.col_upper_, dtype=float)

    var_names = list(lp.col_names_) if lp.col_names_ else [f"C{j}" for j in range(n_col)]
    orig_row_names = list(lp.row_names_) if lp.row_names_ else [f"R{i}" for i in range(n_row)]

    A_ub_rows, b_ub_rows, ub_names = [], [], []
    A_eq_rows, b_eq_rows, eq_names = [], [], []

    for i in range(n_row):
        lo, hi = row_lower[i], row_upper[i]
        row = A[i, :]
        rname = orig_row_names[i]

        if lo == hi:
            A_eq_rows.append(row); b_eq_rows.append(hi); eq_names.append(rname)
        elif np.isinf(hi):
            A_ub_rows.append(-row); b_ub_rows.append(-lo); ub_names.append(rname)
        elif np.isinf(lo):
            A_ub_rows.append(row); b_ub_rows.append(hi); ub_names.append(rname)
        else:  # ranged: both finite, not equal
            A_ub_rows.append(row); b_ub_rows.append(hi); ub_names.append(rname + "_hi")
            A_ub_rows.append(-row); b_ub_rows.append(-lo); ub_names.append(rname + "_lo")

    bounds = [
        (None if np.isinf(lo) else float(lo), None if np.isinf(hi) else float(hi))
        for lo, hi in zip(col_lower, col_upper)
    ]

    return {
        "name": lp.model_name_ or "UNKNOWN",
        "c": c,
        "A_ub": np.array(A_ub_rows) if A_ub_rows else None,
        "b_ub": np.array(b_ub_rows) if b_ub_rows else None,
        "A_eq": np.array(A_eq_rows) if A_eq_rows else None,
        "b_eq": np.array(b_eq_rows) if b_eq_rows else None,
        "bounds": bounds,
        "var_names": var_names,
        "row_names": ub_names + eq_names,
    }
