"""
Minimal free-format MPS reader for Netlib LP benchmark problems.

Netlib LP files (https://www.netlib.org/lp/data/) are distributed as
fixed/free-format .mps files. scipy has no built-in MPS parser, so this
module implements just enough of the spec to feed scipy.optimize.linprog:
    ROWS, COLUMNS, RHS, RANGES, BOUNDS, ENDATA

Only continuous LPs are supported (no MARKER INTORG/INTEND integer
sections beyond skipping them).
"""

from __future__ import annotations

import numpy as np


ROW_TYPES = {"L", "G", "E", "N"}


def read_mps(path: str) -> dict:
    """Parse an MPS file and return a dict with linprog-ready arrays.

    Returns
    -------
    dict with keys:
        name        : problem name (str)
        c           : objective coefficients, shape (n,)
        A_ub, b_ub  : inequality constraints (<=), stacked from L/G/ranged rows
        A_eq, b_eq  : equality constraints, from E rows
        bounds      : list of (lo, hi) tuples, length n
        var_names   : list of column names, length n
        row_names   : list of constraint row names (ub then eq, in that order)
    """
    section = None
    row_type = {}      # row name -> 'L'/'G'/'E'/'N'
    row_order = []      # preserve file order
    obj_row = None
    cols = {}            # col name -> {row_name: coeff}
    col_order = []
    rhs = {}              # row name -> rhs value
    ranges = {}          # row name -> range value
    bounds = {}           # col name -> [lo, hi]
    name = "UNKNOWN"

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("*"):
                continue

            if not line[0].isspace():
                token = line.split()
                keyword = token[0].upper()
                if keyword == "NAME":
                    name = token[1] if len(token) > 1 else "UNKNOWN"
                    section = None
                elif keyword in (
                    "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA",
                ):
                    section = keyword
                else:
                    section = keyword
                continue

            fields = line.split()
            if section == "ROWS":
                rtype, rname = fields[0].upper(), fields[1]
                if rtype not in ROW_TYPES:
                    continue
                row_type[rname] = rtype
                row_order.append(rname)
                if rtype == "N" and obj_row is None:
                    obj_row = rname

            elif section == "COLUMNS":
                if len(fields) >= 3 and fields[1].upper() == "'MARKER'":
                    continue
                cname = fields[0]
                if cname not in cols:
                    cols[cname] = {}
                    col_order.append(cname)
                pairs = fields[1:]
                for i in range(0, len(pairs) - 1, 2):
                    rname, val = pairs[i], float(pairs[i + 1])
                    cols[cname][rname] = val

            elif section == "RHS":
                pairs = fields[1:]
                for i in range(0, len(pairs) - 1, 2):
                    rname, val = pairs[i], float(pairs[i + 1])
                    rhs[rname] = val

            elif section == "RANGES":
                pairs = fields[1:]
                for i in range(0, len(pairs) - 1, 2):
                    rname, val = pairs[i], float(pairs[i + 1])
                    ranges[rname] = val

            elif section == "BOUNDS":
                btype = fields[0].upper()
                cname = fields[2]
                val = float(fields[3]) if len(fields) > 3 else None
                lo, hi = bounds.get(cname, [0.0, np.inf])
                if btype == "UP":
                    hi = val
                    if val < 0 and lo == 0.0:
                        lo = -np.inf
                elif btype == "LO":
                    lo = val
                elif btype == "FX":
                    lo = hi = val
                elif btype == "FR":
                    lo, hi = -np.inf, np.inf
                elif btype == "MI":
                    lo = -np.inf
                elif btype == "PL":
                    hi = np.inf
                elif btype in ("BV",):
                    lo, hi = 0.0, 1.0
                bounds[cname] = [lo, hi]

            elif section == "ENDATA":
                break

    n = len(col_order)
    var_index = {c: i for i, c in enumerate(col_order)}

    c = np.zeros(n)
    if obj_row is not None:
        for cname, rowmap in cols.items():
            if obj_row in rowmap:
                c[var_index[cname]] = rowmap[obj_row]

    A_ub_rows, b_ub_rows, ub_names = [], [], []
    A_eq_rows, b_eq_rows, eq_names = [], [], []

    for rname in row_order:
        rtype = row_type[rname]
        if rtype == "N":
            continue
        coeff = np.zeros(n)
        for cname, rowmap in cols.items():
            if rname in rowmap:
                coeff[var_index[cname]] = rowmap[rname]
        b = rhs.get(rname, 0.0)
        rng = ranges.get(rname)

        if rtype == "L":
            if rng is None:
                A_ub_rows.append(coeff)
                b_ub_rows.append(b)
                ub_names.append(rname)
            else:
                # b - |rng| <= row <= b
                A_ub_rows.append(coeff)
                b_ub_rows.append(b)
                ub_names.append(rname)
                A_ub_rows.append(-coeff)
                b_ub_rows.append(-(b - abs(rng)))
                ub_names.append(rname + "_lo")
        elif rtype == "G":
            if rng is None:
                A_ub_rows.append(-coeff)
                b_ub_rows.append(-b)
                ub_names.append(rname)
            else:
                # b <= row <= b + |rng|
                A_ub_rows.append(-coeff)
                b_ub_rows.append(-b)
                ub_names.append(rname)
                A_ub_rows.append(coeff)
                b_ub_rows.append(b + abs(rng))
                ub_names.append(rname + "_hi")
        elif rtype == "E":
            if rng is None:
                A_eq_rows.append(coeff)
                b_eq_rows.append(b)
                eq_names.append(rname)
            else:
                lo = b if rng >= 0 else b + rng
                hi = b + rng if rng >= 0 else b
                A_ub_rows.append(coeff)
                b_ub_rows.append(hi)
                ub_names.append(rname + "_hi")
                A_ub_rows.append(-coeff)
                b_ub_rows.append(-lo)
                ub_names.append(rname + "_lo")

    bnds = []
    for cname in col_order:
        lo, hi = bounds.get(cname, [0.0, np.inf])
        lo = None if lo == -np.inf else lo
        hi = None if hi == np.inf else hi
        bnds.append((lo, hi))

    return {
        "name": name,
        "c": c,
        "A_ub": np.array(A_ub_rows) if A_ub_rows else None,
        "b_ub": np.array(b_ub_rows) if b_ub_rows else None,
        "A_eq": np.array(A_eq_rows) if A_eq_rows else None,
        "b_eq": np.array(b_eq_rows) if b_eq_rows else None,
        "bounds": bnds,
        "var_names": col_order,
        "row_names": ub_names + eq_names,
    }
