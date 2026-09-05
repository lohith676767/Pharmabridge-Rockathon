"""
AMPL (.mod + .dat) -> shared GPU-format COO file, via a translation
layer through AMPL's own engine -- not a hand-written parser.

Why a translator instead of a parser
-------------------------------------
Unlike MPS/LP/CSV, a .mod file has no matrix in it at all -- it's an
algebraic modeling language (sets, indexed sums, expressions like
`sum{i in FEEDSTOCKS} cost[i]*x[i] <= budget`) that only becomes a
constraint matrix once something actually *interprets and expands* it.
Writing that interpreter from scratch is not something to attempt here
-- AMPL's own engine already does it correctly. So this module: loads
the .mod (+ optional .dat) into AMPL via its official Python API
(amplpy), asks AMPL itself to write out an MPS file (`write m<stub>;`
-- a real AMPL command, not custom logic), then hands that MPS file to
the EXISTING, already-tested canonicalize_mps() from mps_to_txt.py. No
new matrix-building logic lives in this file at all.

Requirements
------------
    pip install amplpy
    python -m amplpy.modules install ampl   # free "Community Edition"
                                             # engine, no license key
                                             # needed, capped around
                                             # 500 variables / 500
                                             # constraints

Both steps need to reach AMPL's own module server (pypi.ampl.com) --
if that's blocked (e.g. behind a restrictive proxy/firewall), this
fails with a clear ImportError-style message rather than a confusing
one.

IMPORTANT -- not tested end-to-end
------------------------------------
amplpy's Python package was confirmed to install cleanly, and its API
(AMPL.read, AMPL.read_data, AMPL.write) was verified directly against
its real docstrings via `help()` -- this is not guessed syntax. But the
actual AMPL solving engine could not be downloaded in the sandboxed
environment this was written in (its module server, pypi.ampl.com, was
blocked by the local network proxy: 403 Forbidden). Test this for real
on a machine with normal internet access -- and with a real .mod/.dat
pair -- before relying on it for a live demo.

GAMS is deliberately NOT supported here: unlike AMPL, GAMS has no free,
pip-installable engine at all (its Python API requires a full licensed
GAMS installation), so a from-scratch GAMS translator can't be built or
tested the same way. If GAMS support becomes a real requirement, it
needs a licensed GAMS install as a prerequisite -- flag that
explicitly rather than silently pretending it's supported.

Usage
-----
    python ampl_to_gpu_format.py model.mod --out problem.txt
    python ampl_to_gpu_format.py model.mod --data model.dat --out problem.txt
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from gpu_format import export_to_gpu_format
from mps_to_txt import canonicalize_mps, write_meta


def canonicalize_ampl(mod_path: str, dat_path: str | None = None) -> dict:
    """Load a .mod (+ optional .dat) file, expand it via AMPL's own
    engine into an MPS file, then canonicalize that MPS exactly like a
    hand-written .mps file (reusing mps_to_txt.py's proven logic).

    Returns the same dict shape as canonicalize_mps() /
    canonicalize_csv() (see convert_to_gpu_format.py), so callers don't
    need to special-case AMPL beyond dispatch.
    """
    try:
        from amplpy import AMPL
    except ImportError as e:
        raise ImportError(
            "AMPL (.mod/.dat) support requires the amplpy package:\n"
            "  pip install amplpy\n"
            "  python -m amplpy.modules install ampl   "
            "# free engine, no license key needed"
        ) from e

    # CHANGED: amplpy imports fine even when the actual AMPL engine
    # binary isn't installed -- the failure only surfaces when AMPL()
    # tries to launch it, as a raw RuntimeError ("cannot execute
    # /x-ampl") that means nothing to someone who hasn't read amplpy's
    # internals. WHY: catch it here and re-raise with the same
    # actionable install instructions as the ImportError case above,
    # instead of letting an internal amplpy path leak into the error.
    try:
        ampl = AMPL()
    except RuntimeError as e:
        raise RuntimeError(
            "amplpy is installed, but the AMPL engine itself isn't. Run:\n"
            "  python -m amplpy.modules install ampl   "
            "# free Community Edition, no license key needed\n"
            f"(original error: {e})"
        ) from e

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ampl.cd(tmp_dir)
            ampl.read(str(mod_path))
            if dat_path:
                ampl.read_data(str(dat_path))

            stub = "gpu_export"
            ampl.write(f"m{stub}")  # AMPL's own `write` command: m-prefix = MPS format

            mps_path = Path(tmp_dir) / f"{stub}.mps"
            if not mps_path.exists():
                raise RuntimeError(
                    f"AMPL did not produce {mps_path.name} from {mod_path} -- "
                    "check that the model defines a solvable objective and "
                    "constraints (and that a .dat file was supplied if the "
                    ".mod file declares data-driven parameters/sets)."
                )

            model = canonicalize_mps(str(mps_path))
    finally:
        ampl.close()

    return model


def convert_file(mod_path, output_file, dat_path=None) -> dict:
    print(f"\nConverting: {mod_path}" + (f" (+ data: {dat_path})" if dat_path else ""))
    model = canonicalize_ampl(mod_path, dat_path)

    export_to_gpu_format(
        model["c"], model["A_ub"], model["b_ub"], model["A_eq"], model["b_eq"], output_file
    )
    meta_file = Path(output_file).with_suffix(".meta.json")
    write_meta(meta_file, model)

    nnz = (model["A_ub"].nnz if model["A_ub"] is not None else 0) + \
          (model["A_eq"].nnz if model["A_eq"] is not None else 0)
    size = model["n_row"] * model["n_col"]
    print(f"Output      : {output_file}")
    print(f"Metadata    : {meta_file}")
    print(f"Dimensions  : {model['n_row']} x {model['n_col']}")
    print(f"Nonzeros    : {nnz}")
    print(f"Density     : {nnz / size if size else 0:.8f}")
    print(f"Obj constant: {model['objective_constant']:.6f}")
    print("Conversion successful.")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="input .mod file")
    p.add_argument("--data", help="companion .dat file (optional)")
    p.add_argument("--out", help="output .txt file (default: same name, .txt extension)")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    if args.data and not Path(args.data).exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    output_path = Path(args.out) if args.out else input_path.with_suffix(".txt")
    convert_file(input_path, output_path, dat_path=args.data)


if __name__ == "__main__":
    main()
