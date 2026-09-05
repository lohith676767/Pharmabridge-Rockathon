"""
Render a "sparsity fingerprint" of one or more GPU-format LP problems as
heatmap images -- every nonzero constraint-matrix entry as a colored
pixel, everything else blank -- so "sparse vs dense" is something a judge
can see at a glance instead of a number in a table.

Uses matplotlib's spy() for the actual plotting, which draws directly
from the sparse matrix's (row, col) nonzero positions -- it never
materializes a dense array, so this works even on a problem too large to
convert to dense (e.g. pds-100: 365,890 x 505,360, ~1.3M nonzeros).

Usage
-----
Single problem:
    python visualize_sparsity.py matrix_input.txt --out sparsity.png

Side-by-side comparison (the "make it concrete" demo image):
    python visualize_sparsity.py synthetic.txt ken-07.txt \
        --labels "Synthetic (block-diagonal)" "ken-07 (real Netlib)" \
        --out sparsity_comparison.png
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")  # no display needed -- just save an image file
import matplotlib.pyplot as plt
from scipy import sparse

from gpu_format import read_gpu_format


def load_full_matrix(filename: str) -> sparse.coo_matrix:
    """Load a GPU-format file and return its full constraint matrix (A_ub stacked on A_eq)."""
    c, A_ub, b_ub, A_eq, b_eq = read_gpu_format(filename)
    parts = [m for m in (A_ub, A_eq) if m is not None]
    if not parts:
        raise ValueError(f"{filename}: no constraint rows found")
    return sparse.vstack(parts, format="coo") if len(parts) > 1 else parts[0].tocoo()


def plot_sparsity(A: sparse.coo_matrix, title: str, ax, color: str = "#2563eb") -> None:
    """Draw one matrix's sparsity pattern onto a matplotlib Axes."""
    density = A.nnz / (A.shape[0] * A.shape[1]) if A.shape[0] and A.shape[1] else 0

    # A large, very sparse matrix needs tiny markers or the plot becomes
    # solid black; a small one needs bigger markers to be visible at all.
    marker_size = 0.3 if A.nnz > 200_000 else (1.0 if A.nnz > 20_000 else 3.0)

    ax.spy(A, markersize=marker_size, color=color, aspect="auto", origin="lower")
    ax.set_title(
        f"{title}\n{A.shape[0]:,} x {A.shape[1]:,}  |  {A.nnz:,} nonzeros  |  {density * 100:.4f}% dense",
        fontsize=10,
    )
    ax.set_xlabel("Variables")
    ax.set_ylabel("Constraints")
    ax.tick_params(labelsize=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="one or more GPU-format .txt files to visualize side by side")
    parser.add_argument("--labels", nargs="*", help="optional label per file (default: filename)")
    parser.add_argument("--out", default="sparsity_fingerprint.png", help="output image path")
    parser.add_argument("--dpi", type=int, default=150, help="output image resolution")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        raise ValueError("--labels must have exactly one label per file if given")
    labels = args.labels if args.labels else args.files

    n = len(args.files)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, filename, label in zip(axes, args.files, labels):
        print(f"Loading {filename} ...")
        A = load_full_matrix(filename)
        print(f"  {A.shape[0]:,} x {A.shape[1]:,}, {A.nnz:,} nonzeros -- plotting...")
        plot_sparsity(A, label, ax)

    fig.suptitle("Sparsity Fingerprint", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
