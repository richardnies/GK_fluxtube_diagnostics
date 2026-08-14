"""Grid of g_vpa/mu contour plots across a 2D (row, col) sweep of runs.

Usage:
    python plot_gvmus_all_dirs.py <config.py>

<config.py> defines `dirnames` (2D list: dirnames[row][col], required),
`row_titles`, `col_titles` (required), and optionally `filename`, `code`,
`time_avg`, `kx_min`, `kx_max`, `nozonal`, `zonal`, `sharex`, `figname`.
Empty-string entries in `dirnames` are placeholders for missing
row/col combinations (skipped, matching the original scan_type blocks
which used "" for cells that don't correspond to a real run).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirnames", "row_titles", "col_titles"))

fontsize_labels = 32
nozonal = getattr(config, "nozonal", False)
zonal = getattr(config, "zonal", True)
kx_min = getattr(config, "kx_min", 0)
kx_max = getattr(config, "kx_max", 0.2)
time_avg = getattr(config, "time_avg", 300)
filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")

label_kx = "" if kx_min is None and kx_max is None else r"$(%.2f < |k_x| < %.2f)$" % (kx_min, kx_max)

Nrows = len(config.row_titles)
Ncols = len(config.col_titles)
fig, axs = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=(8 * Ncols, 8 * Nrows), sharex=getattr(config, "sharex", True))
# plt.subplots returns a 1D array (not 2D) when Nrows==1 or Ncols==1;
# np.atleast_2d always prepends the new axis as the row dimension, which
# is already correct for Nrows==1 but needs transposing back for Ncols==1.
axs = np.atleast_2d(axs)
if Ncols == 1 and Nrows > 1:
    axs = axs.reshape(Nrows, 1)

for i_row in range(Nrows):
    for i_col in range(Ncols):
        dirname = config.dirnames[i_row][i_col]
        if not dirname:
            continue

        ax = axs[i_row, i_col]
        try:
            run = StellaRun(dirname + "/" + filename, code=code)
            _, _, im = run.plot_contour_gvmu_vpa(
                time_idx=-1, logarithmic=True, vmin="symm", nozonal=nozonal, zonal=zonal,
                fig=fig, ax=ax, kx_min=kx_min, kx_max=kx_max, time_avg=time_avg,
            )
            plt.colorbar(im, ax=ax)
        except Exception as e:
            print("COULD NOT LOAD " + dirname)
            print(e)

if nozonal:
    fig.suptitle(r"$V^{-1}\int\mathrm{d}^3 r \;g_\mathrm{NZ}^2 / F_M$ " + label_kx)
elif zonal:
    fig.suptitle(r"$V^{-1}\int\mathrm{d}^3 r \;g_\mathrm{Z}^2 / F_M$ " + label_kx)
else:
    fig.suptitle(r"$V^{-1}\int\mathrm{d}^3 r \;g^2 / F_M$ " + label_kx)

for i_row in range(Nrows):
    axs[i_row, 0].set_ylabel(config.row_titles[i_row], fontsize=fontsize_labels)
for i_col in range(Ncols):
    axs[0, i_col].set_title(config.col_titles[i_col], fontsize=fontsize_labels)

plt.tight_layout()

figname = getattr(config, "figname", None)
if figname is None:
    figname = "fig_gvmus_all_dirs"
    if nozonal:
        figname += "_nozonal"
    if zonal:
        figname += "_zonal"
    figname += "_dtavg-%i" % time_avg if kx_min is None and kx_max is None else "_dtavg-%i_kxmin-%.2f_kxmax-%.2f" % (time_avg, kx_min, kx_max)
    figname += ".pdf"
fig.savefig(figname)
