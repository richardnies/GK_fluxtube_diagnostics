"""|phi|(zeta, t) contour grid across a 2D (row, col) sweep of runs.

Usage:
    python plot_contour_phi_vs_t_zed.py <config.py>

<config.py> defines `dirnames` (2D list: dirnames[row][col], required),
`row_titles`, `col_titles` (required -- matching
plot_gvmus_all_dirs.py's grid convention; for a zeta_center x akyminmax
sweep, format these directly in the config file, e.g.
`col_titles = [r"$k_y\rho_i=%.2f$" % v for v in akyminmax_vals]`) and
optionally `filename`, `code`, `figname_add`, `figname`.

tprim (for the figure suptitle/filename) is read directly from the first
resolved run's own netCDF output, not supplied separately.

NOTE: plot_contour_phi_zed_t does not exist on StellaRun/stellaDiagnostics
(pre-existing bug, predates the restructure -- see README "Known issues").
The closest current equivalents are plot_quantity_zed_t and
RunCollection.plot_contour_phi_vs_zed_theta0. Preserved as-is here (not
silently fixed) since this script has been broken since before this
migration.
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

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
figname_add = getattr(config, "figname_add", "")

Nrows = len(config.row_titles)
Ncols = len(config.col_titles)
fig, axs = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=(18, 20))
# plt.subplots returns a bare Axes (Nrows==Ncols==1) or a 1D array
# (exactly one of Nrows/Ncols is 1), not the 2D array axs[row, col]
# below assumes; np.atleast_2d normalizes both cases.
axs = np.atleast_2d(axs)
if Ncols == 1 and Nrows > 1:
    axs = axs.reshape(Nrows, 1)

tprim = None
for i_row in range(Nrows):
    for i_col in range(Ncols):
        run = StellaRun(config.dirnames[i_row][i_col] + "/" + filename, code=code)
        if tprim is None:
            tprim = float(run.ncdata.variables["tprim"][0])

        ax = axs[i_row, i_col]
        run.plot_contour_phi_zed_t(fig, ax, normalise_phi=True)
        title = config.row_titles[i_row] + ", " + config.col_titles[i_col]
        ax.set_title(title)

fig.suptitle(r"$|\phi|(\zeta, t)$ for $a/L_T =$ %.1f" % tprim if tprim is not None else "")
plt.tight_layout()
plt.savefig(getattr(config, "figname", None) or "fig_contours_phi_zed_tprim_%.1f" % tprim + figname_add + ".pdf")
