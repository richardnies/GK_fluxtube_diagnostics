"""|phi|(zeta, t) contour grid across a zeta_center x akyminmax sweep.

Usage:
    python plot_contour_phi_vs_t_zed.py <config.py>

<config.py> defines `base_dirs`, `base_dir_titles`, `akyminmax_vals`,
`tprim_val`, `filename_template` (%-format taking (base_dir, ky_val,
tprim_val)), all required, plus optionally `figname`.

NOTE: plot_contour_phi_zed_t does not exist on StellaRun/stellaDiagnostics
(pre-existing bug, predates the restructure -- see README "Known issues").
The closest current equivalents are plot_quantity_zed_t and
RunCollection.plot_contour_phi_vs_zed_theta0. Preserved as-is here (not
silently fixed) since this script has been broken since before this
migration.
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(
    sys.argv[1],
    required=("base_dirs", "base_dir_titles", "akyminmax_vals", "tprim_val", "filename_template"),
)

Ncols = len(config.akyminmax_vals)
Nrows = len(config.base_dirs)
fig, axs = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=(18, 20))

for i_base_dir, base_dir in enumerate(config.base_dirs):
    for i_ky, ky_val in enumerate(config.akyminmax_vals):
        filename = config.filename_template % (base_dir, ky_val, config.tprim_val)
        run = StellaRun(filename, code=getattr(config, "code", "stella"))

        ax = axs[i_base_dir, i_ky]
        run.plot_contour_phi_zed_t(fig, ax, normalise_phi=True)
        title = config.base_dir_titles[i_base_dir] + r", $k_y \rho_i = $ %.2f" % (ky_val)
        ax.set_title(title)

fig.suptitle(r"$|\phi|(\zeta, t)$ for $a/L_T =$ %i" % (config.tprim_val))
plt.tight_layout()
plt.savefig(getattr(config, "figname", None) or "fig_contours_phi_zed_tprim_%i.png" % (config.tprim_val))
