"""quantity(x, t) contour grid across a set of run directories, with a
Qflx/phi2 overlay on each panel.

Usage:
    python plot_contour_quantity_vs_t_x.py <config.py>

<config.py> defines `dirnames` (required, flat list of run directories --
if a real scan wants glob-based discovery, call
stella_diagnostics.scan.config.discover_runs(...) directly in the config
file and assign the result to `dirnames`) and optionally `filename`,
`code`, `quantity`, `kx_order`, `mult_zed`, `y_val`, `time_min`/`time_max`
(scalars, broadcast) or `time_min_vals`/`time_max_vals` (per-run lists),
`time_idx_skip`, `normalise`, `only_zonal`, `remove_zonal`, `cmap`, `vmin`,
`vmax`, `logarithmic`, `figname`.

Unlike the original script, the quantity-dependent preset selection
(which vmin/vmax/logarithmic/mult_zed/normalise go with which quantity)
lives in the config, not in this driver -- pick the values that match
whichever quantity you're plotting.
"""
import os
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.quantity_x_t_overlay import add_qflx_phi2_overlay, get_quantity_x_t_title

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirnames",))

quantity = getattr(config, "quantity", "phi")
filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
kx_order = getattr(config, "kx_order", 0)
mult_zed = getattr(config, "mult_zed", None)
y_val = getattr(config, "y_val", None)
time_idx_skip = getattr(config, "time_idx_skip", 1)
normalise = getattr(config, "normalise", False)
only_zonal = getattr(config, "only_zonal", False)
remove_zonal = getattr(config, "remove_zonal", False)
cmap = getattr(config, "cmap", "coolwarm")
vmin = getattr(config, "vmin", "symm")
vmax = getattr(config, "vmax", "last")
logarithmic = getattr(config, "logarithmic", False)

dirnames = config.dirnames
n_dirs = len(dirnames)
time_min_vals = getattr(config, "time_min_vals", [getattr(config, "time_min", 0)] * n_dirs)
time_max_vals = getattr(config, "time_max_vals", [getattr(config, "time_max", 5000)] * n_dirs)

fig, axs = plt.subplots(ncols=n_dirs, figsize=(8 * n_dirs, 8))

for i_dirname, dirname in enumerate(dirnames):
    try:
        run = StellaRun(dirname + "/" + filename, code=code)
        ax = axs if n_dirs == 1 else axs[i_dirname]

        fig, ax, im, _, _, _ = run.plot_quantity_x_t(
            quantity=quantity, fig=fig, ax=ax, remove_zonal=remove_zonal, only_zonal=only_zonal,
            vmin=vmin, vmax=vmax, cmap=cmap, logarithmic=logarithmic, normalise_each_t=normalise,
            time_idx_skip=time_idx_skip, y_val=y_val, kx_order=kx_order, mult_zed=mult_zed,
            time_min=time_min_vals[i_dirname], time_max=time_max_vals[i_dirname],
        )
        # NOTE: the original script built this title as
        # dirname[len(dirname_base):], which silently mangled it (e.g.
        # "run_tprim-4.2000" -> "un_tprim-4.2000") whenever the base
        # directory prefix wasn't a literal string prefix of dirname (e.g.
        # after glob-discovery normalized away a bare "." prefix).
        # os.path.basename is robust to that regardless of how dirname is
        # spelled or where it came from.
        ax.set_title(os.path.basename(dirname.rstrip("/")), fontsize=14)
        plt.colorbar(im, ax=ax)

        add_qflx_phi2_overlay(ax, run)

    except Exception as e:
        print(e)
        print("Could not load " + dirname)

title = get_quantity_x_t_title(quantity, kx_order=kx_order, remove_zonal=remove_zonal, only_zonal=only_zonal, mult_zed=mult_zed)
fig.suptitle(title)
plt.tight_layout()

figname = getattr(config, "figname", None)
if figname is None:
    figname = "fig_contours_" + quantity + "_x_t"
    if remove_zonal:
        figname += "_remove_zonal"
    if only_zonal:
        figname += "_only_zonal"
    if not normalise:
        figname += "_unnormalised"
    if kx_order > 0:
        figname += "_kxorder-%i" % kx_order
    if mult_zed == "vdriftx":
        figname += "_vdriftx"
    figname += ".pdf"
plt.savefig(figname)
