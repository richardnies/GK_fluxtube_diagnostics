"""Movie of several real-space (zed, x) quantity contours side by side vs
time, for one run, plus a time-averaged summary figure (one subplot per
quantity).

Usage:
    python movie_quantities_x_zed.py <config.py>

<config.py> defines `dirname` (required) and optionally `filename`, `code`,
`quantities`, `kx_orders`, `mults`, `labels`, `datanames` (parallel lists
describing each quantity -- defaults match the historically-active
configuration), `time_min`, `time_max`, `time_idx_step`, `rerun_all`,
`fps`, `img_dir`, `overplot_quantity_idx`.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.plotting.movies import render_movie
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.quantities_x_scan import get_quantities_x_zed_tavg

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirname",))

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")

quantities = getattr(config, "quantities", ["phi", "RH_phi", "dyphi-upar", "dyphi-T", "dyphi2", "P_RH_tot", "P_RH_NL", "P_RH_even", "P_RH_odd", "P_RH_coll", "upar", "temperature"])
kx_orders = getattr(config, "kx_orders", [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
mults = getattr(config, "mults", [-1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
labels = getattr(config, "labels", [r"$v_E^Z$", r"$v_E^{\mathrm{RH},Z}$", r"$\Pi_\parallel$", r"$Q$", r"$v_{Ex}^2$", r"$P_\mathrm{RH}$", r"$P_\varphi$", r"$P_\varphi^+$", r"$P_\varphi^-$", r"$P_{\mathrm{RH}}^C$", r"$u_\parallel^Z$", r"$\partial_x T^Z$"])
datanames = getattr(config, "datanames", ["vE", "vE_RH", "Pi_parallel", "Q", "vEx2", "P_RH", "P_RH_phi", "P_phi_even", "P_phi_odd", "P_RH_coll", "upar", "gradTZ"])

overplot_quantity_idx = getattr(config, "overplot_quantity_idx", 0)

run = StellaRun(config.dirname + "/" + filename, code=code)

time_min = getattr(config, "time_min", 500)
time_max = getattr(config, "time_max", 1e6)
time_idx_step = getattr(config, "time_idx_step", 2)
time_idx_min = run.get_time_idx(time_min)
time_idx_max = run.get_time_idx(time_max)
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)


def frame_fn(i, time_idx_val):
    fig, axs = plt.subplots(ncols=len(quantities), figsize=(8 * len(quantities), 8), sharey=True)
    axs = np.atleast_1d(axs)
    for i_quantity, quantity in enumerate(quantities):
        ax = axs[i_quantity]
        _, _, im, x, zed, f_zed_x, zed_avg_x, vmin, vmax = run.plot_quantity_x_zed(
            quantity=quantity, time_idx=time_idx_val, kx_order=kx_orders[i_quantity],
            fig=fig, ax=ax, mult_fac=mults[i_quantity], vmin="symm", cmap="coolwarm", only_zonal=True,
        )
        ax.axvline(0, c="k", alpha=0.75)
        fig.colorbar(im, ax=ax)
        ax.set_title(labels[i_quantity])
    plt.tight_layout()
    return fig


dirname_string = config.dirname.replace("/", "_")
img_dir = getattr(config, "img_dir", None)
if img_dir is None:
    img_dir = "fig_" + dirname_string + "_quantities_x_zed"

render_movie(
    img_dir, time_idx_vals, frame_fn,
    fps=getattr(config, "fps", 15),
    rerun_all=getattr(config, "rerun_all", True),
    video_name="video_quantities_x_zed" + dirname_string + ".mp4",
    on_error="continue",
)

# Time-averaged summary: one (zed, x) contour subplot per quantity.
means = get_quantities_x_zed_tavg(run, quantities, kx_orders, mults, datanames, time_min=time_min, time_max=time_max, time_idx_step=time_idx_step)
x = means["x"]
zed = means["zed"]

fig_tavg, axs_tavg = plt.subplots(ncols=len(quantities), figsize=(8 * len(quantities), 8), sharey=True)
axs_tavg = np.atleast_1d(axs_tavg)

for i_quantity, dataname in enumerate(datanames):
    label = labels[i_quantity]
    ax = axs_tavg[i_quantity]
    avg_zed_x = means[dataname]

    _, _, im, _, _, _, _, vmin, vmax = run.plot_quantity_x_zed(quantity=avg_zed_x, fig=fig_tavg, ax=ax, vmin="symm", cmap="coolwarm")
    ax.axvline(0, c="k", alpha=0.75)
    fig_tavg.colorbar(im, ax=ax)
    ax.set_title(label)

    if overplot_quantity_idx is not None:
        dl_over_B_avg = run.dl_over_B_avg()
        overplot_x = np.sum(means[datanames[overplot_quantity_idx]] * dl_over_B_avg[:, None], axis=0)
        norm = (zed[-1] / 2) / np.nanmax(np.abs(overplot_x))
        rot = transforms.Affine2D().rotate_deg(90)
        base = ax.transData
        ax.plot(x, -norm * overplot_x, c="k", ls=":", alpha=0.75, label=labels[overplot_quantity_idx], transform=rot + base)
        ax.legend(loc="upper left", fontsize=20)

fig_tavg.tight_layout()
fig_tavg.savefig("fig_quantities_x_zed_tavg_" + dirname_string + ".pdf")
