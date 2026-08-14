"""Movie of several real-space (x) quantities overlaid on one axis vs time,
for one run, plus a time-averaged summary figure (one subplot per quantity).

Usage:
    python movie_quantities_x.py <config.py>

<config.py> defines `dirname` (required) and optionally `filename`, `code`,
`quantities`, `kx_orders`, `mults`, `mults_zed`, `colors`, `labels`,
`datanames` (six parallel lists describing each overlaid quantity --
defaults match the historically-active configuration, with `datanames`
de-duplicated: the original had two entries both named "tmp", which
silently made the second quantity's time average overwrite the first's
saved data -- get_quantities_x_tavg asserts datanames are unique instead),
`ylim`, `time_min`, `time_max`, `time_idx_step`, `time_avg`, `rerun_all`,
`fps`, `img_dir`.

`time_avg` here smooths each frame with a window CENTERED on that frame's
own time (quantities/realspace.py::get_quantity_zed_x_y) -- a different
convention from the trailing-window `time_avg` used by
stella_diagnostics.scan.zonal_flow_scan/rh_flux_scan and
physics.velocity_space.plot_contour_gvmu_vpa (kept distinct on purpose,
see get_quantities_x_tavg's docstring; only the field name is shared).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.plotting.movies import render_movie
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.quantities.realspace import get_quantity_x
from stella_diagnostics.scan.quantities_x_scan import get_quantities_x_tavg

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirname",))

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")

quantities = getattr(config, "quantities", ["phi", "dyphi-T", "Pi_RH_NL", "Pi_RH_even", "Pi_RH_odd", "dyphi-upar"])
kx_orders = getattr(config, "kx_orders", [2, 0, 0, 0, 0, 0])
mults = getattr(config, "mults", [-1, 1, 1, 1, 1, 1])
mults_zed = getattr(config, "mults_zed", [None] * len(quantities))
colors = getattr(config, "colors", ["0.5", "forestgreen", "k", "crimson", "mediumblue", "purple"])
labels = getattr(config, "labels", [r"$\gamma_E^Z$", r"$Q/Q_\mathrm{gB}$", r"$\Pi_\mathrm{RH}^\mathrm{NL}$", r"$\Pi_\mathrm{RH}^{\mathrm{NL},+}$", r"$\Pi_\mathrm{RH}^{\mathrm{NL},-}$", r"$\Pi_\parallel$"])
datanames = getattr(config, "datanames", ["gammaE", "Q", "Pi_RH_NL", "Pi_RH_even", "Pi_RH_odd", "Pi_parallel"])

ylim = getattr(config, "ylim", [-1.2, 1.2])
time_avg = getattr(config, "time_avg", None)

run = StellaRun(config.dirname + "/" + filename, code=code)

time_min = getattr(config, "time_min", 500)
time_max = getattr(config, "time_max", 2000)
time_idx_step = getattr(config, "time_idx_step", 2)
time_idx_min = run.get_time_idx(time_min)
time_idx_max = run.get_time_idx(time_max)
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)


# Plot-only display scaling, carried across frames: computed once from the
# first rendered frame's normalisation and then reused as a fixed
# multiplicative factor for every later frame (matching the original's
# plot_factors accumulator). Purely cosmetic -- doesn't feed into the
# cached time-averaged data below.
plot_factors = np.ones(len(quantities))


def frame_fn(i, time_idx_val):
    fig, ax = plt.subplots(figsize=(18, 9))
    for i_quantity, quantity in enumerate(quantities):
        _, _, norm_val, _, _ = run.plot_quantity_x(
            quantity=quantity, time_idx=time_idx_val, kx_order=kx_orders[i_quantity],
            fig=fig, ax=ax, label=labels[i_quantity], mult=mults[i_quantity],
            mult_zed=mults_zed[i_quantity], normalise=(i == 0), color=colors[i_quantity],
            time_avg=time_avg, plot_factor=plot_factors[i_quantity],
        )
        if i == 0:
            plot_factors[i_quantity] *= norm_val
    ax.set_ylim(ylim)
    ax.legend(bbox_to_anchor=(1.01, 1), fontsize=20, loc="upper left")
    plt.tight_layout()
    return fig


dirname_string = config.dirname.replace("/", "_")
img_dir = getattr(config, "img_dir", None)
if img_dir is None:
    img_dir = "fig_" + dirname_string + "_quantities_x"
    if time_avg is not None:
        img_dir += "_dtavg-%i" % time_avg

render_movie(
    img_dir, time_idx_vals, frame_fn,
    fps=getattr(config, "fps", int(30 * 5 / time_idx_step)),
    rerun_all=getattr(config, "rerun_all", True),
    video_name="video_quantities_x_" + dirname_string + ".mp4",
    on_error="continue",
)

# Time-averaged summary: one subplot per quantity, with a light overlay of
# every individual frame's profile (matching the original's
# overplot_time_trace=True) and an inset zoom.
means = get_quantities_x_tavg(run, quantities, kx_orders, mults, mults_zed, datanames, time_min=time_min, time_max=time_max, time_idx_step=time_idx_step, time_avg=time_avg)
x = means["x"]

overplot_quantity_idx = getattr(config, "overplot_quantity_idx", 0)

fig_tavg, axs_tavg = plt.subplots(nrows=len(quantities), figsize=(8, 5 * len(quantities)), sharex=True)
axs_tavg = np.atleast_1d(axs_tavg)

for i_quantity, dataname in enumerate(datanames):
    label = labels[i_quantity]
    ax = axs_tavg[i_quantity]
    avg = means[dataname]

    for time_idx_val in time_idx_vals[1:]:
        _, _, f_Z, _ = get_quantity_x(run, quantity=quantities[i_quantity], time_idx=time_idx_val, kx_order=kx_orders[i_quantity], mult=mults[i_quantity], mult_zed=mults_zed[i_quantity], normalise=False, time_avg=time_avg)
        ax.plot(x, f_Z, c="k", alpha=0.05)

    ax.plot(x, avg, c="crimson", lw=2)
    ax.axhline(np.mean(avg), c="crimson", alpha=0.5, ls="--")

    if overplot_quantity_idx is not None:
        overplot_avg = means[datanames[overplot_quantity_idx]]
        norm = np.nanmax(np.abs(avg)) / np.nanmax(np.abs(overplot_avg))
        ax.plot(x, norm * overplot_avg, c="forestgreen", ls=":", alpha=0.75, label=labels[overplot_quantity_idx])
        ax.legend(loc="upper left", fontsize=20)

    try:
        vmax = np.nanmax(np.abs(avg))
        x1, x2, y1, y2 = x[0], x[-1], -vmax * 1.1, vmax * 1.1
        axins = ax.inset_axes([0.7, 0.7, 0.27, 0.27], xlim=(x1, x2), ylim=(y1, y2))
        axins.plot(x, avg, c="crimson", lw=2)
        axins.axhline(np.mean(avg), c="crimson", alpha=0.5, ls="--")
        axins.grid(alpha=0.5)
        if overplot_quantity_idx is not None:
            axins.plot(x, norm * overplot_avg, c="forestgreen", ls=":", alpha=0.75, label=labels[overplot_quantity_idx])
    except Exception as e:
        print(e)

    ax.set_ylabel(label)

for ax in axs_tavg:
    ax.grid(alpha=0.5)
    ax.set_xlim([x[0], x[-1]])

axs_tavg[-1].set_xlabel(r"$x/\rho_i$")

fig_tavg.tight_layout()
fig_tavg.savefig("fig_quantities_x_tavg_" + dirname_string + ".pdf")
