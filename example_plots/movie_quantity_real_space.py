"""Movie of one quantity in real (x, y) space, at several zed slices side by
side, vs time, for one run -- with an optional overlay of the zonal
(kx-only) profile of a few reference quantities.

Usage:
    python movie_quantity_real_space.py <config.py>

<config.py> defines `dirname` (required) and optionally `filename`, `code`,
`quantity`, `zed_vals`, `remove_zonal`, `overplot_zonal`, `ky_order`,
`mult_zed`, `nx_padded`, `ny_padded`, `time_min`, `time_max`,
`time_idx_step`, `rerun_all`, `fps`, `img_dir`.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.plotting.movies import render_movie
from stella_diagnostics.scan.config import load_scan_config

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirname",))

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
quantity = getattr(config, "quantity", "upar")
zed_vals = getattr(config, "zed_vals", [0, np.pi / 2, np.pi])
remove_zonal = getattr(config, "remove_zonal", True)
overplot_zonal = getattr(config, "overplot_zonal", True)
ky_order = getattr(config, "ky_order", 0)
mult_zed = getattr(config, "mult_zed", None)
nx_padded = getattr(config, "nx_padded", None)
ny_padded = getattr(config, "ny_padded", None)

run = StellaRun(config.dirname + "/" + filename, code=code)

time_idx_min = run.get_time_idx(getattr(config, "time_min", 0))
time_idx_max = run.get_time_idx(getattr(config, "time_max", 1e6))
time_idx_step = getattr(config, "time_idx_step", 1)
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)

# vmin/vmax fixed from the last timestep, matching the original.
_, _, _, vmin, vmax = run.plot_quantity_x_y(quantity=quantity, time_idx=-1, remove_zonal=remove_zonal, ky_order=ky_order, nx=nx_padded, ny=ny_padded, symm=True, zed_val=0)

# Normalisations for the zonal overlay, fixed from the last timestep.
if overplot_zonal:
    dxphizonal, _, y, _ = run.get_quantity_x_y(quantity="phi", time_idx=-1, only_zonal=True, kx_order=1, nx=nx_padded)
    norm_dxphizonal = y[-1] / (np.abs(dxphizonal)).max() / 4
    uparzonal, _, y, _ = run.get_quantity_x_y(quantity="upar", time_idx=-1, only_zonal=True, kx_order=0, nx=nx_padded)
    norm_uparzonal = y[-1] / (np.abs(uparzonal)).max() / 4


def frame_fn(i, time_idx_val):
    fig, axs = plt.subplots(ncols=len(zed_vals), figsize=(9 * len(zed_vals), 9))
    axs = np.atleast_1d(axs)
    for i_zed, zed_val in enumerate(zed_vals):
        ax = axs[i_zed]
        _, _, im, _, _ = run.plot_quantity_x_y(quantity=quantity, time_idx=time_idx_val, remove_zonal=remove_zonal, ky_order=ky_order, nx=nx_padded, ny=ny_padded, vmin=vmin, vmax=vmax, fig=fig, ax=ax, zed_val=zed_val)
        fig.colorbar(im, ax=ax)
        ax.set_aspect("equal")
        if overplot_zonal:
            dxphizonal, x, y, _ = run.get_quantity_x_y(quantity="phi", time_idx=time_idx_val, only_zonal=True, kx_order=1, nx=nx_padded, zed_val=zed_val)
            uparzonal, x, y, _ = run.get_quantity_x_y(quantity="upar", time_idx=time_idx_val, only_zonal=True, kx_order=0, nx=nx_padded, zed_val=zed_val)
            # NOTE: pre-existing dead code (predates the restructure,
            # confirmed via git history) -- tprim_lin was computed and
            # then only referenced inside commented-out lines below;
            # preserved as-is rather than silently dropped or "fixed" by
            # guessing what it should feed into.
            tprim = run.ncdata.variables["tprim"][0]
            tprim_lin = (1 + 1) * (1.33 + 1.91 * 0.8 / 1.4) * (1 - 1.5 * 0.18)

            ax.plot(x, -dxphizonal[:, 0] * norm_dxphizonal, c="forestgreen")
            ax.plot(x, uparzonal[:, 0] * norm_uparzonal, c="c")
            ax.set_ylim([y[0], y[-1]])
            ax.set_xlim([x[0], x[-1]])
        title = r"$\theta$ avg" if zed_val is None else r"$\theta = %.2f$" % (zed_val)
        ax.set_title(title)
    plt.tight_layout()
    return fig


dirname_string = config.dirname.replace("/", "_")
img_dir = getattr(config, "img_dir", None)
if img_dir is None:
    img_dir = "fig_" + dirname_string + "_" + quantity + "_real_space"
    if ky_order != 0:
        img_dir += "_ky-order-%i" % (ky_order)
    if mult_zed is not None:
        img_dir += "_mult_zed-" + mult_zed
    if remove_zonal:
        img_dir += "_no_zonal"

render_movie(
    img_dir, time_idx_vals, frame_fn,
    fps=getattr(config, "fps", 20),
    rerun_all=getattr(config, "rerun_all", False),
    video_name="video_" + quantity + "_real_space.mp4",
    on_error="continue",
)
