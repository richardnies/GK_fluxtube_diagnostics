"""Movie of g_NZ(zed, vpa) and g_Z(zed, vpa) side by side, vs time, for one run.

Usage:
    python movie_gzvs_Z-NZ.py <config.py>

<config.py> defines `dirname` (required) and optionally `filename`,
`code`, `time_min`, `time_max`, `time_idx_step`, `rerun_all`, `fps`,
`img_dir`.
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

run = StellaRun(config.dirname + "/" + filename, code=code)

time_idx_min = run.get_time_idx(getattr(config, "time_min", 0))
time_idx_max = run.get_time_idx(getattr(config, "time_max", 1e10))
time_idx_step = getattr(config, "time_idx_step", 10)
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)


def frame_fn(i, time_idx_val):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(24, 9))

    ax = axs[0]
    _, _, im = run.plot_contour_gzvs(time_idx=time_idx_val, logarithmic=True, vmin="symm", nozonal=True, zonal=False, fig=fig, ax=ax)
    plt.colorbar(im, ax=ax)
    ax.set_title(r"$\int\mathrm{d}^2 r_\perp\int\mathrm{d}v_\perp \; g_\mathrm{NZ}^2$")

    ax = axs[1]
    _, _, im = run.plot_contour_gzvs(time_idx=time_idx_val, logarithmic=True, vmin="symm", nozonal=False, zonal=True, fig=fig, ax=ax)
    plt.colorbar(im, ax=ax)
    ax.set_title(r"$\int\mathrm{d}^2 r_\perp\int\mathrm{d}v_\perp \; g_\mathrm{Z}^2$")

    return fig


dirname_string = config.dirname.replace("/", "_")
img_dir = getattr(config, "img_dir", None) or "fig_" + dirname_string + "_gzvs_Z-NZ"

render_movie(
    img_dir, time_idx_vals, frame_fn,
    fps=getattr(config, "fps", 30),
    rerun_all=getattr(config, "rerun_all", True),
    video_name="video_gzvs_t_" + dirname_string + ".mp4",
)
