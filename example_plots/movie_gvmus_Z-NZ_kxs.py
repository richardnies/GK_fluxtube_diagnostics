"""Movie of g_NZ/g_Z(vpa, mu) side by side, one row per kx band, vs time,
for one run.

Usage:
    python movie_gvmus_Z-NZ_kxs.py <config.py>

<config.py> defines `dirname`, `kx_mins`, `kx_maxs` (required, `kx_mins`/
`kx_maxs` define one row per kx band) and optionally `filename`, `code`,
`vmin`, `vmax`, `time_min`, `time_max`, `time_idx_step`, `time_avg`,
`rerun_all`, `fps`, `img_dir`.
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
config = load_scan_config(sys.argv[1], required=("dirname", "kx_mins", "kx_maxs"))

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
vmin = getattr(config, "vmin", "symm")
vmax = getattr(config, "vmax", None)
time_avg = getattr(config, "time_avg", None)
Nrows = len(config.kx_mins)

run = StellaRun(config.dirname + "/" + filename, code=code)

time_idx_min = run.get_time_idx(getattr(config, "time_min", 0))
time_idx_max = run.get_time_idx(getattr(config, "time_max", 1e10))
time_idx_step = getattr(config, "time_idx_step", 5)
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)


def frame_fn(i, time_idx_val):
    fig, axs = plt.subplots(nrows=Nrows, ncols=2, figsize=(24, 9 * Nrows))
    # plt.subplots returns a 1D array (not 2D) when Nrows==1; np.atleast_2d
    # gives it the row dimension needed for axs[irow, 0]/axs[irow, 1] below.
    axs = np.atleast_2d(axs)

    for irow in range(Nrows):
        kx_min = config.kx_mins[irow]
        kx_max = config.kx_maxs[irow]
        label_kx = r"$(%.2f < |k_x| < %.2f)$" % (kx_min, kx_max)

        ax = axs[irow, 0]
        _, _, im = run.plot_contour_gvmu_vpa(time_idx=time_idx_val, vmin=vmin, vmax=vmax, logarithmic=True, nozonal=True, zonal=False, fig=fig, ax=ax, kx_min=kx_min, kx_max=kx_max, time_avg=time_avg)
        plt.colorbar(im, ax=ax)
        ax.set_title(r"$V^{-1}\int\mathrm{d}^3 r \;g_\mathrm{NZ}^2/F_M$ " + label_kx)

        ax = axs[irow, 1]
        _, _, im = run.plot_contour_gvmu_vpa(time_idx=time_idx_val, vmin=vmin, vmax=vmax, logarithmic=True, nozonal=False, zonal=True, fig=fig, ax=ax, kx_min=kx_min, kx_max=kx_max, time_avg=time_avg)
        plt.colorbar(im, ax=ax)
        ax.set_title(r"$V^{-1}\int\mathrm{d}^3 r \;g_\mathrm{Z}^2/F_M$ " + label_kx)

    return fig


dirname_string = config.dirname.replace("/", "_")
img_dir = getattr(config, "img_dir", None)
if img_dir is None:
    img_dir = "fig_" + dirname_string + "_gvmus_Z-NZ_kxs"
    if time_avg is not None:
        img_dir += "_dtavg-%i" % time_avg

render_movie(
    img_dir, time_idx_vals, frame_fn,
    fps=getattr(config, "fps", 30),
    rerun_all=getattr(config, "rerun_all", True),
    video_name="video_gvmus_t_kxs" + dirname_string + ".mp4",
    on_error="continue",
)
