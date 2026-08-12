"""Movie of g(vpa, mu) vs time for one run.

Usage:
    python movie_gvmus_t.py <config.py>

<config.py> defines `dirname` (required) and optionally `filename`,
`code`, `time_min`, `time_max`, `time_idx_step`, `zonal`, `nozonal`,
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
config = load_scan_config(sys.argv[1], required=("dirname",))

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
zonal = getattr(config, "zonal", False)
nozonal = getattr(config, "nozonal", True)

run = StellaRun(config.dirname + "/" + filename, code=code)

time_idx_min = run.get_time_idx(getattr(config, "time_min", 0))
time_idx_max = run.get_time_idx(getattr(config, "time_max", 1e5))
time_idx_step = getattr(config, "time_idx_step", 100)
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)


def frame_fn(i, time_idx_val):
    fig, ax, im = run.plot_contour_gvmu_vpa(time_idx=time_idx_val, logarithmic=True, vmin="symm", nozonal=nozonal, zonal=zonal)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig


img_dir = getattr(config, "img_dir", None)
if img_dir is None:
    img_dir = "fig_" + config.dirname.replace("/", "_") + "_gvmus_t"
    if zonal:
        img_dir += "_zonal"
    elif nozonal:
        img_dir += "_nozonal"

render_movie(
    img_dir, time_idx_vals, frame_fn,
    fps=getattr(config, "fps", 5),
    rerun_all=getattr(config, "rerun_all", True),
    video_name="video_gvmus_t.mp4",
    # original used -r 5 (input) -vf fps=30 (output): a deliberate 6x
    # slowdown via frame duplication, not just a plain 5fps encode.
    ffmpeg_extra_args=["-vf", "fps=30"],
)
