"""Movie of one quantity mapped onto a 3D torus surface, vs time, for one
run.

Usage:
    python movie_quantity_3d_torus.py <config.py>

<config.py> defines `dirname` (required) and optionally `filename`, `code`,
`quantity`, `remove_zonal`, `ky_order`, `nzeta`, `Delta_zeta`, `lighting`,
`cmap`, `xlim`, `ikymin`, `ikymax`, `time_min`, `time_max`,
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
quantity = getattr(config, "quantity", "density")
remove_zonal = getattr(config, "remove_zonal", False)
ky_order = getattr(config, "ky_order", 0)
nzeta = getattr(config, "nzeta", 100)
Delta_zeta = getattr(config, "Delta_zeta", np.pi * 0.6)
lighting = getattr(config, "lighting", True)
cmap = getattr(config, "cmap", "coolwarm")
xlim = getattr(config, "xlim", None)
ikymin = getattr(config, "ikymin", 0)
ikymax = getattr(config, "ikymax", None)

run = StellaRun(config.dirname + "/" + filename, code=code)

time_idx_min = run.get_time_idx(getattr(config, "time_min", 10000))
time_idx_max = run.get_time_idx(getattr(config, "time_max", 10500))
time_idx_step = getattr(config, "time_idx_step", 2)
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)

# vmin/vmax fixed to None, matching the original (the reference-frame
# rescale that would set them was already commented out there).
vmin = vmax = None


def frame_fn(i, time_idx_val):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig, _, _, _ = run.plot_quantity_3d_torus(
        quantity=quantity, time_idx=time_idx_val, remove_zonal=remove_zonal, ky_order=ky_order,
        nzeta=nzeta, fig=fig, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, Delta_zeta=Delta_zeta,
        xlim=xlim, lighting=lighting, ikymin=ikymin, ikymax=ikymax,
    )
    plt.tight_layout()
    return fig


dirname_string = config.dirname.replace("/", "_")
img_dir = getattr(config, "img_dir", None)
if img_dir is None:
    img_dir = "fig_" + dirname_string + "_" + quantity + "_real_space_3d_torus"
    if ky_order != 0:
        img_dir += "_ky-order-%i" % (ky_order)
    if remove_zonal:
        img_dir += "_no_zonal"

render_movie(
    img_dir, time_idx_vals, frame_fn,
    fps=getattr(config, "fps", 30),
    rerun_all=getattr(config, "rerun_all", True),
    video_name="video_" + quantity + "_real_space.mp4",
    on_error="continue",
)
