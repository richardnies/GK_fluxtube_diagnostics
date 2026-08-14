"""Flux-tube geometry comparison across a set of runs.

Usage:
    python plot_geometry_compare_flux_tubes.py <config.py>

<config.py> defines `dirnames` (required, run directories -- `filename`,
default "CBC", is appended to each to build the RunCollection's
filename_base list, same dirname/filename split as every other multi-run
driver in this package) and optionally `filename`, `labels`, `codes`,
`kwargs` (forwarded to plot_comparison_flux_tube_geometry), `figname`.
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.run_collection import RunCollection

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1])

filename = getattr(config, "filename", "CBC")
scan = RunCollection(
    [d + "/" + filename for d in config.dirnames],
    labels=getattr(config, "labels", None),
    codes=getattr(config, "codes", None),
)
axs = scan.plot_comparison_flux_tube_geometry(**getattr(config, "kwargs", {"zed_times_nfield_periods": True}))

plt.savefig(getattr(config, "figname", "fig_comparison_flux_tube_geometry.pdf"))
