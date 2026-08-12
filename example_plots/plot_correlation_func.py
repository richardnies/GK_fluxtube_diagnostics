"""Parallel correlation function for one run.

Usage:
    python plot_correlation_func.py <config.py>

<config.py> defines `dirname` (required) and optionally `code`, `figname`.
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirname",))

run = StellaRun(config.dirname, code=getattr(config, "code", "stella"))
# NOTE: pre-existing bug (predates this restructure, confirmed present in
# the original repo's very first commit) -- plot_parallel_correlation_function
# returns 5 values (fig, ax, im, avg_delta_chi, k), not 3, so this always
# raises ValueError. Preserved as-is per this project's flag-not-fix
# convention for pre-existing bugs; see README "Known issues".
fig, ax, im = run.plot_parallel_correlation_function()

plt.tight_layout()
plt.savefig(getattr(config, "figname", "fig_correlation_func.png"), dpi=800)
