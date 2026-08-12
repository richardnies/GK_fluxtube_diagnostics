"""Flux(t) for one run.

Usage:
    python plot_fluxes.py <config.py>

<config.py> defines `dirname` (required, the run's filename_base) and
optionally `code`, `figname`.
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
axs = run.plot_flux_over_time()
axs[2].set_yscale("log")
plt.tight_layout()
plt.savefig(getattr(config, "figname", "fig_fluxes.png"))
