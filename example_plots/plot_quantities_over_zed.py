"""Multiple quantities vs zed for one run.

Usage:
    python plot_quantities_over_zed.py <config.py>

<config.py> defines `dirname` (required) and optionally `code`, `kwargs`
(dict forwarded to plot_quantities_over_zed), `ylim`, `figname`.
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
fig, ax = run.plot_quantities_over_zed(**getattr(config, "kwargs", {}))

ax.legend()
ax.set_ylim(getattr(config, "ylim", None))
ax.grid()

plt.savefig(getattr(config, "figname", "fig_quantities_over_zed.png"))
