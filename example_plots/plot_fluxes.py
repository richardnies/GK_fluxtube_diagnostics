"""Flux(t) for one run.

Usage:
    python plot_fluxes.py <config.py>

<config.py> defines `dirname` (required, the run's directory) and
optionally `filename`, `code`, `figname_add`, `figname` -- same
dirname/filename split as every other single-run driver in this package
(changed from treating dirname as the full filename_base, for
config-vocabulary consistency).
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

run = StellaRun(config.dirname + "/" + getattr(config, "filename", "CBC"), code=getattr(config, "code", "stella"))
axs = run.plot_flux_over_time()
# NOTE: heat flux Q can go negative during transients (a plain log scale
# would silently drop those points, showing a blank plot for that whole
# stretch -- same bug class as rh_flux_scan.py/plot_correlation_func.py,
# fixed the same way: symlog handles the sign change, still log-like for
# the many-orders-of-magnitude span once it settles positive).
axs[2].set_yscale("symlog")
plt.tight_layout()
figname_add = getattr(config, "figname_add", "")
plt.savefig(getattr(config, "figname", None) or "fig_fluxes" + figname_add + ".pdf")
