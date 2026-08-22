"""phi(zed) comparison across a set of runs.

Usage:
    python plot_compare_phi_zed.py <config.py>

<config.py> defines `dirnames` (required, flat list of run directories --
for an akyminmax x nfield_periods sweep, build this as a plain Python list
comprehension in the config file itself, e.g.
`dirnames = [tpl % (aky, nfp) for aky in akyminmax_vals for nfp in nfield_periods_vals]`)
and optionally `labels` (one per dirname, e.g. built the same way from a
label template), `filename`, `figname`.
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.run_collection import RunCollection

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirnames",))

filename = getattr(config, "filename", "CBC")
labels = getattr(config, "labels", [None] * len(config.dirnames))

scan = RunCollection([d + "/" + filename for d in config.dirnames], labels)
scan.plot_phi_vs_zed(zed_times_nfield_periods=True)
plt.xlabel(r"$\zeta$")
plt.grid()
plt.legend()
plt.savefig(getattr(config, "figname", "fig_compare_phi_zed.pdf"))
