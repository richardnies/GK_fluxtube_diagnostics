"""Growth-rate (omega vs ky) convergence comparison across a set of
series, overlaid on shared axes -- one series per outer `dirnames` entry
(e.g. an akyminmax sweep at fixed nfield_periods).

Usage:
    python plot_compare_growth_rates.py <config.py>

<config.py> defines `dirnames` (required, nested list: dirnames[i_series]
= flat list of run directories in that series -- for an akyminmax x
nfield_periods sweep, build this as a plain Python list comprehension in
the config file itself, e.g.
`dirnames = [[tpl % (aky, nfp) for aky in akyminmax_vals] for nfp in nfield_periods_vals]`)
and optionally `series_labels` (one per series, e.g. `r"$Nfp=%i$" % nfp`),
`filename`, `figname`.
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
series_labels = getattr(config, "series_labels", [None] * len(config.dirnames))

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(14, 9))

for i_series, series_dirnames in enumerate(config.dirnames):
    filenames = [d + "/" + filename for d in series_dirnames]
    labels = [None] * len(filenames)

    scan = RunCollection(filenames, labels)
    scan.plot_omega_ky(axs=axs, label=series_labels[i_series])

axs[0].legend()
plt.savefig(getattr(config, "figname", "fig_comparison_growth_rates.pdf"))
