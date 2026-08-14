"""Growth-rate (omega vs ky) convergence comparison across an
akyminmax x nfield_periods sweep, overlaid on shared axes.

Usage:
    python plot_compare_growth_rates.py <config.py>

<config.py> defines `akyminmax_vals`, `nfield_periods_vals` (both required)
and `filename_template` (a %-format string taking (aky_val, nfp_val), e.g.
"run_akyminmax-%.4f_nfield_periods-%.4f/precise_QA"), plus optionally
`codes`, `figname`.
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.run_collection import RunCollection

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("akyminmax_vals", "nfield_periods_vals", "filename_template"))

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(14, 9))

for nfield_periods_val in config.nfield_periods_vals:
    filenames = [config.filename_template % (aky_val, nfield_periods_val) for aky_val in config.akyminmax_vals]
    labels = [None] * len(filenames)

    scan = RunCollection(filenames, labels)
    scan.plot_omega_ky(axs=axs, label=r"$Nfp = $%i" % (nfield_periods_val))

axs[0].legend()
plt.savefig(getattr(config, "figname", "fig_comparison_growth_rates.pdf"))
