"""phi(zed) comparison across an akyminmax x nfield_periods sweep.

Usage:
    python plot_compare_phi_zed.py <config.py>

<config.py> defines `akyminmax_vals`, `nfield_periods_vals`,
`filename_template` (%-format taking (aky_val, nfp_val)), and
`label_template` (%-format taking nfp_val), all required, plus optionally
`figname`.
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.run_collection import RunCollection

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(
    sys.argv[1],
    required=("akyminmax_vals", "nfield_periods_vals", "filename_template", "label_template"),
)

filenames = []
labels = []
for aky_val in config.akyminmax_vals:
    for nfp_val in config.nfield_periods_vals:
        filenames.append(config.filename_template % (aky_val, nfp_val))
        labels.append(config.label_template % nfp_val)

scan = RunCollection(filenames, labels)
scan.plot_phi_vs_zed(zed_times_nfield_periods=True)
plt.xlabel(r"$\zeta$")
plt.grid()
plt.legend()
plt.savefig(getattr(config, "figname", "fig_compare_phi_zed.png"))
