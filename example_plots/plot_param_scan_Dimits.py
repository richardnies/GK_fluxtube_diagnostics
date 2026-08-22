"""15-panel R/L_T scan comparison: heat flux, ExB shear, Rosenbluth-Hinton
power transfer, and zonal-flow diagnostics vs R/L_T, one series per outer
dirnames entry.

Usage:
    python plot_param_scan_Dimits.py <config.py>

<config.py> defines `dirnames`, `labels` (required; `dirnames` is a nested
list, dirnames[i_series] = flat list of run directories in that series --
tprim is read directly from each run's own netCDF output, not supplied
separately; a config that wants a single run instead of a whole series
just passes a single-entry inner list) and optionally `filename`, `code`,
`aLT_lin_vals`, `base_colors`, `tprim_exclude`, `substract_lin`, `xlim`,
`markersize`, `time_val_avg`, `time_avg`, `time_max`, `qflx_rel_idx_min`,
`qflx_rel_idx_max`, `kx_max`, `time_idx_skip`, `figname`.

Filename kept matching the original driver's historical name (the "Dimits"
study that motivated this scan), even though the underlying computation
(stella_diagnostics.scan.zonal_flow_scan) is general-purpose. Replaces the
original's data_Dimits.json read: get_Dimits.py generated that file by
running its own copy of this computation separately and writing JSON;
plot_zonal_flow_scan now calls the same get_* functions directly and
transparently, with @cached handling the reuse instead of a hand-rolled
file. get_Dimits.py's diagnostic-page role moved to the new
plot_zonal_shear_diagnostic.py driver.
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.zonal_flow_scan import plot_zonal_flow_scan

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirnames", "labels"))

fig, axs = plot_zonal_flow_scan(
    config.dirnames,
    config.labels,
    aLT_lin_vals=getattr(config, "aLT_lin_vals", None),
    base_colors=getattr(config, "base_colors", None),
    tprim_exclude=getattr(config, "tprim_exclude", None),
    substract_lin=getattr(config, "substract_lin", False),
    xlim=getattr(config, "xlim", [3.6, 8.7]),
    markersize=getattr(config, "markersize", 10),
    filename=getattr(config, "filename", "CBC"),
    code=getattr(config, "code", "stella"),
    time_val_avg=getattr(config, "time_val_avg", None),
    time_avg=getattr(config, "time_avg", 5),
    time_max=getattr(config, "time_max", 1e10),
    qflx_rel_idx_min=getattr(config, "qflx_rel_idx_min", 1e-7),
    qflx_rel_idx_max=getattr(config, "qflx_rel_idx_max", 1e-3),
    kx_max=getattr(config, "kx_max", 0.3),
    time_idx_skip=getattr(config, "time_idx_skip", 10),
)

plt.tight_layout()

figname_add = getattr(config, "figname_add", "")
figname = getattr(config, "figname", None)
if figname is None:
    figname = "fig_param_scan_Dimits" + figname_add + ".pdf"
fig.savefig(figname)
