"""Configurable single-run diagnostic page: heat flux + linear growth rate
+ zonal shear time traces, zonal shear/temperature profile, (x,y)
potential/upar contours, (x,zed) Rosenbluth-Hinton-power/momentum-transport
contours, Rosenbluth-Hinton power vs time.

Usage:
    python plot_zonal_shear_diagnostic.py <config.py>

<config.py> defines `dirname` (required) and optionally `filename`, `code`,
`time_val_avg`, `time_avg`, `time_max`, `qflx_rel_idx_min`,
`qflx_rel_idx_max`, `kx_max`, `time_idx_skip`, `exp_avg`, `avg_norm`,
`nx_padded`, `ny_padded`, `panels`, `theta_vals`, `ncols`, `figname_add`,
`figname`.

`panels`: which panels to include and in what order (default
zonal_flow_scan.DEFAULT_PANELS, a curated 8-panel subset) -- see
stella_diagnostics.scan.zonal_flow_scan.PANEL_REGISTRY for the full list
of available names (22 total, including per-field phi/apar/bpar-only
P_RH (x,zed) contour variants and the original fixed layout's other
(x,zed) contour panels, none of which are in the default set). `ncols`
(default 4) controls how many columns the grid wraps at.

`theta_vals`: one or more zed/theta values (default (0,)) for the (x,y)
contour panels ("vEx_xy", "upar_xy") -- each value produces its own
panel, e.g. panels=["vEx_xy"], theta_vals=[0, 1] gives 2 panels.

Replaces the single-run diagnostic-page half of the original
get_Dimits.py (the other half -- generating data_Dimits.json for later
comparison across runs -- is now transparent: plot_param_scan_Dimits.py
calls the underlying get_* functions directly, see
stella_diagnostics.scan.zonal_flow_scan).
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.zonal_flow_scan import plot_zonal_shear_diagnostic_page

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirname",))

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
figname_add = getattr(config, "figname_add", "")

run = StellaRun(config.dirname + "/" + filename, code=code)

fig, axs = plot_zonal_shear_diagnostic_page(
    run,
    panels=getattr(config, "panels", None),
    theta_vals=getattr(config, "theta_vals", (0,)),
    ncols=getattr(config, "ncols", 4),
    time_val_avg=getattr(config, "time_val_avg", None),
    time_avg=getattr(config, "time_avg", 5),
    time_max=getattr(config, "time_max", 1e10),
    qflx_rel_idx_min=getattr(config, "qflx_rel_idx_min", 1e-7),
    qflx_rel_idx_max=getattr(config, "qflx_rel_idx_max", 1e-3),
    kx_max=getattr(config, "kx_max", 0.3),
    time_idx_skip=getattr(config, "time_idx_skip", 10),
    exp_avg=getattr(config, "exp_avg", 2),
    avg_norm=getattr(config, "avg_norm", 2),
    nx_padded=getattr(config, "nx_padded", 256),
    ny_padded=getattr(config, "ny_padded", 256),
)

plt.tight_layout()

dirname_string = config.dirname.replace("/", "_")
figname = getattr(config, "figname", None)
if figname is None:
    figname = "fig_gammalin_gammaE_" + dirname_string + figname_add + ".pdf"
fig.savefig(figname, bbox_inches="tight")
