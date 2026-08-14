"""E_RH(tprim)/E_phi(tprim)/E_RH/E_phi/chihat/gammaE comparison across a
set of base directories (each glob-discovered for run_tprim*00 runs).

Usage:
    python plot_ERH_Ephi.py <config.py>

<config.py> defines `base_dirs`, `base_labels`, `aLT_lin_vals` (required,
one linear-threshold value per base_dir -- typically built with
stella_diagnostics.physics.gradients.get_aLT_lin_analytic) and optionally
`filename`, `code`, `time_avg`, `base_colors`, `markersize`, `figname`.

`time_avg` is the trailing-window width (time units before each run's last
sample), the same convention used across every quantity-time-averaging
function in this codebase -- setting it once in a shared per-run config
gives the same window everywhere.

tprim/qinp/eps/qflx_avg/gammaE_avg/gammaE_std come from
stella_diagnostics.scan.zonal_flow_scan (get_growth_rate_from_flux/
estimate_eps_from_bmag/get_zonal_shear_profiles, all @cached); a run that
fails any of these gets nan for this call, matching the original's own
graceful degradation.
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.rh_flux_scan import plot_ERH_Ephi_vs_tprim

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("base_dirs", "base_labels", "aLT_lin_vals"))

time_avg = getattr(config, "time_avg", 800)

fig, axs = plot_ERH_Ephi_vs_tprim(
    config.base_dirs,
    config.base_labels,
    config.aLT_lin_vals,
    filename=getattr(config, "filename", "CBC"),
    code=getattr(config, "code", "stella"),
    time_avg=time_avg,
    base_colors=getattr(config, "base_colors", None),
    markersize=getattr(config, "markersize", 10),
)

plt.tight_layout()
figname_add = getattr(config, "figname_add", "") + "_dtavg-%i" % time_avg
plt.savefig(getattr(config, "figname", None) or "fig_ERH_Ephi_Dimits" + figname_add + ".pdf")
