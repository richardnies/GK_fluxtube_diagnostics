"""Qflx(nu)/gammaE(nu)/vE_RH(nu)/upar(nu) comparison across a
collisionality x tprim sweep.

Usage:
    python plot_flux_coll.py <config.py>

<config.py> defines `dirs_nu`, `vals_nu`, `tprim_vals` (required) and
optionally `filename`, `code`, `time_avg`, `rhoc`, `q`, `shat`, `figname`.

`time_avg` is the trailing-window width (time units before each run's last
sample), the same convention used across every quantity-time-averaging
function in this codebase (stella_diagnostics.scan.zonal_flow_scan,
stella_diagnostics.scan.rh_flux_scan, stella_diagnostics.physics.
velocity_space.plot_contour_gvmu_vpa) -- setting it once in a shared
per-run config gives the same window everywhere.

vE_RH_avg/vE_avg/gammaE_avg/upar_avg come from
stella_diagnostics.scan.zonal_flow_scan.get_zonal_shear_profiles (@cached,
computed transparently on first use -- no separate script needs to have
been run first).
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.rh_flux_scan import plot_qflx_vs_nu_scan

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirs_nu", "vals_nu", "tprim_vals"))

time_avg = getattr(config, "time_avg", 200)

fig, axs = plot_qflx_vs_nu_scan(
    config.dirs_nu,
    config.vals_nu,
    config.tprim_vals,
    filename=getattr(config, "filename", "CBC"),
    code=getattr(config, "code", "stella"),
    time_avg=time_avg,
    rhoc=getattr(config, "rhoc", 0.18),
    q=getattr(config, "q", 1.4),
    shat=getattr(config, "shat", 0.8),
)

plt.tight_layout()
plt.savefig(getattr(config, "figname", None) or "fig_Q_nu_dtavg-%i.pdf" % time_avg)
plt.close()
