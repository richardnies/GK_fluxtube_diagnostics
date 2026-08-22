"""Zonal-energy contribution spectrum dE_Z/dt(kx) -- Reynolds stress
(phi/P_perp), magnetic-drift, and streaming contributions -- for one run,
plus (optionally) E_Z(kx) split by |omega| threshold.

Usage:
    python plot_dEZ_kx.py <config.py>

<config.py> defines `dirname` (required) and optionally `filename`, `code`,
`time_min`, `time_max`, `time_idx_skip`, `kxmax_plot`, `omega_sep_vals`,
`alt_slow_eval`, `figname_add`, `figname`.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirname",))

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
time_min = getattr(config, "time_min", 1000)
time_max = getattr(config, "time_max", 1200)
time_idx_skip = getattr(config, "time_idx_skip", 1)
kxmax_plot = getattr(config, "kxmax_plot", 1)
omega_sep_vals = getattr(config, "omega_sep_vals", None)
alt_slow_eval = getattr(config, "alt_slow_eval", True)
figname_add = getattr(config, "figname_add", "")

run = StellaRun(config.dirname + "/" + filename, code=code)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

(
    kx, EZ_kx, dEZ_reynolds_phi_nablax2_kx, dEZ_reynolds_Pprp_nablax2_kx,
    dEZ_reynolds_phi_nablaxy_kx, dEZ_reynolds_Pprp_nablaxy_kx, dEZ_vDx_P_kx,
    dEZ_upar_kx, dE_mean_pressure_tr_kx, dE_delt_pressure_tr_kx,
    dE_par_mom_tr_kx, du_par_mom_tr_kx, du_cos_par_mom_tr_kx,
) = run.get_time_avg_zonal_energy_contributions_kx(time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, alt_slow_eval=alt_slow_eval)

dEZ_reynolds_phi_kx = dEZ_reynolds_phi_nablax2_kx + dEZ_reynolds_phi_nablaxy_kx
dEZ_reynolds_Pprp_kx = dEZ_reynolds_Pprp_nablax2_kx + dEZ_reynolds_Pprp_nablaxy_kx
dEZ_reynolds_kx = dEZ_reynolds_phi_kx + dEZ_reynolds_Pprp_kx
dEZ_tot_kx = dEZ_reynolds_phi_kx + dEZ_reynolds_Pprp_kx + dEZ_vDx_P_kx + dEZ_upar_kx

print(np.abs(dEZ_reynolds_kx).max())
print(np.abs(dEZ_vDx_P_kx).max())
print(np.abs(dEZ_upar_kx).max())
print(np.sum(dEZ_reynolds_kx))
print(np.sum(dEZ_vDx_P_kx))
print(np.sum(dEZ_upar_kx))

ax = axs[0]
ax.plot(kx, dEZ_reynolds_phi_kx, c="c", label=r"Reynolds $\varphi$", marker=".")
ax.plot(kx, dEZ_reynolds_Pprp_kx, c="brown", label=r"Reynolds $P_\perp$", marker=".")
ax.plot(kx, dEZ_vDx_P_kx, c="b", label="Mag. drift", marker=".")
ax.plot(kx, dEZ_upar_kx, c="r", label="Streaming", marker=".")
ax.plot(kx, dEZ_tot_kx, c="m", label="Total", marker=".", ls=":")

ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$\langle \mathrm{d}E^Z_\varphi/\mathrm{d}t \rangle_T$")
ax.grid()
ax.legend()
if kxmax_plot is not None:
    ax.set_xlim([0, kxmax_plot])
else:
    ax.set_xlim(xmin=0)

# Plot EZ(kx)
ax = axs[1]

if omega_sep_vals is not None:
    _, _, _, kx, omega, EZ_kx_omega = run.plot_quantity_kx_omega(quantity="E_Z", time_min=time_min, time_max=time_max, mult_zed="nablax-nablax", no_plot=True)
    # Factor 2 for negative kx's
    EZ_kx_omega = 2 * np.abs(EZ_kx_omega[kx >= 0])
    kx = kx[kx >= 0]

    ax.plot(kx, np.sum(EZ_kx_omega[:, np.abs(omega) <= omega_sep_vals[0]], axis=1), marker=".", label=r"$|\omega| < %.4f$" % (omega_sep_vals[0]))
    for omega_sep_val in omega_sep_vals:
        ax.plot(kx, np.sum(EZ_kx_omega[:, np.abs(omega) >= omega_sep_val], axis=1), marker=".", label=r"$|\omega| > %.4f$" % (omega_sep_val))
    ax.legend()

else:
    ax.plot(kx, EZ_kx, marker=".")

ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$\langle E_Z \rangle_T$")
ax.grid()
ax.set_ylim(ymin=0)
if kxmax_plot is not None:
    ax.set_xlim([0, kxmax_plot])
else:
    ax.set_xlim(xmin=0)

plt.tight_layout()

dirname_string = config.dirname.replace("/", "_")
fig.savefig(getattr(config, "figname", None) or "fig_dEZ_kx_" + dirname_string + figname_add + ".pdf")
