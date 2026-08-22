"""Kx/Ky-resolved nonlinear energy transfer functions (PiNZ, zonal
kx-advection, heat-flux-driven contribution) across a curated set of
runs: one figure per run (colormesh of zonal kx-advection transfer + 1D
Kx/Ky transfer plots), plus two cross-run summary figures (Kx and Ky).

Requires the PiNZ_Kx/PiNZ_Ky/PiZ_kxadv_Kx netCDF diagnostics -- not
written by every stella build (needs the corresponding compile-time
diagnostic output enabled) -- see
stella_diagnostics.physics.energy_transfer.get_energy_transfer_kx_ky.

Usage:
    python plot_energyflux_Pi_Kx_Ky_NEW.py <config.py>

<config.py> defines `dirnames`, `tmin_vals`, `tmax_vals` (each entry
either a number or "auto" -- picks the first/last time index where
PiNZ_Kx isn't NaN), `colors`, `labels` (required, all lists, one entry
per run) and optionally `filename`, `code`, `only_last`, `Kmax`,
`norm_each_Kx`, `log_scale`, `gradient`, `figname_prefix`, `figname_add`.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.spectral.stats import dt_weighted_mean

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirnames", "tmin_vals", "tmax_vals", "colors", "labels"))

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
only_last = getattr(config, "only_last", False)
Kmax = getattr(config, "Kmax", 1)
norm_each_Kx = getattr(config, "norm_each_Kx", False)
log_scale = getattr(config, "log_scale", False)
gradient = getattr(config, "gradient", True)
figname_prefix = getattr(config, "figname_prefix", "fig_Pi_energytransfer")
figname_add = getattr(config, "figname_add", "")

dirnames = list(config.dirnames)
tmin_vals = list(config.tmin_vals)
tmax_vals = list(config.tmax_vals)
colors = list(config.colors)
labels = list(config.labels)

if only_last:
    dirnames = [dirnames[-1]]
    tmin_vals = [tmin_vals[-1]]
    tmax_vals = [tmax_vals[-1]]
    labels = [labels[-1]]
    colors = [colors[-1]]

Ndirs = len(dirnames)

fig, axs_all = plt.subplots(nrows=Ndirs, ncols=3, figsize=(18, 6 * Ndirs))
fig_s, ax_s = plt.subplots(figsize=(9, 5))
fig_s_ky, ax_s_ky = plt.subplots(figsize=(9, 5))

if Ndirs == 1:
    axs_all = [axs_all]

for i_dir, dirname in enumerate(dirnames):

    axs = axs_all[i_dir]

    run = StellaRun(dirname + "/" + filename, code=code)

    ###########################
    # Time range to evaluate
    time_all = run.get_time_array()
    if tmin_vals[i_dir] == "auto" or tmax_vals[i_dir] == "auto":
        PiNZ = run.ncdata.variables['PiNZ_Kx'][:, 0, 0, 2]
        time_idx_min = np.argwhere(~np.isnan(PiNZ))[0]
        time_idx_max = np.argwhere(~np.isnan(PiNZ))[-1] + 1
        time_min = time_all[time_idx_min]
        time_max = time_all[time_idx_max - 1]
    else:
        time_min = tmin_vals[i_dir]
        time_max = tmax_vals[i_dir]

    result = run.get_energy_transfer_kx_ky(time_min=time_min, time_max=time_max)
    time_idx_vals = result["time_idx_vals"]
    Kx_vals, Ky_vals = result["Kx_vals"], result["Ky_vals"]
    PiNZ_Kx, PiNZ_Ky = result["PiNZ_Kx"], result["PiNZ_Ky"]
    PiZ_Kx_kxadv = result["PiZ_Kx_kxadv"]
    dKx_PiQ_Kx, dKy_PiQ_Ky = result["dKx_PiQ_Kx"], result["dKy_PiQ_Ky"]
    tprim = result["tprim"]

    print(np.abs(dKx_PiQ_Kx).max())

    ###########################
    # Gradients
    if gradient:
        dKx_PiNZ_Kx = np.diff(PiNZ_Kx) / np.diff(Kx_vals)
        dKx_PiZ_Kx_kxadv = np.diff(PiZ_Kx_kxadv, axis=0) / np.diff(Kx_vals)[:, None]
        dKy_PiNZ_Ky = np.diff(PiNZ_Ky) / np.diff(Ky_vals)

        Kxmid_vals = 0.5 * (Kx_vals[1:] + Kx_vals[:-1])
        Kymid_vals = 0.5 * (Ky_vals[1:] + Ky_vals[:-1])

    else:
        dKx_PiNZ_Kx = PiNZ_Kx
        dKx_PiZ_Kx_kxadv = PiZ_Kx_kxadv
        dKy_PiNZ_Ky = PiNZ_Ky

        Kxmid_vals = Kx_vals + 0.5 * Kx_vals[1]
        Kymid_vals = Ky_vals + 0.5 * Ky_vals[1]

    #### Plot

    # Colormesh of zonal advection
    ax = axs[0]
    X, Y = np.meshgrid(Kxmid_vals, Kx_vals, indexing='ij')
    Z = dKx_PiZ_Kx_kxadv
    vmax = np.abs(Z).max(); vmin = -vmax
    im = ax.pcolormesh(X, Y, Z, shading='auto', vmin=vmin, vmax=vmax, cmap='coolwarm')
    ax.plot(Kxmid_vals, Kxmid_vals, ls='--', c='k')
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(ymin=Kxmid_vals[1], ymax=Kmax)
    else:
        ax.set_xlim([0, Kmax])
        ax.set_ylim([0, Kmax])

    ax.set_xlabel(r"$K_x \rho_i$")
    ax.set_ylabel(r"$k^Z \rho_i$")
    if log_scale:
        ax.set_title(r"$\partial_{K_x} \mathcal{T}_K^Z$")
    else:
        ax.set_title(r"$\mathcal{T}_K^Z$")
    plt.colorbar(im, ax=ax)

    # 1D transfer in Kx
    ax = axs[1]
    dKx_PiZ_lowKx = np.sum(dKx_PiZ_Kx_kxadv[:, np.abs(Kx_vals) < 0.3], axis=1)
    dKx_PiZ_hiKx = np.sum(dKx_PiZ_Kx_kxadv[:, np.abs(Kx_vals) >= 0.3], axis=1)
    dKx_PiZ_Kx = np.sum(dKx_PiZ_Kx_kxadv, axis=1)
    dKx_PiZNZ_Kx = dKx_PiZ_Kx + dKx_PiNZ_Kx
    ax.plot(Kxmid_vals, dKx_PiNZ_Kx, c='forestgreen', marker='.', label=r"Nonzonal")
    ax.plot(Kxmid_vals, dKx_PiZ_lowKx, c='crimson', marker='.', label=r"Zonal $(|k^Z \rho_i| < 0.3)$", alpha=0.5)
    ax.plot(Kxmid_vals, dKx_PiZ_hiKx, c='mediumblue', marker='.', label=r"Zonal $(|k^Z \rho_i| > 0.3)$", alpha=0.5)
    ax.plot(Kxmid_vals, dKx_PiZ_Kx, c='purple', marker='.', label=r"Zonal")
    ax.plot(Kxmid_vals, dKx_PiZNZ_Kx, c='k', marker='.', label=r"Z+NZ")
    ax.set_xlabel(r"$K_x \rho_i$")
    if gradient:
        ax.set_ylabel(r"$\partial_{K_x}\mathcal{T}_{K_x}$")
        ax.plot(Kx_vals, dKx_PiQ_Kx, c='0.5', marker='.', label=r"$Q$")
    else:
        ax.set_ylabel(r"$\mathcal{T}_{K_x}$")
    ax.set_xlim(xmin=Kxmid_vals[0])
    if i_dir == 0:
        ax.legend(fontsize=20)

    ax.set_title(labels[i_dir])

    # 1D transfer in Ky
    ax = axs[2]
    ax.plot(Kymid_vals, dKy_PiNZ_Ky, c='forestgreen', marker='.', label=r"Nonzonal")
    ax.set_xlabel(r"$K_y \rho_i$")
    if gradient:
        ax.set_ylabel(r"$\partial_{K_y}\mathcal{T}_{K_y}$")
        ax.plot(Ky_vals, dKy_PiQ_Ky, c='0.5', marker='.', label=r"$Q$")
    else:
        ax.set_ylabel(r"$\mathcal{T}_{K_y}$")
    ax.set_xlim(xmin=Kymid_vals[0])

    for ax in [axs[1], axs[2]]:
        ax.set_xlim(xmax=Kmax)
        if not log_scale:
            ax.set_xlim(xmin=0)
        else:
            ax.set_xscale('log')

    # Plot summary
    delta_t = time_all[time_idx_vals[-1]] - time_all[time_idx_vals[0]]
    if norm_each_Kx:
        dKx_PiZ_lowKx = dKx_PiZ_lowKx / dKx_PiZNZ_Kx
        dKx_PiZ_hiKx = dKx_PiZ_hiKx / dKx_PiZNZ_Kx
        dKx_PiNZ_Kx = dKx_PiNZ_Kx / dKx_PiZNZ_Kx
        norm = 1
    else:
        _, _, qflx, time = run.get_fluxes_over_time()
        if time_max is None:
            time_idx_min_flux = np.argmin(np.abs(time - time_min))
            qflx_avg = qflx[time_idx_min_flux]
        else:
            mask = time > time[-1] - delta_t
            qflx_avg = dt_weighted_mean(qflx[mask], time=time[mask])

        norm = qflx_avg * tprim
        print("Q grad(T) = %.2e \n" % (norm))

    dKx_PiTot_Kx = dKx_PiNZ_Kx + dKx_PiZ_lowKx + dKx_PiZ_hiKx
    ax_s.plot(Kxmid_vals, dKx_PiTot_Kx / norm, c=colors[i_dir], label=labels[i_dir], lw=2)
    ax_s.plot(Kxmid_vals, dKx_PiNZ_Kx / norm, c=colors[i_dir], ls='--')
    ax_s.plot(Kxmid_vals, dKx_PiZ_lowKx / norm, c=colors[i_dir], ls=':')
    ax_s.plot(Kxmid_vals, dKx_PiZ_hiKx / norm, c=colors[i_dir], ls='-.')

    ax_s_ky.plot(Kymid_vals, dKy_PiNZ_Ky / norm, c=colors[i_dir], label=labels[i_dir], lw=2)

fig.tight_layout()

figname = figname_prefix
if gradient:
    figname += "_dK"
if log_scale:
    figname += "_logscale"
fig.savefig(figname + figname_add + ".pdf")

ax_s.set_xlim([0, Kmax])
if not gradient:
    ax_s.set_ylim(ymin=0, ymax=1)
if norm_each_Kx:
    ax_s.set_ylim(ymax=1)
ax_s.grid(True)
ax_s.legend(loc='upper right', handlelength=1.2, handletextpad=0.4, borderaxespad=0.4, labelspacing=0.4, borderpad=0.4)
ax_s.set_xlabel(r"$K_x \rho_i$")
if gradient:
    ylabel = r"$\partial_{K_x}\mathcal{T}_{K_x}$"
else:
    ylabel = r"$\mathcal{T}_{K_x}$"
if not norm_each_Kx:
    ylabel += r"$/ \mathcal{I}$"
ax_s.set_ylabel(ylabel)
fig_s.tight_layout()

figname_summary = figname + "_summary"
if norm_each_Kx:
    figname_summary += "_norm_each_Kx"
if only_last:
    figname_summary += "_last"
fig_s.savefig(figname_summary + figname_add + ".pdf")

# ky plot
ax_s_ky.set_xlim([0, Kmax])
if not gradient:
    ax_s_ky.set_ylim(ymin=0, ymax=1)
if norm_each_Kx:
    ax_s_ky.set_ylim(ymax=1)
ax_s_ky.grid(True)
ax_s_ky.legend(loc='upper right', handlelength=1.2, handletextpad=0.4, borderaxespad=0.4, labelspacing=0.4, borderpad=0.4)
ax_s_ky.set_xlabel(r"$K_y \rho_i$")
if gradient:
    ylabel = r"$\partial_{K_y}\mathcal{T}_{K_y}$"
else:
    ylabel = r"$\mathcal{T}_{K_y}$"
if not norm_each_Kx:
    ylabel += r"$/ \mathcal{I}$"
ax_s_ky.set_ylabel(ylabel)
fig_s_ky.tight_layout()

figname_summary_ky = figname + "_summary_Ky"
if norm_each_Kx:
    figname_summary_ky += "_norm_each_Ky"
if only_last:
    figname_summary_ky += "_last"
fig_s_ky.savefig(figname_summary_ky + figname_add + ".pdf")
