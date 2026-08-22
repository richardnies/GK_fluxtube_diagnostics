"""Per-kx RH phi_I/E_RH/P_RH diagnostic figures for one run, plus two
summary figures (summed over kx vs time, and time-averaged vs kx).

Usage:
    python plot_RH_phi_E_P_t_all_kx.py <config.py>

<config.py> defines `dirname` (required) and optionally `filename`, `code`,
`fphi`, `fapar`, `fbpar`, `fcoll`, `kx_max`, `time_min`, `time_max`,
`ylim_P_RH`, `linthresh`, `D_hyper`, `passing_trapped`, `combine_fields`,
`combine_even_odd`, `figname_add` (applied to the two summary figures
only, not the per-kx figures in the figs_RH_*/ subdirectories).

`fapar`/`fbpar` default to 1 (included), not 0 -- safe even for runs
without electromagnetic effects, since read_RH_fluxes falls back to an
all-zero contribution when the corresponding netCDF variables aren't
present, rather than raising. `combine_fields`/`combine_even_odd`
(both default False, i.e. show everything separately) control how the
P_RH figures break phi/apar/bpar and even/odd-parity contributions
down -- see physics.rosenbluth_hinton.plot_P_RH's docstring for the
four resulting display modes.

Individual per-kx figures (RH_phi_I/E_RH/P_RH vs time, one per kx) go into
figs_RH_{I_phi,E,P}_<dirname>/ subdirectories, matching the original. The
two summary figures are saved as fig_E_RH_P_RH_total<dirname>.pdf and
fig_E_RH_P_RH_mean_kx_<dirname>.pdf.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.rh_per_kx_scan import plot_RH_per_kx_summary_vs_kx, plot_RH_per_kx_summary_vs_time
from stella_diagnostics.spectral.stats import dt_weighted_mean

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirname",))

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
fphi = getattr(config, "fphi", 1)
fapar = getattr(config, "fapar", 1)
fbpar = getattr(config, "fbpar", 1)
fcoll = getattr(config, "fcoll", 1)
kx_max = getattr(config, "kx_max", 1e4)
time_min = getattr(config, "time_min", 500)
time_max = getattr(config, "time_max", 1e6)
ylim_P_RH = getattr(config, "ylim_P_RH", [-1e-2, 1e-2])
linthresh = getattr(config, "linthresh", 1e-4)
D_hyper = getattr(config, "D_hyper", None)
passing_trapped = getattr(config, "passing_trapped", "both")
combine_fields = getattr(config, "combine_fields", False)
combine_even_odd = getattr(config, "combine_even_odd", False)
figname_add = getattr(config, "figname_add", "")

dirname_string = config.dirname.replace("/", "_")
fig_dir_I_phi_RH = "figs_RH_I_phi_" + dirname_string
fig_dir_E_RH = "figs_RH_E_" + dirname_string
fig_dir_P_RH = "figs_RH_P_" + dirname_string
fig_add = {"passing": "_passing", "trapped": "_trapped"}.get(passing_trapped, "")
if combine_fields:
    fig_add += "_fieldsCombined"
if combine_even_odd:
    fig_add += "_evenOddCombined"
fig_dir_P_RH += fig_add

for fig_dir in (fig_dir_I_phi_RH, fig_dir_E_RH, fig_dir_P_RH):
    os.makedirs(fig_dir, exist_ok=True)

run = StellaRun(config.dirname + "/" + filename, code=code)
kx_all = run.ncdata["kx"][:]

str_fig = ""
if abs(fphi - 1) > 1e-10:
    str_fig += "_fphi-%.2f" % fphi

# Per-kx individual figures: RH_phi_I(t), E_RH(t), P_RH(t) -- each existing
# rosenbluth_hinton.py plot function called directly, unchanged.
i_fig = 0
for i_kx in range(len(kx_all)):
    if kx_all[i_kx] <= 0 or np.abs(kx_all[i_kx]) > kx_max:
        continue
    idxs_kx = np.array([i_kx])

    fig, axs, RH_phi_I, time, kx = run.plot_RH_phi_I(time_min=time_min, time_max=time_max, idxs_kx=idxs_kx)
    fig.suptitle(r"$ k_x \rho_i = %.4f$" % kx_all[i_kx])
    plt.tight_layout()
    axs[2].set_xlim(xmin=time_min)
    if time_max < 1e5:
        axs[2].set_xlim(xmax=time_max)
    axs[2].set_ylim(ymin=0)
    fig.savefig(fig_dir_I_phi_RH + "/fig_RH_phi_I_kx_t" + str_fig + "_%i.pdf" % i_fig)
    plt.close()

    fig, ax, E_RH_t_kx, t, kx = run.plot_E_RH(time_min=time_min, time_max=time_max, idxs_kx=idxs_kx)
    P_RH_t_kx_num = np.gradient(E_RH_t_kx, t, axis=0)
    ax.set_ylim(ymin=0)
    fig.suptitle(r"$ k_x \rho_i = %.4f$" % kx_all[i_kx])
    plt.tight_layout()
    fig.savefig(fig_dir_E_RH + "/fig_E_RH_kx_t" + str_fig + "_%i.pdf" % i_fig)
    plt.close()

    (fig, axs, P_RH_even_t_kx, P_RH_odd_t_kx,
     P_RH_phi_even_t_kx, P_RH_phi_odd_t_kx,
     P_RH_apar_even_t_kx, P_RH_apar_odd_t_kx,
     P_RH_bpar_even_t_kx, P_RH_bpar_odd_t_kx,
     P_RH_coll_even_t_kx, P_RH_coll_odd_t_kx,
     P_RH_hyper_t_kx, time, kx) = run.plot_P_RH(
        passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, idxs_kx=idxs_kx,
        fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll, D_hyper=D_hyper,
        combine_fields=combine_fields, combine_even_odd=combine_even_odd,
    )
    mean_P_RH_even = dt_weighted_mean(P_RH_even_t_kx[:, 0], time=time)
    mean_P_RH_odd = dt_weighted_mean(P_RH_odd_t_kx[:, 0], time=time)
    P_RH_phi_t_kx = P_RH_phi_even_t_kx + P_RH_phi_odd_t_kx
    mean_P_RH_phi = dt_weighted_mean(P_RH_phi_t_kx[:, 0], time=time)
    P_RH_apar_t_kx = P_RH_apar_even_t_kx + P_RH_apar_odd_t_kx
    mean_P_RH_apar = dt_weighted_mean(P_RH_apar_t_kx[:, 0], time=time)
    P_RH_bpar_t_kx = P_RH_bpar_even_t_kx + P_RH_bpar_odd_t_kx
    mean_P_RH_bpar = dt_weighted_mean(P_RH_bpar_t_kx[:, 0], time=time)
    P_RH_coll_t_kx = P_RH_coll_even_t_kx + P_RH_coll_odd_t_kx
    mean_P_RH_coll = dt_weighted_mean(P_RH_coll_t_kx[:, 0], time=time)
    mean_P_RH = mean_P_RH_even + mean_P_RH_odd
    if D_hyper is not None:
        mean_P_RH_hyper = dt_weighted_mean(P_RH_hyper_t_kx[:, 0], time=time)
        mean_P_RH += mean_P_RH_hyper
    mean_P_RH_num = dt_weighted_mean(P_RH_t_kx_num[:, 0], time=t)

    # axs[-1] is always the total/cross-check panel, regardless of
    # combine_even_odd (which changes how many panels plot_P_RH creates).
    ax_total = axs[-1]
    if not combine_fields:
        if fphi != 0:
            ax_total.axhline(mean_P_RH_phi, c="mediumblue", label=r"$P_{\mathrm{RH}, \varphi}$")
        if fapar != 0:
            ax_total.axhline(mean_P_RH_apar, c="crimson", label=r"$P_{\mathrm{RH}, A_\parallel}$")
        if fbpar != 0:
            ax_total.axhline(mean_P_RH_bpar, c="forestgreen", label=r"$P_{\mathrm{RH}, B_\parallel}$")
    if fcoll != 0:
        ax_total.axhline(mean_P_RH_coll, c="orange", label=r"$P_{\mathrm{RH}, C}$")
    ax_total.axhline(mean_P_RH, c="k", label=r"Total")
    ax_total.axhline(mean_P_RH_num, c="0.5", ls="--", label=r"Numerical")
    if ylim_P_RH is not None:
        ax_total.set_ylim(ylim_P_RH)
    ax_total.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    ax_total.set_yscale("symlog", linthresh=linthresh)
    fig.suptitle(r"$ k_x \rho_i = %.4f$" % kx_all[i_kx])
    plt.tight_layout()
    fig.savefig(fig_dir_P_RH + "/fig_P_RH_kx_t" + str_fig + "_%i.pdf" % i_fig, bbox_inches="tight")
    plt.close()

    i_fig += 1

means_kwargs = dict(
    time_min=time_min, time_max=time_max, kx_max=kx_max, passing_trapped=passing_trapped,
    fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll, D_hyper=D_hyper,
    combine_fields=combine_fields, combine_even_odd=combine_even_odd,
)

fig, axs = plot_RH_per_kx_summary_vs_time(run, ylim_P_RH=ylim_P_RH, linthresh=linthresh, **means_kwargs)
plt.tight_layout()
fig.savefig("fig_E_RH_P_RH_total" + dirname_string + "_" + str_fig + fig_add + figname_add + ".pdf", bbox_inches="tight")
plt.close()

fig, axs = plot_RH_per_kx_summary_vs_kx(run, **means_kwargs)
plt.tight_layout()
fig.savefig("fig_E_RH_P_RH_mean_kx_" + dirname_string + "_" + str_fig + fig_add + figname_add + ".pdf", bbox_inches="tight")
plt.close()
