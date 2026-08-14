"""Quantity(kx, omega) contour comparison across a set of runs.

Usage:
    python plot_contour_quantity_vs_kx_omega.py <config.py>

<config.py> defines `dirnames` (required, list of run directories) and
optionally `labels`, `filename`, `code`, `figname_add`, `quantity`,
`kx_order`, `par_der_order`, `mult_zed`, `zed_val`, `plot_omega2_kx2`,
`time_min`/`time_max` (scalars, broadcast to every run) or
`time_min_vals`/`time_max_vals` (per-run lists), `remove_zonal`,
`only_zonal`, `logarithmic`, `vmin`, `vmax`, `kx_min`, `kx_max`,
`omega_min`, `omega_max`, `normalise_GAM`, `omega_GAM`, `eps`,
`write_text`, `overlay_secondary`, `vExP_secondary`, `basedir_secondary`.

NOTE: the original script built a `title_ax` from a large quantity/zed_val
-dependent if/elif chain, then immediately overwrote it twice more before
it was ever used (dead code, dropped here) -- see README "Known issues".
It also had a `plot_qinps` branch (hardcoded off) that referenced an
undefined `basedir` name; dropped here rather than preserved as a dormant
NameError, since it was unreachable in the original too.
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
config = load_scan_config(sys.argv[1], required=("dirnames",))

quantity = getattr(config, "quantity", "phi")
labels = getattr(config, "labels", [None] * len(config.dirnames))
filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
kx_order = getattr(config, "kx_order", 1)
par_der_order = getattr(config, "par_der_order", 0)
mult_zed = getattr(config, "mult_zed", None)
zed_val = getattr(config, "zed_val", None)
plot_omega2_kx2 = getattr(config, "plot_omega2_kx2", False)
remove_zonal = getattr(config, "remove_zonal", False)
only_zonal = getattr(config, "only_zonal", True)
logarithmic = getattr(config, "logarithmic", True)
vmin = getattr(config, "vmin", 1e-3)
vmax = getattr(config, "vmax", None)
kx_min = getattr(config, "kx_min", 0)
kx_max = getattr(config, "kx_max", 1.0)
omega_min = getattr(config, "omega_min", -0.1)
omega_max = getattr(config, "omega_max", 2)
normalise_GAM = getattr(config, "normalise_GAM", False)
omega_GAM = getattr(config, "omega_GAM", np.sqrt(7 / 4 + 1))
eps = getattr(config, "eps", 1)
write_text = getattr(config, "write_text", False)
overlay_secondary = getattr(config, "overlay_secondary", False)
vExP_secondary = getattr(config, "vExP_secondary", [0.35])
basedir_secondary = getattr(config, "basedir_secondary", None)
figname_add = getattr(config, "figname_add", "")

n_dirs = len(config.dirnames)
time_min_vals = getattr(config, "time_min_vals", [getattr(config, "time_min", 500)] * n_dirs)
time_max_vals = getattr(config, "time_max_vals", [getattr(config, "time_max", 1e5)] * n_dirs)

fig, axs = plt.subplots(ncols=n_dirs, figsize=(6.5 * n_dirs, 5), sharey=True)

for i_dir, dirname in enumerate(config.dirnames):
    time_min = time_min_vals[i_dir]
    time_max = time_max_vals[i_dir]

    run = StellaRun(dirname + "/" + filename, code=code)
    ax = axs if n_dirs == 1 else axs[i_dir]

    fig, ax, im, kx, omega, f_kx_omega = run.plot_quantity_kx_omega(
        quantity=quantity, time_min=time_min, time_max=time_max, fig=fig, ax=ax,
        remove_zonal=remove_zonal, only_zonal=only_zonal, vmin=vmin, vmax=vmax,
        logarithmic=logarithmic, kx_order=kx_order, zed_val=zed_val, mult_zed=mult_zed,
        omega_min=omega_min, omega_max=omega_max, plot_omega2_kx2=plot_omega2_kx2,
        par_der_order=par_der_order, alt_slow_eval=False, scale_eps=eps, cmap="inferno",
    )

    title_ax = labels[i_dir]
    ax.set_title(title_ax)
    fig.colorbar(im, ax=ax)

    if quantity == "phi" and write_text:
        props = dict(boxstyle="round", facecolor="white")
        fontsize_text = 20
        ax.text(0.06, 0.17, r"Toroidal secondary", c="w", rotation=49 * (1 + eps), rotation_mode="anchor", transform_rotates_text=True, fontsize=fontsize_text)
        ax.text(0.16, -0.11, r"Stationary ZF", c="w", fontsize=fontsize_text)
        ax.text(0.01, 0.82 * omega_GAM, r"GAM", c="w", fontsize=fontsize_text)

    plt.subplots_adjust(wspace=0.06)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel(r"$k_x \rho_i$")

    if overlay_secondary:
        # NOTE: pre-existing bug (predates this restructure) -- reads
        # "data_omegar.dat", but nothing anywhere in this codebase ever
        # writes a file by that name (this script's own kx/omega/
        # f_kx_omega used to be dumped as "data_kx.dat"/"data_omega.dat"/
        # "data_f_kx_omega.dat" -- since removed in favor of the cache
        # system, see get_kx_omega_spectrum). This overlay was therefore
        # always reading from some other, external process's pre-generated
        # output at dir_secondary, not from a self-produced cache -- left
        # untouched rather than routed through StellaRun/the cache system,
        # since dir_secondary isn't necessarily even a stella run
        # directory in the same sense as everything else in this file.
        for vExP in vExP_secondary:
            dir_secondary = basedir_secondary + "/tpar-0.00_tprp-3.00_phase-0.00_fprim-0.00_tprim-0.00_qinp-%.2f_vExP-%.2f_kyP-0.01/" % (2.8, vExP)
            k_sec = np.loadtxt(dir_secondary + "/data_kx.dat")
            omega_sec = np.loadtxt(dir_secondary + "/data_omegar.dat")
            ax.plot(k_sec[1:], omega_sec[1:] * omega_GAM, c="w", ls="--")

    try:
        if normalise_GAM:
            if not plot_omega2_kx2:
                ax.set_xlim([kx_min, kx_max])
                ax.set_ylim([omega_min / omega_GAM, omega_max / omega_GAM])
                ax.set_ylabel(r"$\omega/\omega_\mathrm{GAM}$" if i_dir == 0 else None)
            else:
                ax.set_xlim([0, kx_max**2])
                ax.set_ylim([0, (omega_max / omega_GAM) ** 2])
                ax.set_ylabel(r"$\omega^2/\omega^2_\mathrm{GAM}$")
        else:
            if not plot_omega2_kx2:
                ax.set_xlim([kx_min, kx_max])
                labelpad = -15 if i_dir == 0 else 0
                ax.set_ylabel(r"$\omega R/v_{Ti}$", labelpad=labelpad)
            else:
                ax.set_xlim([0, kx_max**2])
                ax.set_ylim([0, omega_max**2])
                ax.axhline(omega_GAM**2, ls="--", c="white", lw=3, alpha=0.5)
                ax.text(0.4, omega_GAM**2 * 1.03, r"$\omega_\mathrm{GAM}$", c="white", alpha=0.5)
    except Exception:
        continue

figname = "fig_contours_" + quantity + "_kx_omega"
if kx_order > 0:
    figname += "_kx-order-%i" % kx_order
if remove_zonal:
    figname += "_remove_zonal"
if only_zonal:
    figname += "_only_zonal"
if mult_zed is not None:
    figname += "_mult-zed-" + mult_zed
if zed_val is not None:
    figname += "_zed_val-%.2f" % zed_val
if overlay_secondary:
    figname += "_overlay-secondary"

plt.tight_layout()
plt.savefig(figname + figname_add + ".pdf")
