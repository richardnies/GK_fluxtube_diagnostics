"""Parallel (field-line-following) correlation function comparison, in
k-space (ky and kx separately) across one or two scans (a tprim sweep
and/or a qinp sweep, each optionally spanning multiple codes -- e.g.
stella vs GX).

Usage:
    python plot_correlation_func.py <config.py>

<config.py> defines at least one of `tprim_scan`/`qinp_scan` (each a dict:
`dirnames`, `labels`, `filenames`, `codes`, `ls`, `markers` -- all lists,
one entry per dirname -- plus `tprim_vals`/`qinp_vals` and, for qinp_scan,
`tprim_val_per_dir`; see plot_correlation_func_perp.py's docstring/config
for the identical scan-definition shape) and optionally `quantities`,
`figname_prefix`, `figname_add`.

Each dirname's actual run directory is resolved the same way as
plot_correlation_func_perp.py: `<dirname>/run_tprim_val-<val>/<filename>`
for tprim_scan, `<dirname>/run_qinp_val-<val>/<filename>` or
`<dirname>/run_qinp-<val>/<filename>` (whichever exists) for qinp_scan --
skipped if neither `.nc` nor `.out.nc` exists for it.

`ylim_ky`/`xlim_ky`, `ylim_kx`/`xlim_kx`, `ylim_kx_unscaled` (all default
None, letting matplotlib autoscale) override each panel's axis limits --
the correlation function isn't guaranteed to stay in [0,1] (it can go
negative), so a hardcoded positive-only default would silently produce a
blank plot instead of showing what's actually there.
"""
import sys
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=())

tprim_scan = getattr(config, "tprim_scan", None)
qinp_scan = getattr(config, "qinp_scan", None)
if tprim_scan is None and qinp_scan is None:
    sys.exit("config must define at least one of tprim_scan/qinp_scan")

runs_list, labels_list, colors_list, ls_list, marker_list, tprim_list, qinp_list = [], [], [], [], [], [], []

if tprim_scan is not None:
    dirnames = tprim_scan["dirnames"]
    labels = tprim_scan["labels"]
    filenames = tprim_scan["filenames"]
    codes = tprim_scan["codes"]
    ls = tprim_scan["ls"]
    markers = tprim_scan["markers"]
    tprim_vals = np.asarray(tprim_scan["tprim_vals"])
    colors_tprim = sns.color_palette("rocket", len(tprim_vals))

    for i_dir, dirname in enumerate(dirnames):
        for i_tprim, tprim_val in enumerate(tprim_vals):
            filename = dirname + "/run_tprim_val-%.4f/" % (tprim_val) + filenames[i_dir]
            if exists(filename + ".nc") or exists(filename + ".out.nc"):
                runs_list.append(StellaRun(filename, code=codes[i_dir]))
                if i_dir == 1 and (i_tprim == 0 or i_tprim == len(tprim_vals) - 1):
                    label = r"$\kappa = %.1f$" % (tprim_val * 2.778)
                else:
                    label = None
                labels_list.append(label)
                colors_list.append(colors_tprim[i_tprim])
                tprim_list.append(tprim_val)
                qinp_list.append(1.4)
                ls_list.append(ls[i_dir])
                marker_list.append(markers[i_dir])

if qinp_scan is not None:
    dirnames = qinp_scan["dirnames"]
    filenames = qinp_scan["filenames"]
    codes = qinp_scan["codes"]
    ls = qinp_scan["ls"]
    markers = qinp_scan["markers"]
    tprim_val_per_dir = qinp_scan["tprim_val_per_dir"]
    qinp_vals = np.asarray(qinp_scan["qinp_vals"])
    colors_qinp = sns.color_palette("mako", len(qinp_vals))

    for i_dir, dirname in enumerate(dirnames):
        for i_qinp, qinp_val in enumerate(qinp_vals):
            filename1 = dirname + "/run_qinp_val-%.4f/" % (qinp_val) + filenames[i_dir]
            filename2 = dirname + "/run_qinp-%.4f/" % (qinp_val) + filenames[i_dir]

            filename = None
            if exists(filename1 + ".nc") or exists(filename1 + ".out.nc"):
                filename = filename1
            if exists(filename2 + ".nc") or exists(filename2 + ".out.nc"):
                filename = filename2
            if filename is not None:
                runs_list.append(StellaRun(filename, code=codes[i_dir]))
                if i_dir == 1 and (i_qinp == 0 or i_qinp == len(qinp_vals) - 1):
                    label = r"$q = %.1f$" % (qinp_val)
                else:
                    label = None
                labels_list.append(label)
                colors_list.append(colors_qinp[i_qinp])
                qinp_list.append(qinp_val)
                tprim_list.append(tprim_val_per_dir[i_dir])
                ls_list.append(ls[i_dir])
                marker_list.append(markers[i_dir])

quantities = getattr(config, "quantities", ["phi"])
figname_prefix = getattr(config, "figname_prefix", "fig_correlation_func_")
ylim_ky = getattr(config, "ylim_ky", None)
xlim_ky = getattr(config, "xlim_ky", None)
ylim_kx = getattr(config, "ylim_kx", None)
xlim_kx = getattr(config, "xlim_kx", None)
ylim_kx_unscaled = getattr(config, "ylim_kx_unscaled", None)
figname_add = getattr(config, "figname_add", "")

for quantity in quantities:

    plt.close()
    fig_ky, ax_ky = plt.subplots(figsize=(7.5, 4.5))
    fig_kx, ax_kx = plt.subplots(figsize=(7.5, 4.5))
    fig_kx_unscaled, ax_kx_unscaled = plt.subplots(figsize=(7.5, 4.5))

    for i_run, run in enumerate(runs_list):

        # ky
        _, _, im, avg_delta_chi, ky = run.plot_parallel_correlation_function(quantity=quantity, no_plot=True)
        ky_scaled = ky * tprim_list[i_run] * qinp_list[i_run]
        ax_ky.loglog(ky_scaled, avg_delta_chi, c=colors_list[i_run], ls=ls_list[i_run], label=labels_list[i_run], marker=marker_list[i_run])

        # kx
        _, _, im, avg_delta_chi, kx = run.plot_parallel_correlation_function(quantity=quantity, no_plot=True, kx_instead_of_ky=True)
        kx = np.abs(kx)
        kx_scaled = kx * qinp_list[i_run]

        ax_kx_unscaled.loglog(kx, avg_delta_chi, c=colors_list[i_run], ls=ls_list[i_run], label=labels_list[i_run], marker=marker_list[i_run])
        ax_kx.loglog(kx_scaled, avg_delta_chi, c=colors_list[i_run], ls=ls_list[i_run], label=labels_list[i_run], marker=marker_list[i_run])

    # Finish ky plot
    ax = ax_ky
    ax.grid()
    if ylim_ky is not None:
        ax.set_ylim(ylim_ky)
    if xlim_ky is not None:
        ax.set_xlim(xlim_ky)
    ax.set_xlabel(r"$k_y \rho_i q \kappa$")
    ax.set_ylabel(r"$(\overline{\Delta \theta})_{k_y}$", labelpad=-30)

    fig_ky.tight_layout()
    fig_ky.savefig(figname_prefix + "ky_" + quantity + figname_add + ".pdf")

    # Finish kx plot
    ax = ax_kx
    ax.grid()
    if ylim_kx is not None:
        ax.set_ylim(ylim_kx)
    if xlim_kx is not None:
        ax.set_xlim(xlim_kx)
    ax.set_xlabel(r"$k_x \rho_i q$")
    ax.set_ylabel(r"$(\overline{\Delta \theta})_{k_x}$", labelpad=-30)

    fig_kx.tight_layout()
    fig_kx.savefig(figname_prefix + "kx_" + quantity + figname_add + ".pdf")

    # Finish kx unscaled plot
    ax = ax_kx_unscaled
    ax.grid()
    if ylim_kx_unscaled is not None:
        ax.set_ylim(ylim_kx_unscaled)
    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel(r"$\overline{\Delta \theta} \sim l_\parallel / qR$", labelpad=-30)

    fig_kx_unscaled.tight_layout()
    fig_kx_unscaled.savefig(figname_prefix + "kx_" + quantity + "_unscaled" + figname_add + ".pdf")
