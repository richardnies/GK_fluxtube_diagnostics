"""Real-space perpendicular (x and y) correlation-function comparison
across one or two scans (a tprim sweep and/or a qinp sweep, each
optionally spanning multiple codes -- e.g. stella vs GX), for several
quantities and length-scale-rescaling theories.

Usage:
    python plot_correlation_func_perp.py <config.py>

<config.py> defines at least one of `tprim_scan`/`qinp_scan` (each a dict:
`dirnames`, `labels`, `filenames`, `codes`, `ls`, `markers` -- all lists,
one entry per dirname -- plus `tprim_vals`/`qinp_vals` and, for qinp_scan,
`tprim_val_per_dir`; see scan_configs/correlation_func_perp_default.py for
a worked example) and optionally `quantities`, `scale_theories`,
`aspect_ratio`, `xmax`, `ymax`, `sum_other`, `log_abs`, `ylim_logabs`
(y-axis limits for the log_abs semilogy plots, e.g. `(1e-3, None)` --
default None lets matplotlib autoscale, since the correlation function
can be negative and a hardcoded positive lower bound would silently hide
the whole curve), `qinp_default` (the fixed qinp used for tprim_scan
points), `figname_prefix`, `figname_add`.

Each dirname's actual run directory is resolved the same way the original
script did: `<dirname>/run_tprim_val-<val>/<filename>` for tprim_scan,
`<dirname>/run_qinp_val-<val>/<filename>` or `<dirname>/run_qinp-<val>/
<filename>` (whichever exists) for qinp_scan -- skipped if neither
`.nc` nor `.out.nc` exists for it.
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

aspect_ratio = getattr(config, "aspect_ratio", 2.778)
qinp_default = getattr(config, "qinp_default", 1.4)

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
        for i_file, tprim_val in enumerate(tprim_vals):
            filename = dirname + "/run_tprim_val-%.4f/" % (tprim_val) + filenames[i_dir]
            if exists(filename + ".nc") or exists(filename + ".out.nc"):
                runs_list.append(StellaRun(filename, code=codes[i_dir]))
                if i_dir == 1 and (i_file == 0 or i_file == len(tprim_vals) - 1):
                    label = r"$(q, \kappa) = (%.1f, %.1f)$" % (qinp_default, tprim_val * aspect_ratio)
                else:
                    label = None
                labels_list.append(label)
                colors_list.append(colors_tprim[i_file])
                tprim_list.append(tprim_val)
                qinp_list.append(qinp_default)
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
                if i_dir == 1 and (i_qinp == len(qinp_vals) - 1):
                    label = r"$(q, \kappa) = (%.1f, %.1f)$" % (qinp_val, tprim_val_per_dir[i_dir] * aspect_ratio)
                else:
                    label = None
                labels_list.append(label)
                colors_list.append(colors_qinp[i_qinp])
                qinp_list.append(qinp_val)
                tprim_list.append(tprim_val_per_dir[i_dir])
                ls_list.append(ls[i_dir])
                marker_list.append(markers[i_dir])

quantities = getattr(config, "quantities", ["phi", "density", "temperature"])
scale_theories = getattr(config, "scale_theories", ["GCB", "None", "CB"])
xmax = getattr(config, "xmax", 10)
ymax = getattr(config, "ymax", 3)
sum_other = getattr(config, "sum_other", True)
log_abs = getattr(config, "log_abs", False)
ylim_logabs = getattr(config, "ylim_logabs", None)
figname_prefix = getattr(config, "figname_prefix", "fig_correlation_perp_func_")
figname_add = getattr(config, "figname_add", "")

for quantity in quantities:
    for scale_theory in scale_theories:

        plt.close()

        fig_x, ax_x = plt.subplots(figsize=(5.8, 5.5))
        fig_x.subplots_adjust(left=0.2, bottom=0.2)
        fig_y, ax_y = plt.subplots(figsize=(5.8, 5.5))
        fig_y.subplots_adjust(left=0.2, bottom=0.2)

        for i_file, run in enumerate(runs_list):

            x_corr, f_corr_x_norm, y_corr, f_corr_y_norm = run.get_perp_correlation_function(quantity, remove_zonal=True, sum_other=sum_other)

            ax = ax_x

            # Rescale according to theory
            if scale_theory == "CB":
                x_corr = x_corr / (aspect_ratio * tprim_list[i_file] * qinp_list[i_file])
                xlabel = r"$\Delta x / q \kappa \rho_i$"
            elif scale_theory == "GCB":
                x_corr = x_corr / (qinp_list[i_file])
                xlabel = r"$\Delta x /  q \rho_i  $"
            else:
                xlabel = r"$\Delta x / \rho_i$"

            if log_abs:
                ax.semilogy(x_corr, np.abs(f_corr_x_norm), label=labels_list[i_file], marker=marker_list[i_file], c=colors_list[i_file], ls=ls_list[i_file])
                if ylim_logabs is not None:
                    ax.set_ylim(ylim_logabs)
            else:
                ax.plot(x_corr, f_corr_x_norm, label=labels_list[i_file], marker=marker_list[i_file], c=colors_list[i_file], ls=ls_list[i_file])

            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\mathcal{C}(\Delta x)$")
            ax.grid(True)
            ax.legend(fontsize=20)
            ax.set_xlim(xmin=0, xmax=xmax)

            ax = ax_y

            # Rescale according to theory
            if scale_theory == "CB" or scale_theory == "GCB":
                y_corr = y_corr / (aspect_ratio * tprim_list[i_file] * qinp_list[i_file])
                xlabel = r"$\Delta y /  q\kappa\rho_i $"
            else:
                xlabel = r"$\Delta y / \rho_i$"

            if log_abs:
                ax.semilogy(y_corr, np.abs(f_corr_y_norm), label=labels_list[i_file], marker=marker_list[i_file], c=colors_list[i_file], ls=ls_list[i_file])
                if ylim_logabs is not None:
                    ax.set_ylim(ylim_logabs)
            else:
                ax.plot(y_corr, f_corr_y_norm, label=labels_list[i_file], marker=marker_list[i_file], c=colors_list[i_file], ls=ls_list[i_file])

            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\mathcal{C}(\Delta y)$")
            ax.grid(True)
            ax.set_xlim(xmin=0, xmax=ymax)

        title = figname_prefix + quantity
        title = title + "_" + scale_theory
        if log_abs:
            title += "_logabs"
        title += figname_add
        fig_x.savefig(title + "_x.pdf")
        fig_y.savefig(title + "_y.pdf")
