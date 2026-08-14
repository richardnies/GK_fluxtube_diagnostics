"""E_zonal vs q*kappa^2 scaling comparison across a tprim sweep.

Usage:
    python plot_phiZ_TS_qkappa2.py <config.py>

<config.py> defines `dirnames`, `tprim_vals` (required) and optionally
`labels`, `filenames`, `code`, `ls`, `markers`, `qinp`, `aspect_ratio`,
`kxmin`, `kxmax`, `fac_rescale`, `time_idx`, `time_avg`, `ylim` (default
None, letting matplotlib autoscale -- the previous hardcoded `ymin=4`
would silently clip any run whose E_zonal came out below 4), `figname`.
`time_idx`/`time_avg` are forwarded to
stella_diagnostics.scan.zonal_spectrum_scaling.get_Ezonal (a trailing
window of width `time_avg` ending at `time[time_idx]`, or just the single
frame at `time_idx` if `time_avg` is None -- see get_phi_k_spectrum's own
docstring in stella_diagnostics.scan.spectrum_scan).
"""
import sys
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.zonal_spectrum_scaling import get_Ezonal

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirnames", "tprim_vals"))

fontsize_legend = 16
# NOTE: the original script's per-dirname `labels` list was assigned but
# never actually used -- the plotted label is always the kappa-value
# string below, gated only by (i_dir, i_tprim) position. Preserved as-is.
filenames = getattr(config, "filenames", ["CBC"] * len(config.dirnames))
code = getattr(config, "code", "stella")
ls = getattr(config, "ls", ["--", "-", ":"])
markers = getattr(config, "markers", ["o", "x", "s"])
qinp = getattr(config, "qinp", 1.4)
aspect_ratio = getattr(config, "aspect_ratio", 1)
kxmin = getattr(config, "kxmin", 0.3)
kxmax = getattr(config, "kxmax", 1e4)
fac_rescale = getattr(config, "fac_rescale", 2 * 2.8**2)
ylim = getattr(config, "ylim", None)
time_idx = getattr(config, "time_idx", -1)
time_avg = getattr(config, "time_avg", None)

tprim_vals = np.asarray(config.tprim_vals)
colors_tprim = sns.color_palette("rocket", len(tprim_vals))

runs_list, labels_list, colors_list, ls_list, marker_list, tprim_list, qinp_list = [], [], [], [], [], [], []
for i_dir, dirname in enumerate(config.dirnames):
    for i_tprim, tprim_val in enumerate(tprim_vals):
        filename = dirname + "/run_tprim_val-%.4f/" % tprim_val + filenames[i_dir]
        if exists(filename + ".nc") or exists(filename + ".out.nc"):
            runs_list.append(StellaRun(filename, code=code))
            if i_dir == 0 and (i_tprim == 0 or i_tprim == len(tprim_vals) - 1):
                label = r"$\kappa = %.1f$" % (tprim_val * aspect_ratio)
            else:
                label = None
            labels_list.append(label)
            colors_list.append(colors_tprim[i_tprim])
            tprim_list.append(tprim_val)
            qinp_list.append(qinp)
            ls_list.append(ls[i_dir])
            marker_list.append(markers[i_dir])

fig, ax = plt.subplots(figsize=(4.5, 4.5))

# Linear fit reference line
qkappa2_plot = np.linspace(6e1, 4e3, 100)
ax.plot(qkappa2_plot, qkappa2_plot / 2.8, c="green", alpha=0.75)
ax.text(1.3e3, 1.2e3, r"$\sim q\kappa^2$", c="green", fontsize=fontsize_legend, alpha=0.75)

qkappa2_data, Ezonal_data = [], []
for i_file in range(len(runs_list)):
    Ezonal, Ezonal_stddev = get_Ezonal(runs_list[i_file], kxmin=kxmin, kxmax=kxmax, fac_rescale=fac_rescale, time_idx=time_idx, time_avg=time_avg)
    qkappa2 = qinp_list[i_file] * (tprim_list[i_file] * 2.8) ** 2

    ax.errorbar(qkappa2, Ezonal, yerr=Ezonal_stddev, marker=marker_list[i_file], color=colors_list[i_file], label=labels_list[i_file], ls="None")

    qkappa2_data.append(qkappa2)
    Ezonal_data.append(Ezonal)

ax.set_xlabel(r"$q \kappa^2$")
if kxmax > 1e3:
    ax.set_ylabel(r"$ (R/\rho_i)^2 (E^\mathrm{ZF}/T_i)_{|k_x \rho_i| > %.1f}$" % kxmin)
else:
    ax.set_ylabel(r"$(v_E^Z)^2\; (|k_x \rho_i| \in [%.1f,%.1f])$" % (kxmin, kxmax))
ax.grid()
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=fontsize_legend, handlelength=0.3, handletextpad=0.4, borderaxespad=0.4, labelspacing=0.4, borderpad=0.4)
if ylim is not None:
    ax.set_ylim(ylim)

plt.tight_layout()
fig.savefig(getattr(config, "figname", "fig_phiZ_TS_qkappa2.pdf"))
