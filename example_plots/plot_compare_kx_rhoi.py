"""kx*rhoi outer-scale vs time comparison across a base_dir x tprim x
zeta_center sweep, plus the resulting scaling-vs-tprim figure.

Usage:
    python plot_compare_kx_rhoi.py <config.py>

<config.py> defines `base_dirs`, `tprim_vals`, `zeta_center_vals`
(required) and optionally `filename`, `code`, `markers`, `ls_base_dirs`,
`colors_tube`, `take_last`, `time_avg`, `time_val_avg`, `figname_time`,
`figname_scaling`.

`time_avg`/`time_val_avg`: trailing (time_val_avg=None, default) or
centered (time_val_avg=X) window WIDTH -- same convention as every other
quantity-time-averaging function in this codebase. NOTE: before a fix to
stella_diagnostics.scan.kx_rhoi_scan.get_time_avg_kx_rhoi, `time_avg` here
meant an absolute threshold (average everything after t=time_avg), not a
window width -- a real, deliberate behavior change, not a rename; see that
function's docstring.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.kx_rhoi_scan import get_time_avg_kx_rhoi

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("base_dirs", "tprim_vals", "zeta_center_vals"))

filename = getattr(config, "filename", "precise_QA_NL")
code = getattr(config, "code", "stella")
markers = getattr(config, "markers", ["o", "x"])
ls_base_dirs = getattr(config, "ls_base_dirs", ["-", "--"])
colors_tube = getattr(config, "colors_tube", ["r", "g"])
take_last = getattr(config, "take_last", False)
time_avg = getattr(config, "time_avg", 350)
time_val_avg = getattr(config, "time_val_avg", None)

colors = sns.color_palette("rocket", len(config.tprim_vals))

dirnames, labels, color_list, marker_list, ls_list, tprims_list, colors_tube_list = [], [], [], [], [], [], []
for i_base_dir, base_dir in enumerate(config.base_dirs):
    for i_tprim, tprim_val in enumerate(config.tprim_vals):
        for i_zetactr, zeta_center_val in enumerate(config.zeta_center_vals):
            dirnames.append(base_dir + "run_tprim-%.4f_zeta_center-%.4f" % (tprim_val, zeta_center_val))
            labels.append(r"$a/L_T = %.1f$" % tprim_val if i_base_dir == 0 and i_zetactr == 0 else None)
            ls_list.append(ls_base_dirs[i_base_dir])
            color_list.append(colors[i_tprim])
            marker_list.append(markers[i_base_dir])
            tprims_list.append(tprim_val)
            colors_tube_list.append(colors_tube[i_zetactr])

kx_rhoi_O_avg = np.zeros(len(dirnames))

plt.figure(figsize=(5, 3))
for i_dir, dirname in enumerate(dirnames):
    run = StellaRun(dirname + "/" + filename, code=code)
    time, kx_rhoi_O, avg = get_time_avg_kx_rhoi(run, time_avg=time_avg, take_last=take_last, time_val_avg=time_val_avg)
    kx_rhoi_O_avg[i_dir] = avg
    plt.plot(time, kx_rhoi_O, label=labels[i_dir], ls=ls_list[i_dir], c=color_list[i_dir])

plt.grid()
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\langle (k_x \rho_i)^{-1} \rangle^{-1}$")
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.tight_layout()
plt.savefig(getattr(config, "figname_time", "fig_kxrhoi_outer_time_QA.pdf"))

plt.close()
plt.figure(figsize=(5, 3))
tprim_vals = np.asarray(config.tprim_vals)
plt.loglog(tprim_vals, kx_rhoi_O_avg[0] * tprim_vals[0] / tprim_vals, ls=":", c="k", label=r"$(q \kappa)^{-1}$")
plt.loglog(tprim_vals, kx_rhoi_O_avg[0] * np.sqrt(tprim_vals[0] / tprim_vals), ls=":", c="0.5", label=r"$(q \kappa)^{-1/2}$")
for i in range(len(dirnames)):
    plt.scatter(tprims_list[i], kx_rhoi_O_avg[i], marker=marker_list[i], c=colors_tube_list[i])
plt.legend()
plt.xlabel(r"$a/L_T$")
plt.ylabel(r"$\langle (k_x \rho_i)^{-1} \rangle^{-1}$")
plt.tight_layout()
plt.savefig(getattr(config, "figname_scaling", "fig_kxrhoi_outer_QA.pdf"))
