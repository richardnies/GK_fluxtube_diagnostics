"""Collisional P_RH(kx)/(nu_ii*E_RH) vs kx, across a vnew x tprim sweep.

Usage:
    python plot_RH_P_C_kx_from_file.py <config.py>

<config.py> defines `vnew_vals`, `tprim_vals`, `vnew_labels` (dict mapping
vnew -> LaTeX label string, e.g. {0.0001: r"$10^{-4}$"}), `vnew_dirs`
(dict mapping vnew -> its directory-name suffix, e.g. {0.0001: "0.0001"}),
`basedir` (the run directory prefix, e.g.
"2026-.../run_..._vnew-"), all required, plus optionally `eps`,
`filename`, `code`, `figname`, `time_min`, `time_max`, `kx_max`,
`passing_trapped`, `fphi`, `fapar`, `fbpar`, `fcoll` (forwarded to
get_RH_per_kx_means, see that function's own defaults).

NOTE: the original script mapped vnew -> its LaTeX label (and directory
suffix) via an if/elif chain with no else/default -- if a new vnew value
were added to vnew_vals without a matching branch, it silently reused
whatever label was left over from the previous loop iteration. Config now
requires explicit `vnew_labels`/`vnew_dirs` entries per vnew value, so a
missing mapping raises a clear KeyError instead.
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.rh_collisional_kx import get_P_RH_coll_normalized_vs_kx

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("vnew_vals", "tprim_vals", "vnew_labels", "vnew_dirs", "basedir"))

eps = getattr(config, "eps", 0.18)
filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
rh_kwargs = {}
for name in ("time_min", "time_max", "kx_max", "passing_trapped", "fphi", "fapar", "fbpar", "fcoll"):
    if hasattr(config, name):
        rh_kwargs[name] = getattr(config, name)

fig, ax = plt.subplots(figsize=(9, 6))

for vnew in config.vnew_vals:
    vnew_str = config.vnew_labels[vnew]

    for tprim in config.tprim_vals:
        dirname = config.basedir + config.vnew_dirs[vnew] + "/run_tprim-%.4f" % tprim

        try:
            run = StellaRun(dirname + "/" + filename, code=code)
            kx, P_RH_coll_mean_kx_norm = get_P_RH_coll_normalized_vs_kx(run, vnew, eps=eps, **rh_kwargs)
            if kx is None:
                continue

            ax.semilogx(kx[kx > 0], -P_RH_coll_mean_kx_norm[kx > 0], marker=".", label=r"$\nu_{ii} R/v_{Ti} = $" + vnew_str + r"$, R/L_T = %.2f$" % tprim)
        except Exception as e:
            print(e)

ax.set_xlabel(r"$k_x \rho_i$")
ax.set_ylabel(r"$-P_\mathrm{RH}^C / (\epsilon^{-2} \nu_{ii} E_\mathrm{RH})$")
ax.grid(True)
ax.legend(fontsize=14)
ax.set_ylim(ymin=0)

plt.tight_layout()
fig.savefig(getattr(config, "figname", "fig_P_RH_C_normalised.pdf"))
