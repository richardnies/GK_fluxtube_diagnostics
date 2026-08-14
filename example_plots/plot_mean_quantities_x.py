"""Grid of time-averaged real-space (x) quantity comparisons across a 2D
(row, col) sweep of runs -- several display modes selected by
`quantities_plot`.

Usage:
    python plot_mean_quantities_x.py <config.py>

<config.py> defines `dirnames` (2D list: dirnames[row][col], required),
`row_titles`, `col_titles` (required), and optionally `filename`, `code`,
`quantities_plot` ("Q", "P_RH_scatter", "P_RH", "Pi_RH", or "Z_profiles"),
`scatter_vE_norm_pow`, `scatter_vE_excl_frac`, `sharex`, `time_min`,
`time_max`, `time_idx_step`, `time_avg`, `figname`. Empty-string entries in
`dirnames` are placeholders for missing row/col combinations (skipped).

`time_avg` here smooths each frame with a window CENTERED on that frame's
own time -- see movie_quantities_x.py's docstring and
get_quantities_x_tavg's docstring for why this is a deliberately different
convention from the trailing-window `time_avg` elsewhere.

Replaces the original's per-mode ``np.loadtxt(dirname+"/data_<name>.dat")``
reads (written by a prior, separately-run movie_quantities_x.py) with one
get_quantities_x_tavg() call per dirname, using the full historical set of
quantities (stella_diagnostics.scan.quantities_x_scan.FULL_QUANTITIES_X) --
so every quantities_plot mode works from a single pass per run, with no
need to have previously run movie_quantities_x.py with a particular
`quantities` block uncommented to populate the right .dat files.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.quantities_x_scan import FULL_QUANTITIES_X, get_quantities_x_tavg

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirnames", "row_titles", "col_titles"))

fontsize_labels = 32
filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
quantities_plot = getattr(config, "quantities_plot", "Q")
scatter_vE_norm_pow = getattr(config, "scatter_vE_norm_pow", 2)
scatter_vE_excl_frac = getattr(config, "scatter_vE_excl_frac", 0.3)
sharex = getattr(config, "sharex", True)
time_min = getattr(config, "time_min", 500)
time_max = getattr(config, "time_max", 2000)
time_idx_step = getattr(config, "time_idx_step", 2)
time_avg = getattr(config, "time_avg", None)

Nrows = len(config.row_titles)
Ncols = len(config.col_titles)

if quantities_plot in ["P_RH_scatter"]:
    fig_extra, ax_extra = plt.subplots(figsize=(9, 5))
    sharex = False

fig, axs = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=(10 * Ncols, 4 * Nrows), sharex=sharex)
axs = np.atleast_2d(axs)
if Ncols == 1 and Nrows > 1:
    axs = axs.reshape(Nrows, 1)

figname = getattr(config, "figname", "fig_mean_quantities_x")
label_add = ""

for i_row in range(Nrows):
    for i_col in range(Ncols):
        ax = axs[i_row, i_col]
        dirname = config.dirnames[i_row][i_col]

        ax.grid(True, alpha=0.75)
        ax.axhline(0, c="k", alpha=0.75)

        if not dirname:
            continue

        try:
            run = StellaRun(dirname + "/" + filename, code=code)
            means = get_quantities_x_tavg(run, **FULL_QUANTITIES_X, time_min=time_min, time_max=time_max, time_idx_step=time_idx_step, time_avg=time_avg)
            x = means["x"]
            vE = means["vE"]
            vE_RH = means["vE_RH"]

            if quantities_plot == "Q":
                Q = means["Q"]
                ax.plot(x, Q, c="crimson", lw=4, label=r"$Q/Q_\mathrm{gB}$")
                vE_norm = np.abs(Q).max() / np.abs(vE).max()

            if quantities_plot == "P_RH_scatter":
                P_phi_even = means["P_phi_even"]
                P_phi_odd = means["P_phi_odd"]
                P_RH_phi = means["P_RH_phi"]
                P_RH_coll = means["P_RH_coll"]
                P_RH_tot = means["P_RH"]

                if scatter_vE_norm_pow > 0:
                    vE_excl = vE_RH[np.abs(vE) > scatter_vE_excl_frac * np.abs(vE).max()]
                    P_phi_even = P_phi_even[np.abs(vE) > scatter_vE_excl_frac * np.abs(vE).max()] / np.abs(vE_excl) ** scatter_vE_norm_pow
                    P_phi_odd = P_phi_odd[np.abs(vE) > scatter_vE_excl_frac * np.abs(vE).max()] / np.abs(vE_excl) ** scatter_vE_norm_pow
                    P_RH_phi = P_RH_phi[np.abs(vE) > scatter_vE_excl_frac * np.abs(vE).max()] / np.abs(vE_excl) ** scatter_vE_norm_pow
                    P_RH_coll = P_RH_coll[np.abs(vE) > scatter_vE_excl_frac * np.abs(vE).max()] / np.abs(vE_excl) ** scatter_vE_norm_pow
                    P_RH_tot = P_RH_tot[np.abs(vE) > scatter_vE_excl_frac * np.abs(vE).max()] / np.abs(vE_excl) ** scatter_vE_norm_pow
                    vE = vE_excl
                    label_add = r"$/v_E^%i$" % (scatter_vE_norm_pow)
                else:
                    label_add = ""

                ax.scatter(vE, P_phi_even, c="crimson", alpha=0.75, label=r"$P_\mathrm{RH, \varphi}^+$" + label_add)
                ax.scatter(vE, P_phi_odd, c="mediumblue", alpha=0.75, label=r"$P_\mathrm{RH, \varphi}^-$" + label_add)
                ax.scatter(vE, P_RH_phi, c="forestgreen", alpha=0.75, label=r"$P_\mathrm{RH, \varphi}$" + label_add)
                ax.scatter(vE, P_RH_coll, c="orange", alpha=0.75, label=r"$P_\mathrm{RH}^C$" + label_add)
                ax.scatter(vE, P_RH_tot, c="k", alpha=0.75, label=r"$P_\mathrm{RH}^\mathrm{tot}$" + label_add)

                try:
                    nu = run.ncdata["vnew"][0]
                    ax_extra.scatter(vE, -P_RH_coll / nu, alpha=0.75, label=dirname)
                except Exception as e:
                    print(e)
                    print("Could not plot P/nu for " + dirname)

            if quantities_plot == "P_RH":
                P_phi_even = means["P_phi_even"]
                P_phi_odd = means["P_phi_odd"]
                P_RH_coll = means["P_RH_coll"]
                P_RH_tot = means["P_RH"]

                ax.plot(x, P_phi_even, c="crimson", lw=4, label=r"$P_\mathrm{RH, \varphi}^+$")
                ax.plot(x, P_phi_odd, c="mediumblue", lw=4, label=r"$P_\mathrm{RH, \varphi}^-$")
                ax.plot(x, P_RH_coll, c="orange", lw=4, label=r"$P_\mathrm{RH}^C$")
                ax.plot(x, P_RH_tot, c="k", lw=4, label=r"$P_\mathrm{RH}^\mathrm{tot}$")

                vE_norm = (np.abs(P_phi_even).max()) / np.abs(vE).max()

            if quantities_plot == "Pi_RH":
                Pi_RH_NL = means["Pi_RH_NL"]
                Pi_RH_even = means["Pi_RH_even"]
                Pi_RH_odd = means["Pi_RH_odd"]
                Pi_parallel = means["Pi_parallel"]

                ax.plot(x, Pi_RH_even, c="crimson", lw=4, label=r"$\Pi_\mathrm{RH}^+$")
                ax.plot(x, Pi_RH_odd, c="mediumblue", lw=4, label=r"$\Pi_\mathrm{RH}^-$")
                ax.plot(x, Pi_RH_NL, c="k", lw=4, label=r"$\Pi_\mathrm{RH}$")
                ax.plot(x, Pi_parallel, c="purple", lw=4, label=r"$\Pi_\parallel$", alpha=0.5)

                vE_norm = (np.abs(Pi_RH_even).max()) / np.abs(vE).max()

            if quantities_plot == "Z_profiles":
                upar = means["upar"]
                upar_cos = means["upar_cos"]
                dxTZ = means["gradTZ"]

                gammaE_RH = np.gradient(vE_RH, x)

                ax.plot(x, vE_RH, c="k", lw=4, label=r"$v_E^\mathrm{RH}$")
                ax.plot(x, gammaE_RH * 10, c="purple", lw=8, label=r"$10\gamma_E^\mathrm{RH}$")
                ax.plot(x, dxTZ * 5, c="crimson", lw=4, label=r"$5\partial_x T^Z$")
                ax.plot(x, upar, c="mediumblue", lw=4, label=r"$\langle u_\parallel \rangle_\psi$")
                ax.plot(x, upar_cos, c="forestgreen", lw=4, label=r"$\langle u_\parallel \cos\theta\rangle_\psi$")

                vE_norm = 1

            if quantities_plot in ["Q", "P_RH", "Z_profiles", "Pi_RH"]:
                ax.plot(x, vE * vE_norm, c="k", alpha=0.75, lw=2, label=r"$v_E^Z$ (a.u.)", ls="--")
                ax.set_xlim([x[0], x[-1]])

        except Exception as e:
            print("COULD NOT LOAD " + dirname)
            print(e)

for i_row in range(Nrows):
    axs[i_row, 0].set_ylabel(config.row_titles[i_row], fontsize=fontsize_labels)
for i_col in range(Ncols):
    axs[0, i_col].set_title(config.col_titles[i_col], fontsize=fontsize_labels)

    if quantities_plot in ["Q", "P_RH", "Z_profiles"]:
        axs[-1, i_col].set_xlabel(r"$x/\rho_i$")
    elif quantities_plot in ["P_RH_scatter"]:
        axs[-1, i_col].set_xlabel(r"$v_E^Z$")

axs[0, 0].legend(fontsize=18)

fig.tight_layout()
fig.savefig(figname + "_" + quantities_plot + ".pdf")

if quantities_plot in ["P_RH_scatter"]:
    ax_extra.set_xlabel(r"$v_E$")
    ax_extra.set_ylabel(r"$-\nu_{ii}^{-1} P_\mathrm{RH}^C$" + label_add)
    ax_extra.set_yscale("log")
    try:
        # -P_RH_coll/nu can have no positive values at all (e.g. very low
        # collisionality, or too short a time average) -- a log-scale axis
        # can't be laid out in that case. The main grid figure above has
        # already been saved regardless; only this companion figure is
        # skipped.
        fig_extra.tight_layout()
        fig_extra.savefig("fig_PRH_nu.pdf")
    except ValueError as e:
        print("Could not save fig_PRH_nu.pdf: " + str(e))
