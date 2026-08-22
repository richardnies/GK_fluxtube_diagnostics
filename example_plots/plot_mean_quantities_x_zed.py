"""One figure per quantity: grid of time-averaged real-space (zed, x)
quantity contours across a 2D (row, col) sweep of runs.

Usage:
    python plot_mean_quantities_x_zed.py <config.py>

<config.py> defines `dirnames` (2D list: dirnames[row][col], required),
`row_titles`, `col_titles` (required), and optionally `filename`, `code`,
`sharex`, `time_min`, `time_max`, `time_idx_step`, `figname_prefix`,
`figname_add`.
Empty-string entries in `dirnames` are placeholders for missing row/col
combinations (skipped).

Replaces the original's ``np.loadtxt(dirname+"/data_zed_x_<name>.dat")``
reads (written by a prior, separately-run movie_quantities_x_zed.py) with
one get_quantities_x_zed_tavg() call per dirname, using the full
historical set of quantities
(stella_diagnostics.scan.quantities_x_scan.FULL_QUANTITIES_X_ZED). The
original's "gradPZ" panel (data_zed_x_gradnZ.dat + data_zed_x_gradTZ.dat)
depended on a "density" quantity that was never part of the historically-
active movie_quantities_x_zed.py configuration either -- dropped here for
the same reason it silently failed (COULD NOT LOAD) in the original.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.quantities_x_scan import FULL_QUANTITIES_X_ZED, get_quantities_x_zed_tavg

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirnames", "row_titles", "col_titles"))

fontsize_labels = 32
filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
sharex = getattr(config, "sharex", False)
time_min = getattr(config, "time_min", 500)
time_max = getattr(config, "time_max", 1e6)
time_idx_step = getattr(config, "time_idx_step", 2)
alpha = 0.5
lw = 2

labels = getattr(config, "labels", [r"$v_E^Z$", r"$v_E^{\mathrm{RH},Z}$", r"$Q$", r"$v_{Ex}^2$", r"$P_\mathrm{RH}$", r"$P_\varphi$", r"$P_\varphi^+$", r"$P_\varphi^-$", r"$P_{\mathrm{RH}}^C$", r"$u_\parallel^Z$", r"$\partial_x T^Z$", r"$\Pi_\parallel$"])
datanames = FULL_QUANTITIES_X_ZED["datanames"]

Nrows = len(config.row_titles)
Ncols = len(config.col_titles)
figname_prefix = getattr(config, "figname_prefix", "fig_mean_quantities_x_zed")
figname_add = getattr(config, "figname_add", "")

for i_dataname, dataname in enumerate(datanames):
    label = labels[i_dataname]

    fig, axs = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=(8 * Ncols, 5 * Nrows), sharex=sharex)
    axs = np.atleast_2d(axs)
    if Ncols == 1 and Nrows > 1:
        axs = axs.reshape(Nrows, 1)
    figname = figname_prefix + "_" + dataname

    for i_row in range(Nrows):
        for i_col in range(Ncols):
            ax = axs[i_row, i_col]
            dirname = config.dirnames[i_row][i_col]
            if not dirname:
                continue

            try:
                run = StellaRun(dirname + "/" + filename, code=code)
                dl_over_B_avg = run.dl_over_B_avg()

                ax.axvline(0, c="k", alpha=0.25)

                rot = transforms.Affine2D().rotate_deg(90)
                base = ax.transData

                means = get_quantities_x_zed_tavg(run, **FULL_QUANTITIES_X_ZED, time_min=time_min, time_max=time_max, time_idx_step=time_idx_step)
                f_zed_x = means[dataname]

                _, _, im, x, zed, f_zed_x, zed_avg_x, vmin, vmax = run.plot_quantity_x_zed(quantity=f_zed_x, fig=fig, ax=ax, vmin="symm", cmap="coolwarm")
                fig.colorbar(im, ax=ax)

                f_x = np.sum(f_zed_x * dl_over_B_avg[:, None], axis=0)
                norm = (zed[-1] / 2) / np.nanmax(np.abs(f_x))
                ax.plot(x, -f_x * norm, c="purple", alpha=alpha, transform=rot + base, label=r"$\langle f \rangle$", lw=lw)

                ax.plot(x, -zed_avg_x, c="forestgreen", alpha=alpha, transform=rot + base, label=r"$\langle f \zeta \rangle / \langle f \rangle$", lw=lw)

                vE_x = np.sum(means["vE"] * dl_over_B_avg[:, None], axis=0)
                norm = (zed[-1] / 2) / np.nanmax(np.abs(vE_x))
                ax.plot(x, -vE_x * norm, c="k", alpha=alpha, transform=rot + base, label=r"$v_E$", lw=lw)

                f_RMS_theta = np.sqrt(np.sum(f_zed_x ** 2, axis=1))
                norm = x[-1] / f_RMS_theta.max()
                ax.plot(zed, norm * f_RMS_theta + x[0], c="mediumblue", alpha=alpha, label=r"$\langle f^2 \rangle_x^{1/2}$", lw=lw)
                ax.plot(zed, 0.5 * (1 + np.cos(zed)) * x[-1] + x[0], c="mediumblue", alpha=alpha, ls="--", lw=lw)

            except Exception as e:
                print("COULD NOT LOAD " + dirname)
                print(e)

    for i_row in range(Nrows):
        axs[i_row, 0].set_ylabel(config.row_titles[i_row], fontsize=fontsize_labels)
    for i_col in range(Ncols):
        axs[0, i_col].set_title(config.col_titles[i_col], fontsize=fontsize_labels)
        axs[-1, i_col].set_xlabel(r"$\theta$")

    axs[0, 0].legend(fontsize=18)

    plt.tight_layout()
    fig.savefig(figname + figname_add + ".pdf")
    plt.close()
