"""phi(k) spectrum comparisons (ky, kx nonzonal, kx zonal) -- one figure
group per outer `dirnames` entry (e.g. a nu scan, a qinp scan, a
convergence check, or a single run).

Usage:
    python plot_phi_spectrum_compare.py <config.py>

<config.py> defines `dirnames` (required, nested list: dirnames[i_group] =
flat list of run directories that make up one figure group) and optionally
`labels` (nested, matching `dirnames`; per-line labels within each group --
defaults to unlabeled), `figname_suffixes` (one string per group, appended
to each group's output filenames alongside the tprim read from that
group's own runs; defaults to ""), `scaling_theory_vals`,
`W_instead_of_phi`, `lw`, `overplot_kx_ky`, `plot_legend`,
`plot_alpha_spectrum`, `time_avg`, `load_from_file`, `plot_slides`,
`add_arrows`, `colors`, `filename`, `code`, `ylim_ky`, `ylim_kx_nonzonal`,
`ylim_kx_zonal`, `ylim_alpha_spectrum` (y-axis limits for the three
spectrum panels and the alpha-spectrum variant of each -- all default
None, letting matplotlib autoscale; these spectra span many orders of
magnitude and vary a lot run to run, so a fixed window tuned for one scan
can silently show a blank plot for another).

tprim/qinp (for `tprim_norm_list`/`qinp_norm_list`, the scaling-theory
normalization) are read directly from each resolved run's own netCDF
output (run.ncdata.variables["tprim"][0]/["q"].getValue(),
species-index-0 -- same convention used throughout this package), not
supplied separately.

`time_avg`: trailing-window width ending at the run's last sample --
same convention as every other quantity-time-averaging function in this
codebase (renamed from `delta_t_avg`).
"""
import sys
from os.path import exists

import matplotlib.pyplot as plt

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.run_collection import RunCollection

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirnames",))

fontsize_legend = 5
W_instead_of_phi = getattr(config, "W_instead_of_phi", False)
lw = getattr(config, "lw", 0.5)
overplot_kx_ky = getattr(config, "overplot_kx_ky", True)
plot_legend = getattr(config, "plot_legend", False)
plot_alpha_spectrum = getattr(config, "plot_alpha_spectrum", False)
time_avg = getattr(config, "time_avg", 500)
load_from_file = getattr(config, "load_from_file", True)
plot_slides = getattr(config, "plot_slides", False)
add_arrows = getattr(config, "add_arrows", False)
scaling_theory_vals = getattr(config, "scaling_theory_vals", ["unscaled"])
colors = getattr(config, "colors", ["k", "orange", "crimson", "crimson", "forestgreen", "mediumblue", "purple", "c", "pink"])
filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
ylim_ky = getattr(config, "ylim_ky", None)
ylim_kx_nonzonal = getattr(config, "ylim_kx_nonzonal", None)
ylim_kx_zonal = getattr(config, "ylim_kx_zonal", None)
ylim_alpha_spectrum = getattr(config, "ylim_alpha_spectrum", None)
figname_suffixes = getattr(config, "figname_suffixes", [""] * len(config.dirnames))
group_labels = getattr(config, "labels", [[None] * len(d) for d in config.dirnames])

figsize = (7, 4.5) if plot_slides else (4.5, 4.5)


for i_group, group_dirnames in enumerate(config.dirnames):
    add_str = ("_W_instead_of_phi" if W_instead_of_phi else "") + figname_suffixes[i_group]
    labels = group_labels[i_group]

    filenames_list, labels_list, tprim_list, qinp_list, codes_list = [], [], [], [], []
    colors_list = colors
    for i_dir, dirname in enumerate(group_dirnames):
        full_filename = dirname + "/" + filename
        if exists(full_filename + ".nc") or exists(full_filename + ".out.nc"):
            run = StellaRun(full_filename, code=code)
            filenames_list.append(full_filename)
            tprim_list.append(float(run.ncdata.variables["tprim"][0]))
            labels_list.append(labels[i_dir])
            qinp_list.append(float(run.ncdata.variables["q"].getValue()))
            codes_list.append(code)

    if not filenames_list:
        print("No runs found for figure group %d" % i_group)
        continue
    add_str += "_tprim-%.4f" % tprim_list[0]

    for scaling_theory in scaling_theory_vals:
        scanObj = RunCollection(filenames_list, labels_list, codes_list)
        scale_kmin = True

        # Phi2(ky)
        fig, ax = plt.subplots(figsize=figsize)
        fig, ax = scanObj.plot_phi_k_spectrum(fig=fig, ax=ax, plot_kx=False, color_list=colors_list, tprim_norm_list=tprim_list, qinp_norm_list=qinp_list, scale_kmin=scale_kmin, k_exp=0, scaling_theory=scaling_theory, load_from_file=load_from_file, time_avg=time_avg, plot_alpha_spectrum=plot_alpha_spectrum, lw=lw, W_instead_of_phi=W_instead_of_phi)
        ax.grid(True)
        # nu0_only (and any other mode with no per-line label, e.g. a
        # single unlabeled run) would otherwise still draw an empty
        # legend box -- only draw one if some line actually has a label.
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=fontsize_legend, ncols=1, loc="lower left", columnspacing=0.7, markerscale=1, handlelength=1.5)
        figname = "fig_phi_ky_spectrum_" + scaling_theory
        if plot_alpha_spectrum:
            figname += "_alpha-spectrum"
            if ylim_alpha_spectrum is not None:
                ax.set_ylim(ylim_alpha_spectrum)
            ax.axhline(-7 / 3, ls="--", c="0.5")
        else:
            if ylim_ky is not None:
                ax.set_ylim(ylim_ky)
        plt.tight_layout()
        plt.savefig(figname + add_str + ".pdf")
        plt.close()

        # Phi2_NZ(kx)
        fig, ax = plt.subplots(figsize=figsize)
        fig, ax = scanObj.plot_phi_k_spectrum(fig=fig, ax=ax, plot_kx=True, remove_zonal=True, color_list=colors_list, qinp_norm_list=qinp_list, tprim_norm_list=tprim_list, scale_kmin=scale_kmin, k_exp=0, scaling_theory=scaling_theory, load_from_file=load_from_file, time_avg=time_avg, plot_alpha_spectrum=plot_alpha_spectrum, lw=lw, W_instead_of_phi=W_instead_of_phi)

        figname = "fig_phi_kx_spectrum_nonzonal_" + scaling_theory
        if plot_alpha_spectrum:
            figname += "_alpha-spectrum"
            if ylim_alpha_spectrum is not None:
                ax.set_ylim(ylim_alpha_spectrum)
            ax.axhline(-7 / 3, ls="--", c="0.5")
        else:
            if ylim_kx_nonzonal is not None:
                ax.set_ylim(ylim_kx_nonzonal)

        if add_arrows:
            x_or, y_or = 7, 1e-1
            x_dest = [0.6 * 7, 0.6 * 5.6, 0.6 * 4.9]
            y_dest = [2.8e-2, 5e-2, 6e-2]
            for i in range(len(x_dest)):
                ax.plot([x_or, x_dest[i]], [y_or, y_dest[i]], c="k")

        ax.grid(True)
        # nu0_only (and any other mode with no per-line label, e.g. a
        # single unlabeled run) would otherwise still draw an empty
        # legend box -- only draw one if some line actually has a label.
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=fontsize_legend, ncols=1, loc="lower left", columnspacing=0.7, markerscale=1, handlelength=1.5)
        plt.tight_layout()
        plt.savefig(figname + add_str + ".pdf")

        if overplot_kx_ky:
            fig, ax = scanObj.plot_phi_k_spectrum(fig=fig, ax=ax, plot_kx=False, color_list=colors_list, tprim_norm_list=tprim_list, qinp_norm_list=qinp_list, scale_kmin=scale_kmin, k_exp=0, scaling_theory=scaling_theory, load_from_file=load_from_file, time_avg=time_avg, plot_alpha_spectrum=plot_alpha_spectrum, lw=2 * lw, W_instead_of_phi=W_instead_of_phi)
            ax.set_xlabel(r"$k_\perp \rho_i$")
            plt.savefig(figname + add_str + "_kxky-overplot.pdf")

        plt.close()

        # Phi2_Z(kx)
        fig, ax = plt.subplots(figsize=figsize)
        if len(labels_list) == 1:
            scanObj.list_labels = [r"$E_\mathrm{RH}$"]
            colors_list = ["crimson"]
            fontsize_legend = 24
        fig, ax = scanObj.plot_phi_k_spectrum(fig=fig, ax=ax, plot_kx=True, only_zonal=True, color_list=colors_list, qinp_norm_list=qinp_list, tprim_norm_list=tprim_list, scale_kmin=scale_kmin, k_exp=0, scaling_theory=scaling_theory, load_from_file=load_from_file, time_avg=time_avg, plot_alpha_spectrum=plot_alpha_spectrum, lw=4 * lw, W_instead_of_phi=W_instead_of_phi, plot_RH_phi_spectrum=True)
        if len(labels_list) == 1:
            scanObj.list_labels = [r"$E_\varphi$"]
            colors_list = ["mediumblue"]
        fig, ax = scanObj.plot_phi_k_spectrum(fig=fig, ax=ax, plot_kx=True, only_zonal=True, color_list=colors_list, qinp_norm_list=qinp_list, tprim_norm_list=tprim_list, scale_kmin=scale_kmin, k_exp=0, scaling_theory=scaling_theory, load_from_file=load_from_file, time_avg=time_avg, plot_alpha_spectrum=plot_alpha_spectrum, lw=4 * lw, W_instead_of_phi=W_instead_of_phi, alpha_plot=0.5)

        figname = "fig_phi_kx_spectrum_zonal_" + scaling_theory
        if plot_alpha_spectrum:
            figname += "_alpha-spectrum"
            if ylim_alpha_spectrum is not None:
                ax.set_ylim(ylim_alpha_spectrum)
            ax.axhline(-7 / 3, ls="--", c="0.5")
        elif ylim_kx_zonal is not None:
            ax.set_ylim(ylim_kx_zonal)

        ax.grid(True)
        # nu0_only (and any other mode with no per-line label, e.g. a
        # single unlabeled run) would otherwise still draw an empty
        # legend box -- only draw one if some line actually has a label.
        if plot_legend and ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=fontsize_legend, ncols=1, loc="lower left", columnspacing=0.7, markerscale=1, handlelength=1.5)
        plt.tight_layout()
        plt.savefig(figname + add_str + ".pdf")
        plt.close()
