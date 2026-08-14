"""RunCollection: aggregates multiple StellaRun objects for cross-run scan comparisons (omega vs ky/kx, flux-tube geometry, phi vs zed)."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.special as specialfunc
from scipy.interpolate import interp1d as interp
from scipy.interpolate import RegularGridInterpolator as interp2D
from scipy import integrate
from scipy.interpolate import interpn
from scipy.signal import argrelextrema
import seaborn as sns
from glob import glob
from os.path import exists

from stella_diagnostics.io.run import StellaRun
import stella_diagnostics.scan.omega_scan as scan_omega_scan
import stella_diagnostics.scan.spectrum_scan as scan_spectrum_scan


def load_omegas(run, timestep=-1, om_avg=True, time_avg=None, time_val_avg=None, check_convergence=True):

    run.om_time = []
    run.ky      = []
    run.kx      = []
    run.omega_r = []
    run.omega_i = []

    for i, single_run in enumerate(run.list_runs):
        try:
            om_time, ky, kx, omega_r, omega_i = \
                     single_run.read_data_omega_k(timestep=timestep, om_avg=om_avg, time_avg=time_avg, time_val_avg=time_val_avg, check_convergence=check_convergence)
        except:    
            om_time = ky = kx = omega_r = omega_i = [[0]]

        run.om_time.append(om_time)
        run.ky.append(ky)
        run.kx.append(kx)
        run.omega_r.append(omega_r)
        run.omega_i.append(omega_i)

    run.om_time = np.array(run.om_time)
    run.ky      = np.array(run.ky)
    run.kx      = np.array(run.kx)
    run.omega_r = np.array(run.omega_r)
    run.omega_i = np.array(run.omega_i)

    return run.om_time, run.ky, run.kx, run.omega_r, run.omega_i


def load_phi_vs_zed(run):

    list_phi_vs_t = []
    list_zed      = []
    for i, single_run in enumerate(run.list_runs):

        phi_vs_t, zed = single_run.read_phi_vs_zed()
        list_phi_vs_t.append(phi_vs_t)
        list_zed.append(zed)

    return list_phi_vs_t, list_zed


def plot_comparison_flux_tube_geometry(run, plot_phi=False, zed_times_nfield_periods=False, load_from_nc=True, normalise_bmag=False, colors=None, fig=None, axs=None, norm_gradpar=False):

    if fig is None and axs is None:
        fig, axs = plt.subplots(nrows=3,ncols=4, figsize=(24,10))
        #fig, axs = plt.subplots(nrows=3,ncols=4, figsize=(24,18))
        plt.subplots_adjust(hspace=0,left=0.08,right=0.95,top=0.9,bottom=0.1,wspace=0.45)

    for i, single_run in enumerate(run.list_runs):
        if colors is None:
            color = None
        else:
            color=colors[i]
        single_run.plot_flux_tube_geometry(axs=axs, label=run.list_labels[i], plot_phi=plot_phi, zed_times_nfield_periods=zed_times_nfield_periods, load_from_nc=load_from_nc, normalise_bmag=normalise_bmag, color=color, ls=run.list_ls[i], norm_gradpar=norm_gradpar)

    return fig, axs


def plot_phi_vs_zed(run, ax=None, zed_times_nfield_periods=False):

    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    for i, single_run in enumerate(run.list_runs):
        single_run.plot_phi_vs_zed(ax=ax, label=run.list_labels[i], zed_times_nfield_periods=zed_times_nfield_periods)

    return ax


class RunCollection:

    def __init__(self, filenames_base, labels=None, codes=None, ls=None):

        self.filenames_base = filenames_base
        self.Nr_files       = len(filenames_base)

        # Initialise StellaDiagnostics objects
        self.list_runs = []
        self.list_labels  = []
        self.list_ls  = []
        for i_file, filename_base in enumerate(self.filenames_base):
            if codes is None:
                code = "stella"
            else:
                code = codes[i_file]

            if ls is None:
                self.list_ls.append("-")
            else:
                self.list_ls.append(ls[i_file])

            try:
                single_run = StellaRun(filename_base, code=code)
                self.list_runs.append( single_run )
                if labels is None:
                    self.list_labels.append(None)
            except:
                print("Couldn't load " + filename_base)
                if labels is not None:
                    del labels[i_file]
                continue

        if labels is not None:
            self.list_labels = labels

        assert(len(self.list_runs) == len(self.list_labels))

    def load_omegas(self, timestep=-1, om_avg=True, time_avg=None, time_val_avg=None, check_convergence=True):
        return load_omegas(self, timestep=timestep, om_avg=om_avg, time_avg=time_avg, time_val_avg=time_val_avg, check_convergence=check_convergence)

    def load_phi_vs_zed(self):
        return load_phi_vs_zed(self)

    def plot_comparison_flux_tube_geometry(self, plot_phi=False, zed_times_nfield_periods=False, load_from_nc=True, normalise_bmag=False, colors=None, fig=None, axs=None, norm_gradpar=False):
        return plot_comparison_flux_tube_geometry(self, plot_phi=plot_phi, zed_times_nfield_periods=zed_times_nfield_periods, load_from_nc=load_from_nc, normalise_bmag=normalise_bmag, colors=colors, fig=fig, axs=axs, norm_gradpar=norm_gradpar)

    def plot_omega_ky(self, fig=None, axs=None, label=None, ls=None, color=None, markersize=10, marker='o', gamma_min=-np.inf, time_avg=None, time_val_avg=None, kx_idx=0, check_convergence=True, rescale_vT=1, rescale_omega=1):
        return scan_omega_scan.plot_omega_ky(self, fig=fig, axs=axs, label=label, ls=ls, color=color, markersize=markersize, marker=marker, gamma_min=gamma_min, time_avg=time_avg, time_val_avg=time_val_avg, kx_idx=kx_idx, check_convergence=check_convergence, rescale_vT=rescale_vT, rescale_omega=rescale_omega)

    def plot_omega_kx(self, axs=None, label=None, ls=None, color=None, marker='o', gamma_min=-np.inf, time_avg=None, time_val_avg=None, ky_idx=0):
        return scan_omega_scan.plot_omega_kx(self, axs=axs, label=label, ls=ls, color=color, marker=marker, gamma_min=gamma_min, time_avg=time_avg, time_val_avg=time_val_avg, ky_idx=ky_idx)

    def plot_contour_gamma_kx_ky(self, ax=None, time_avg=None, time_val_avg=None):
        return scan_omega_scan.plot_contour_gamma_kx_ky(self, ax=ax, time_avg=time_avg, time_val_avg=time_val_avg)

    def plot_phi_vs_zed(self, ax=None, zed_times_nfield_periods=False):
        return plot_phi_vs_zed(self, ax=ax, zed_times_nfield_periods=zed_times_nfield_periods)

    def plot_contour_phi_vs_zed_theta0(self, fig=None, ax=None, normalise_phi=False, logarithmic=False, vmin=None, vmax=None):
        return scan_spectrum_scan.plot_contour_phi_vs_zed_theta0(self, fig=fig, ax=ax, normalise_phi=normalise_phi, logarithmic=logarithmic, vmin=vmin, vmax=vmax)

    def plot_phi_k_spectrum(self, plot_kx, fig=None, ax=None, time_idx=-1, ls_list=None, marker_list=None, color_list=None, tprim_norm_list=None, qinp_norm_list=None, xdrift_norm_list=None, time_avg=None, only_zonal=False, remove_zonal=False, scale_kmin=False, k_exp=0, alpha_kx_O=1, beta_kx_O=0, lw=None, no_label=False, scaling_theory='GCB', W_instead_of_phi=False, scale_fac_vals=None, zonal_stationary=False, load_from_file=False, mult_k=False, plot_alpha_spectrum=False, plot_RH_phi_spectrum=False, alpha_plot=1, markersize=3):
        return scan_spectrum_scan.plot_phi_k_spectrum(self, plot_kx=plot_kx, fig=fig, ax=ax, time_idx=time_idx, ls_list=ls_list, marker_list=marker_list, color_list=color_list, tprim_norm_list=tprim_norm_list, qinp_norm_list=qinp_norm_list, xdrift_norm_list=xdrift_norm_list, time_avg=time_avg, only_zonal=only_zonal, remove_zonal=remove_zonal, scale_kmin=scale_kmin, k_exp=k_exp, alpha_kx_O=alpha_kx_O, beta_kx_O=beta_kx_O, lw=lw, no_label=no_label, scaling_theory=scaling_theory, W_instead_of_phi=W_instead_of_phi, scale_fac_vals=scale_fac_vals, zonal_stationary=zonal_stationary, load_from_file=load_from_file, mult_k=mult_k, plot_alpha_spectrum=plot_alpha_spectrum, plot_RH_phi_spectrum=plot_RH_phi_spectrum, alpha_plot=alpha_plot, markersize=markersize)

    def plot_Q_k_spectrum(self, plot_kx, species_idx=0, tube=0, fig=None, ax=None, time_idx=-1, ls_list=None, marker_list=None, color_list=None, delta_t_avg=None, zed_val=None, scale_k=False, scale_kmin=True, kfilter_vals=None, plot_k_qk=False):
        return scan_spectrum_scan.plot_Q_k_spectrum(self, plot_kx=plot_kx, species_idx=species_idx, tube=tube, fig=fig, ax=ax, time_idx=time_idx, ls_list=ls_list, marker_list=marker_list, color_list=color_list, delta_t_avg=delta_t_avg, zed_val=zed_val, scale_k=scale_k, scale_kmin=scale_kmin, kfilter_vals=kfilter_vals, plot_k_qk=plot_k_qk)

