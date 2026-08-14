"""Cross-run comparison plots of complex frequency omega versus ky/kx."""

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

from stella_diagnostics.io.codes import get_rho_label, get_vt_label


def _reference_run_ncdata(run):
    """`run` here is actually a RunCollection (see the delegate methods in
    scan.run_collection) -- pulls the first underlying StellaRun's ncdata
    for the species-label helpers, or None if the collection is empty/
    unloaded, so callers can fall back to an unsubscripted label instead
    of crashing.
    """
    try:
        return run.list_runs[0].ncdata
    except (IndexError, AttributeError):
        return None


def _ensure_omegas_loaded(run, time_avg=None, time_val_avg=None, check_convergence=True):
    """Loads run.omega_r/omega_i/kx/ky via run.load_omegas() unless
    they're already present as attributes (e.g. from an earlier call) --
    the same guard duplicated verbatim across plot_omega_ky, plot_omega_kx,
    and plot_contour_gamma_kx_ky. load_omegas always sets om_time/ky/kx/
    omega_r/omega_i together, so checking all 4 here (the 3 original
    guards each checked a different subset of them) is equivalent in
    practice to what each one checked before.
    """
    try:
        run.omega_r
        run.omega_i
        run.kx
        run.ky
    except:
        run.load_omegas(time_avg=time_avg, time_val_avg=time_val_avg, check_convergence=check_convergence)


def plot_omega_ky(run, fig=None, axs=None, label=None, ls=None, color=None, markersize=10, marker='o', gamma_min=-np.inf, time_avg=None, time_val_avg=None, kx_idx=0, check_convergence=True, rescale_vT=1, rescale_omega=1):

    _ensure_omegas_loaded(run, time_avg=time_avg, time_val_avg=time_val_avg, check_convergence=check_convergence)

    # Pick kx index with maximum gamma for each ky
    if kx_idx =="max":
        omega_r_plt = []
        omega_i_plt = []
        ky_plt      = []

        for i in range(np.shape(run.ky)[0]):
            for j in range(np.shape(run.ky)[1]):
                idx_kx_max = np.argmax(run.omega_i[i,j])
                print("kx_max = %.3f for ky = %.3f" % (run.kx[i,j,idx_kx_max], run.ky[i,j,idx_kx_max]))
                omega_r_plt.append(run.omega_r[i,j,idx_kx_max])
                omega_i_plt.append(run.omega_i[i,j,idx_kx_max])
                ky_plt.append(run.ky[i,j,idx_kx_max])
            
    else:
        omega_r_plt = (run.omega_r[:,:,kx_idx]).flatten()
        omega_i_plt = (run.omega_i[:,:,kx_idx]).flatten()
        ky_plt      = (run.ky[:,:,kx_idx]).flatten()
        print("kx val = %.3f" % (run.kx[0,0,kx_idx]))

    # Order in ky
    idx_sort_ky = np.argsort(ky_plt)
    omega_r_plt = np.array(omega_r_plt)[idx_sort_ky]
    omega_i_plt = np.array(omega_i_plt)[idx_sort_ky]
    ky_plt      = np.array(ky_plt)[idx_sort_ky]

    if axs is None:
        fig, axs = plt.subplots(nrows=2,ncols=1, figsize=(12,9))

    ncdata = _reference_run_ncdata(run)
    vt_label = get_vt_label(ncdata) if ncdata is not None else "T"
    rho_label = get_rho_label(ncdata) if ncdata is not None else r"\rho"

    axs[0].plot(ky_plt[omega_i_plt>gamma_min]/rescale_vT, omega_i_plt[omega_i_plt>gamma_min]*rescale_vT*rescale_omega, ls=ls, c=color, label=label, marker=marker, markersize=markersize, markerfacecolor='None')
    #axs[0].plot(run.ky[run.omega_i>gamma_min], run.omega_i[run.omega_i>gamma_min], ls=ls, c=color, label=label, marker=marker)
    axs[0].set_ylabel(r"$\gamma a/%s$" % vt_label)
    axs[0].set_xticklabels([])

    axs[1].plot(ky_plt[omega_i_plt>gamma_min]/rescale_vT, omega_r_plt[omega_i_plt>gamma_min]*rescale_vT*rescale_omega, ls=ls, c=color, label=label, marker=marker, markersize=markersize, markerfacecolor='None')
    #axs[1].plot(run.ky[run.omega_i>gamma_min], run.omega_r[run.omega_i>gamma_min], ls=ls, c=color, label=label, marker=marker)
    axs[1].set_ylabel(r"$\omega_r a/%s$" % vt_label)

    axs[1].set_xlabel(r"$k_y %s$" % rho_label)

    return fig, axs, omega_r_plt, omega_i_plt, ky_plt


def plot_omega_kx(run, axs=None, label=None, ls=None, color=None, marker='o', gamma_min=-np.inf, time_avg=None, time_val_avg=None, ky_idx=0):

    _ensure_omegas_loaded(run, time_avg=time_avg, time_val_avg=time_val_avg)

    # Pick ky index with maximum gamma for each kx
    if ky_idx =="max":
        omega_r_plt = []
        omega_i_plt = []
        kx_plt      = []

        for i in range(np.shape(run.kx)[0]): # Loop over directories
            for j in range(np.shape(run.kx)[1]): # Loop over kx values
                idx_ky_max = np.argmax(run.omega_i[i])
                omega_r_plt.append(run.omega_r[i,idx_ky_max,j])
                omega_i_plt.append(run.omega_i[i,idx_ky_max,j])
                kx_plt.append(run.kx[i,idx_ky_max,j])
            
    else:
        omega_r_plt = (run.omega_r[:,ky_idx,:]).flatten()
        omega_i_plt = (run.omega_i[:,ky_idx,:]).flatten()
        kx_plt      = (run.kx[:,ky_idx,:]     ).flatten()

    # Order in kx
    idx_sort_kx = np.argsort(kx_plt)
    omega_r_plt = np.array(omega_r_plt)[idx_sort_kx]
    omega_i_plt = np.array(omega_i_plt)[idx_sort_kx]
    kx_plt      = np.array(kx_plt)[idx_sort_kx]

    if axs is None:
        fig, axs = plt.subplots(nrows=2,ncols=1, figsize=(12,9))

    ncdata = _reference_run_ncdata(run)
    vt_label = get_vt_label(ncdata) if ncdata is not None else "T"
    rho_label = get_rho_label(ncdata) if ncdata is not None else r"\rho"

    axs[0].plot(kx_plt[omega_i_plt>gamma_min], omega_i_plt[omega_i_plt>gamma_min], ls=ls, c=color, label=label, marker=marker)
    axs[0].set_ylabel(r"$\gamma a/%s$" % vt_label)
    axs[0].set_xticklabels([])

    axs[1].plot(kx_plt[omega_i_plt>gamma_min], omega_r_plt[omega_i_plt>gamma_min], ls=ls, c=color, label=label, marker=marker)
    axs[1].set_ylabel(r"$\omega_r a/%s$" % vt_label)

    axs[1].set_xlabel(r"$k_x %s$" % rho_label)

    return axs


def plot_contour_gamma_kx_ky(run, ax=None, time_avg=None, time_val_avg=None):

    _ensure_omegas_loaded(run, time_avg=time_avg, time_val_avg=time_val_avg)

    omega_i_plt = run.omega_i[0,:,:]
    kx_plt      = run.kx[0,0,:]
    ky_plt      = run.ky[0,:,0]

    # Order in kx
    idx_sort_kx = np.argsort(kx_plt)
    omega_i_plt = np.array(omega_i_plt)[:,idx_sort_kx]
    kx_plt      = np.array(kx_plt)[idx_sort_kx]

    X, Y = np.meshgrid(kx_plt, ky_plt)
    Z = omega_i_plt

    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='magma', vmin=0)

    ncdata = _reference_run_ncdata(run)
    vt_label = get_vt_label(ncdata) if ncdata is not None else "T"
    rho_label = get_rho_label(ncdata) if ncdata is not None else r"\rho"

    ax.set_title(r"$\gamma a/%s$" % vt_label)
    ax.set_xlabel(r"$k_x %s$" % rho_label)
    ax.set_ylabel(r"$k_y %s$" % rho_label)

    return ax, im
