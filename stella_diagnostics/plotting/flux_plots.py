"""Plots of fluxes and net radial drift versus time."""

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
from stella_diagnostics.plotting.mpl_helpers import get_or_create_ax
from stella_diagnostics.io.codes import get_nspecies, get_species_label, get_species_name, get_vt_label
from stella_diagnostics.spectral.stats import dt_weighted_mean


def evaluate_net_radial_drift(run, B_bounce=0.9):
    cvdrift0 = run.ncdata.variables['cvdrift0'][:,0] # drift * grad(x) * shat
    bmag     = run.ncdata.variables['bmag'][:,0]
    dl_over_B_avg = run.dl_over_B_avg()
    zed     = run.ncdata.variables['zed'][:]

    bmag_norm = (bmag - bmag.min())/(bmag.max() - bmag.min())

    func_integrand = dl_over_B_avg*cvdrift0

    bmag_norm_signed = bmag_norm*np.sign(zed)
    idx_bnc_pls  = np.argmin( np.abs(bmag_norm_signed - B_bounce) )
    idx_bnc_min  = np.argmin( np.abs(bmag_norm_signed + B_bounce) )

    dB_dzed_bnc_pls = (bmag_norm[idx_bnc_pls]-bmag_norm[idx_bnc_pls-1])/(zed[idx_bnc_pls]-zed[idx_bnc_pls-1])
    dB_dzed_bnc_min = (bmag_norm[idx_bnc_min]-bmag_norm[idx_bnc_min+1])/(zed[idx_bnc_min]-zed[idx_bnc_min+1])


    endpoint_contr_pls = 2*func_integrand[idx_bnc_pls]*np.sqrt(np.abs( (zed[idx_bnc_pls]-zed[idx_bnc_min]) *B_bounce/dB_dzed_bnc_pls) )
    endpoint_contr_min = 2*func_integrand[idx_bnc_min]*np.sqrt(np.abs( (zed[idx_bnc_pls]-zed[idx_bnc_min]) *B_bounce/dB_dzed_bnc_min) )

    net_radial_drift      =        endpoint_contr_pls         + endpoint_contr_min
    net_radial_drift_norm = np.abs(endpoint_contr_pls) + np.abs(endpoint_contr_min)

    dzed = zed[1]-zed[0]
    for i_zed in range(idx_bnc_min+1, idx_bnc_pls):
        integrand =  func_integrand[i_zed]      /np.sqrt(1-bmag_norm[i_zed]/B_bounce) \
                  - func_integrand[idx_bnc_pls]/np.sqrt(np.abs(dB_dzed_bnc_pls/B_bounce*(zed[i_zed]-zed[idx_bnc_pls]))) \
                  - func_integrand[idx_bnc_min]/np.sqrt(np.abs(dB_dzed_bnc_min/B_bounce*(zed[i_zed]-zed[idx_bnc_min])))
        net_radial_drift      +=        integrand  *dzed
        net_radial_drift_norm += np.abs(integrand) *dzed

    return net_radial_drift/net_radial_drift_norm


def plot_net_radial_drift(run, fig=None, ax=None, label=None, ls=None, color=None):
    zed     = run.ncdata.variables['zed'][:]
    zed_pos = zed[zed>=0]

    fig, ax = get_or_create_ax(fig, ax, nrows=1, ncols=1, figsize=(12,9))

    # evaluate_net_radial_drift is parameterized by a normalized-|B|
    # bounce threshold (B_bounce, in [0,1]), not a zed coordinate --
    # convert each zed_pos grid point to its normalized |B| value first
    # (same normalization evaluate_net_radial_drift computes internally
    # from the full zed grid). Previously this called
    # evaluate_net_radial_drift(zed_b=zed_b), a kwarg the function has
    # never accepted (only B_bounce), which always raised TypeError.
    bmag = run.ncdata.variables['bmag'][:,0]
    bmag_norm = (bmag - bmag.min()) / (bmag.max() - bmag.min())
    bmag_norm_at_zed_pos = np.interp(zed_pos, zed, bmag_norm)

    # Evaluate net drift
    net_radial_drift = np.zeros(len(zed_pos))
    for i_b, B_bounce in enumerate(bmag_norm_at_zed_pos):
        net_radial_drift[i_b] = run.evaluate_net_radial_drift(B_bounce=B_bounce)

    # Plot
    ax.plot(zed_pos, net_radial_drift, ls=ls, label=label, color=color)

    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\Delta \psi$ (a.u.)")
    ax.set_xlim([0,np.pi])

    return fig, ax


def plot_flux_over_time(run, axs=None, label=None, species_idx=None, ls='-', color=None, marker=None, timeavg=None, timemax=np.inf, log=False):
    """Plots particle/momentum/heat flux (each normalized to its
    gyroBohm value, per stella's own output convention -- see
    physics.fluxes.flux_norm) vs. time, one line per species.

    species_idx=None (default) plots every species in the run; pass an
    int to restrict to just one (the previous, single-species-only
    default was species_idx=0). Each line's legend label is that
    species' readable name ("Ions"/"Electrons", from the sign of its
    charge -- see io.codes.get_species_name; the y-axis labels already
    say which quantity/normalization, so the legend only needs to
    distinguish species). The time axis is labeled with the subscript
    of the reference species (the first ion species if any is present,
    else the first electron species -- see io.codes.get_vt_label).

    NOTE: does NOT separate electrostatic vs. electromagnetic
    contributions -- checked directly against the STELLA Fortran source
    (diagnostics_fluxes_fluxtube.f90's calculate_fluxes_fluxtube): both
    contributions are computed and then summed in place
    (`flux_out = flux_out + ...`) into the same pflux_vs_s/vflux_vs_s/
    qflux_vs_s before anything is written out, so that split isn't
    recoverable from stella's saved output at all, regardless of how
    this function reads it.

    timeavg: if given, returns {species_idx: avg} dicts instead of
    scalars, since there can now be more than one species.
    """
    if axs is None:
        fig, axs = plt.subplots(3,1,figsize=(12,9))
        plt.subplots_adjust(hspace=0,left=0.08,right=0.95,top=0.9,bottom=0.1,wspace=0.45)

    species_idxs = [species_idx] if species_idx is not None else list(range(get_nspecies(run.ncdata)))
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    pflx_avg_vs_s, vflx_avg_vs_s, qflx_avg_vs_s = {}, {}, {}

    for i_loop, sp_idx in enumerate(species_idxs):
        pflx, vflx, qflx, time = run.get_fluxes_over_time(species_idx=sp_idx)

        if log:
            pflx = np.abs(pflx)
            vflx = np.abs(vflx)
            qflx = np.abs(qflx)

        vflx = np.nan_to_num(vflx, 1)

        sp_color = color if color is not None else default_colors[i_loop % len(default_colors)]
        sp_label = get_species_label(run.ncdata, sp_idx)
        sp_name = get_species_name(run.ncdata, sp_idx)

        if timeavg is not None:
            timemax  = min(timemax, time[-1])
            mask = (time > timemax-timeavg) & (time <= timemax)
            pflx_avg = dt_weighted_mean(pflx[mask], time=time[mask])
            vflx_avg = dt_weighted_mean(vflx[mask], time=time[mask])
            qflx_avg = dt_weighted_mean(qflx[mask], time=time[mask])
            pflx_avg_vs_s[sp_idx] = pflx_avg
            vflx_avg_vs_s[sp_idx] = vflx_avg
            qflx_avg_vs_s[sp_idx] = qflx_avg

            print(run.filename_base + ": qflx_avg (species %s) = %e" % (sp_label, qflx_avg))

            xmin_plot = max(timemax-timeavg, 0)
            xmax_plot = timemax
            axs[0].plot([xmin_plot, xmax_plot], [pflx_avg, pflx_avg], ls=ls, marker=marker, c='0.5', lw=2)
            axs[1].plot([xmin_plot, xmax_plot], [vflx_avg, vflx_avg], ls=ls, marker=marker, c='0.5', lw=2)
            axs[2].plot([xmin_plot, xmax_plot], [qflx_avg, qflx_avg], ls=ls, marker=marker, c='0.5', lw=2)

        prefix = (label + " ") if label else ""
        axs[0].plot(time, pflx, label=prefix + sp_name, ls=ls, marker=marker, c=sp_color)
        axs[1].plot(time, vflx, label=prefix + sp_name, ls=ls, marker=marker, c=sp_color)
        axs[2].plot(time, qflx, label=prefix + sp_name, ls=ls, marker=marker, c=sp_color)

    if log:
        for ax in axs:
            ax.set_yscale('log')

    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])

    axs[0].set_ylabel(r"$\Gamma/\Gamma_{\mathrm{gB}}$")
    axs[1].set_ylabel(r"$\Pi/\Pi_{\mathrm{gB}}$")
    axs[2].set_ylabel(r"$Q/Q_{\mathrm{gB}}$")
    axs[2].set_xlabel(r"$t %s/a$" % get_vt_label(run.ncdata))

    axs[0].legend()

    axs[0].set_xlim(xmin=0)
    axs[1].set_xlim(xmin=0)
    axs[2].set_xlim(xmin=0)

    axs[0].grid()
    axs[1].grid()
    axs[2].grid()

    if timeavg is not None:
        return axs, pflx_avg_vs_s, vflx_avg_vs_s, qflx_avg_vs_s
    else:
        return axs


def plot_flux_spectra(run, fig=None, ax=None, species_idx=0, tube=0, time_idx=-1, kx_idx=0):
    # NOTE: pre-existing (predates the restructure, confirmed against real
    # stella runs) -- read_flux_spectra() below looks up the netCDF
    # variable 'qflx_kxky', which some stella versions instead write as
    # 'qflux_vs_kxkys'; on those runs this raises KeyError. Same for
    # plot_flux_spectra_kx_ky below. See README "Known issues".
    qflx_t_zed_kx_ky, time, zed, kx, ky = run.read_flux_spectra(species_idx, tube)

    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    print("Note len(ky) = %i, len(zed) = %i, len(kx) = %i, len(t) = %i." % (len(ky), len(zed), len(kx), len(time)))

    Y, X = np.meshgrid( ky, zed)
    Z = np.abs(qflx_t_zed_kx_ky)[time_idx, :, kx_idx, :]

    eps_rel = 1e-4
    im = ax.pcolormesh(X, Y, Z, 
    #im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=max(Z.min(), eps_rel*Z.max()), vmax=Z.max()),
               shading='auto', cmap='inferno')

    ax.set_xlabel(r"$\zeta$ (scaled)")
    ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_ylabel(r"$k_y \rho_i$")

    return fig, ax, im, time[time_idx], kx[kx_idx]


def plot_flux_spectra_kx_ky(run, fig=None, ax=None, species_idx=0, tube=0, time_idx=-1, normalise_ky=False):

    qflx_t_zed_kx_ky, time, zed, kx, ky = run.read_flux_spectra(species_idx, tube)

    qflx_zed_kx_ky = qflx_t_zed_kx_ky[time_idx]

    # zeta-summed
    dl_over_B_avg = run.dl_over_B_avg()
    qflx_kx_ky = np.sum(qflx_zed_kx_ky*dl_over_B_avg[:,None,None], axis=0)

    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    Y, X = np.meshgrid( ky, kx)

    Z = np.abs(qflx_kx_ky)

    if normalise_ky:
        for i_ky in range(len(ky)):
            Z[:,i_ky] = Z[:,i_ky]/ky[i_ky]

    eps_rel = 1e-4
    im = ax.pcolormesh(X, Y, Z, 
    #im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=max(Z.min(), eps_rel*Z.max()), vmax=Z.max()),
               shading='auto', cmap='inferno')

    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel(r"$k_y \rho_i$")
    ax.set_title(r"$Q_{k_x, k_y} (t=%.2f)$" % (time[time_idx]))

    return fig, ax, im
