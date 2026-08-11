"""Rosenbluth-Hinton (RH) residual zonal-flow test: computes and plots the RH inertia/flux/energy quantities used to benchmark zonal-flow damping against theory."""

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
from stella_diagnostics.io.codes import get_nspecies


def get_RH_inertia(run, species_idx="sum", kx_max=1e5, idxs_kx=None):

    # kx indices
    kx_all, _, zed = run.get_kx_ky_zed()
    if idxs_kx is None:
        idxs_kx = np.arange(len(kx_all))
    idxs_kx = idxs_kx[ np.abs(kx_all[idxs_kx]) <= kx_max ]
    kx_vals = kx_all[idxs_kx]

    if species_idx == "sum":
        # Sum over all species to get total RH inertia
        nspecies = get_nspecies(run.ncdata)

        RH_inertia_zed_kx_ri = run.ncdata.variables['RH_inertia'][0,0,:,idxs_kx,:]

        for i_spec in np.arange(nspecies-1):
            RH_inertia_zed_kx_ri += run.ncdata.variables['RH_inertia'][i_spec+1,0,:,idxs_kx,:]

    else:
        # Evaluate RH inertia from one species only
        RH_inertia_zed_kx_ri = run.ncdata.variables['RH_inertia'][species_idx,0,:,idxs_kx,:]

    # Convert to complex
    RH_inertia_zed_kx = RH_inertia_zed_kx_ri[:,:,0] + 1j*RH_inertia_zed_kx_ri[:,:,1]

    return RH_inertia_zed_kx, zed, kx_vals


def get_RH_fluxes(run, species_idx="sum", passing_trapped="both", time_min=0, time_max=1e10, time_idx_skip=1, kx_max=1e5, idxs_kx=None, fphi=1, fapar=1, fbpar=1, fcoll=1):

    # Determine time indices
    time_idx_min = run.get_time_idx(time_min)
    time_idx_max = run.get_time_idx(time_max)
    time_all = run.get_time_array()
    time = time_all[time_idx_min:time_idx_max-1:time_idx_skip]

    # kx indices
    kx_all, ky, zed = run.get_kx_ky_zed()
    if idxs_kx is None:
        idxs_kx = np.arange(len(kx_all))
    idxs_kx = idxs_kx[ np.abs(kx_all[idxs_kx]) <= kx_max ]
    kx_vals = kx_all[idxs_kx]

    nspecies = get_nspecies(run.ncdata)

    # Get shape of array
    try:
        RH_fluxes_phi_even_t_zed_kx_ky_ri = run.ncdata.variables['RH_fluxes_phi_even'][time_idx_min:time_idx_max-1:time_idx_skip,0,0,:,idxs_kx,:,:]
    except:
        # For backwards compatibility (<1st April 2026)
        RH_fluxes_phi_even_t_zed_kx_ky_ri = run.ncdata.variables['RH_fluxes_phi_even_passing'][time_idx_min:time_idx_max-1:time_idx_skip,0,0,:,idxs_kx,:,:]
    
    # Start with zero-filled arrays
    RH_fluxes_phi_even_t_zed_kx_ky_ri[:] = 0
    RH_fluxes_phi_odd_t_zed_kx_ky_ri   = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)
    RH_fluxes_apar_even_t_zed_kx_ky_ri = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)
    RH_fluxes_apar_odd_t_zed_kx_ky_ri  = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)
    RH_fluxes_bpar_even_t_zed_kx_ky_ri = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)
    RH_fluxes_bpar_odd_t_zed_kx_ky_ri  = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)
    RH_fluxes_coll_t_zed_kx_ky_ri      = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)

    # Sum over species
    for i_spec in np.arange(nspecies):

        if species_idx == "sum" or species_idx == i_spec:

            try:

                if passing_trapped == "passing" or passing_trapped == "both":
                    RH_fluxes_phi_even_t_zed_kx_ky_ri += run.ncdata.variables['RH_fluxes_phi_even_passing'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    RH_fluxes_phi_odd_t_zed_kx_ky_ri  += run.ncdata.variables['RH_fluxes_phi_odd_passing'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    try:
                        RH_fluxes_apar_even_t_zed_kx_ky_ri += run.ncdata.variables['RH_fluxes_apar_even_passing'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        RH_fluxes_apar_odd_t_zed_kx_ky_ri  += run.ncdata.variables['RH_fluxes_apar_odd_passing'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    except:
                        pass

                    try:
                        RH_fluxes_bpar_even_t_zed_kx_ky_ri += run.ncdata.variables['RH_fluxes_bpar_even_passing'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        RH_fluxes_bpar_odd_t_zed_kx_ky_ri  += run.ncdata.variables['RH_fluxes_bpar_odd_passing'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    except:
                        pass


                if passing_trapped == "trapped" or passing_trapped == "both":
                    RH_fluxes_phi_even_t_zed_kx_ky_ri += run.ncdata.variables['RH_fluxes_phi_even_trapped'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    RH_fluxes_phi_odd_t_zed_kx_ky_ri  += run.ncdata.variables['RH_fluxes_phi_odd_trapped'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    try:
                        RH_fluxes_apar_even_t_zed_kx_ky_ri += run.ncdata.variables['RH_fluxes_apar_even_trapped'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        RH_fluxes_apar_odd_t_zed_kx_ky_ri  += run.ncdata.variables['RH_fluxes_apar_odd_trapped'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    except:
                        pass

                    try:
                        RH_fluxes_bpar_even_t_zed_kx_ky_ri += run.ncdata.variables['RH_fluxes_bpar_even_trapped'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        RH_fluxes_bpar_odd_t_zed_kx_ky_ri  += run.ncdata.variables['RH_fluxes_bpar_odd_trapped'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    except:
                        pass


            # For backwards compatibility (<1st April 2026)
            except:
                RH_fluxes_phi_even_t_zed_kx_ky_ri += run.ncdata.variables['RH_fluxes_phi_even'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                RH_fluxes_phi_odd_t_zed_kx_ky_ri  += run.ncdata.variables['RH_fluxes_phi_odd'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                try:
                    RH_fluxes_apar_even_t_zed_kx_ky_ri += run.ncdata.variables['RH_fluxes_apar_even'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    RH_fluxes_apar_odd_t_zed_kx_ky_ri  += run.ncdata.variables['RH_fluxes_apar_odd'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                except:
                    pass

                try:
                    RH_fluxes_bpar_even_t_zed_kx_ky_ri += run.ncdata.variables['RH_fluxes_bpar_even'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    RH_fluxes_bpar_odd_t_zed_kx_ky_ri  += run.ncdata.variables['RH_fluxes_bpar_odd'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                except:
                    pass

            # Collisional flux
            try:
                tmp  = run.ncdata.variables['RH_fluxes_collisional'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:]

                RH_fluxes_coll_t_zed_kx_ky_ri[:,:,:,0,:]  += tmp

#                    # Normalise by -1/(i*kx) to get P_RH contribution in same way as NL fluxes
#                    RH_fluxes_coll_t_zed_kx_ky_ri[:,:,:,0,0]  += -tmp[:,:,:,1]/kx_vals[None,None,:]
#                    RH_fluxes_coll_t_zed_kx_ky_ri[:,:,:,0,1]  +=  tmp[:,:,:,0]/kx_vals[None,None,:]
            
            except Exception as e:
                print(e)
                pass

    # Add up contributions from {phi, Apar, Bpar}
    RH_fluxes_even_t_zed_kx_ky_ri = fphi*RH_fluxes_phi_even_t_zed_kx_ky_ri + fapar*RH_fluxes_apar_even_t_zed_kx_ky_ri + fbpar*RH_fluxes_bpar_even_t_zed_kx_ky_ri + fcoll*RH_fluxes_coll_t_zed_kx_ky_ri
    RH_fluxes_odd_t_zed_kx_ky_ri  = fphi*RH_fluxes_phi_odd_t_zed_kx_ky_ri  + fapar*RH_fluxes_apar_odd_t_zed_kx_ky_ri  + fbpar*RH_fluxes_bpar_odd_t_zed_kx_ky_ri

    RH_fluxes_even_t_zed_kx_ky = RH_fluxes_even_t_zed_kx_ky_ri[:,:,:,:,0] + 1j*RH_fluxes_even_t_zed_kx_ky_ri[:,:,:,:,1]
    RH_fluxes_odd_t_zed_kx_ky = RH_fluxes_odd_t_zed_kx_ky_ri[:,:,:,:,0] + 1j*RH_fluxes_odd_t_zed_kx_ky_ri[:,:,:,:,1]

    return RH_fluxes_even_t_zed_kx_ky, RH_fluxes_odd_t_zed_kx_ky, time, zed, kx_vals, ky


def get_RH_phi_I(run, species_idx="sum", time_min=0, time_max=1e10, time_idx_skip=1, kx_max=1e5, idxs_kx=None):

    # Determine time indices
    time_idx_min = run.get_time_idx(time_min)
    time_idx_max = run.get_time_idx(time_max)
    time_all = run.get_time_array()
    time = time_all[time_idx_min:time_idx_max-1:time_idx_skip]

    # kx indices
    kx_all, ky, zed = run.get_kx_ky_zed()
    if idxs_kx is None:
        idxs_kx = np.arange(len(kx_all))

    idxs_kx = idxs_kx[ np.abs(kx_all[idxs_kx]) <= kx_max ]
    kx_vals = kx_all[idxs_kx]

    # Load RH_phi_I
    if species_idx == "sum":
        # Sum over all species to get total RH phi*I
        nspecies = get_nspecies(run.ncdata)

        RH_phi_I_t_zed_kx_ri = run.ncdata.variables['RH_phi_I'][time_idx_min:time_idx_max-1:time_idx_skip,0,0,:,idxs_kx,:]

        for i_spec in np.arange(nspecies-1):
            RH_phi_I_t_zed_kx_ri += run.ncdata.variables['RH_phi_I'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec+1,0,:,idxs_kx,:]

    else:
        # Evaluate RH phi*I from one species only
        RH_phi_I_t_zed_kx_ri = run.ncdata.variables['RH_phi_I'][time_idx_min:time_idx_max-1:time_idx_skip,species_idx,0,:,idxs_kx,:]

    RH_phi_I_t_zed_kx = RH_phi_I_t_zed_kx_ri[:,:,:,0] + 1j*RH_phi_I_t_zed_kx_ri[:,:,:,1]

    return RH_phi_I_t_zed_kx, time, zed, kx_vals


def get_RH_fluxes_t_kx(run, species_idx="sum", passing_trapped="both", time_min=0, time_max=1e10, time_idx_skip=1, kx_max=1e5, idxs_kx=None, fphi=1, fapar=1, fbpar=1, fcoll=1):

    RH_fluxes_even_t_zed_kx_ky, RH_fluxes_odd_t_zed_kx_ky,\
            time, zed, kx_vals, ky_vals \
            = run.get_RH_fluxes(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll)

    dl_over_B_avg = run.dl_over_B_avg()

    RH_fluxes_even_t_kx = np.sum(RH_fluxes_even_t_zed_kx_ky*dl_over_B_avg[None,:,None,None], axis=(1,3))
    RH_fluxes_odd_t_kx  = np.sum(RH_fluxes_odd_t_zed_kx_ky* dl_over_B_avg[None,:,None,None], axis=(1,3))

    return RH_fluxes_even_t_kx, RH_fluxes_odd_t_kx, time, kx_vals


def get_RH_phi_I_t_kx(run, species_idx="sum", time_min=0, time_max=1e10, time_idx_skip=1, kx_max=1e5, idxs_kx=None):

    RH_phi_I_t_zed_kx, time, zed, kx_vals \
            = run.get_RH_phi_I(species_idx=species_idx, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx)

    dl_over_B_avg = run.dl_over_B_avg()

    RH_phi_I_t_kx = np.sum(RH_phi_I_t_zed_kx*dl_over_B_avg[None,:,None], axis=1)

    return RH_phi_I_t_kx, time, kx_vals


def get_E_RH_t_kx(run, species_idx="sum", time_min=0, time_max=1e10, kx_max=1e5, idxs_kx=None):

    RH_phi_I_t_zed_kx, time, zed, kx_vals \
            = run.get_RH_phi_I(species_idx=species_idx, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)

    dl_over_B_avg = run.dl_over_B_avg()

    RH_phi_I_t_kx = np.sum(RH_phi_I_t_zed_kx*dl_over_B_avg[None,:,None], axis=1)

    RH_inertia_zed_kx, zed, kx = run.get_RH_inertia(species_idx=species_idx, kx_max=kx_max, idxs_kx=idxs_kx)
    RH_inertia_kx = np.sum(dl_over_B_avg[:,None]*RH_inertia_zed_kx, axis=0)

    # Evaluate 1-Gamma0 (single species!)
    Gamma0_vals = np.zeros_like(kx)
    for i_kx, kx_val in enumerate(kx):
        shat   = run.ncdata.variables['shat'].getValue()
        gds22  = run.ncdata.variables['gds22'][:,0]/shat**2 # |nabla(x)|^2
        bmag   = run.ncdata.variables['bmag'][:,0]
        kperp2 = (kx_val/bmag)**2 * gds22
        Gamma0_vals[i_kx] = np.sum(dl_over_B_avg * specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2))

    E_RH_t_kx =  np.abs(RH_phi_I_t_kx)**2 / (2*np.abs(RH_inertia_kx[None,:])**2) * (1-Gamma0_vals)[None, :]

    return E_RH_t_kx, time, kx_vals


def get_P_RH(run, species_idx="sum", passing_trapped="both", time_min=0, time_max=1e10, time_idx_skip=1, kx_max=1e5, idxs_kx=None, fphi=1, fapar=1, fbpar=1, fcoll=1):

    RH_fluxes_even_t_kx, RH_fluxes_odd_t_kx, time, kx = \
            run.get_RH_fluxes_t_kx(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll)

    dl_over_B_avg = run.dl_over_B_avg()

    # Evaluate RH inertia
    RH_inertia_zed_kx, zed, kx = run.get_RH_inertia(species_idx=species_idx, kx_max=kx_max, idxs_kx=idxs_kx)
    RH_inertia_kx = np.sum(dl_over_B_avg[:,None]*RH_inertia_zed_kx, axis=0)

    # Obtain phi_RH*I_RH from simulation
    RH_phi_I_t_kx, time, kx_vals = run.get_RH_phi_I_t_kx(species_idx=species_idx, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx)

    # Evaluate 1-Gamma0 (single species!)
    Gamma0_vals = np.zeros_like(kx)
    for i_kx, kx_val in enumerate(kx):
        shat   = run.ncdata.variables['shat'].getValue()
        gds22  = run.ncdata.variables['gds22'][:,0]/shat**2 # |nabla(x)|^2
        bmag   = run.ncdata.variables['bmag'][:,0]
        kperp2 = (kx_val/bmag)**2 * gds22
        Gamma0_vals[i_kx] = np.sum(dl_over_B_avg * specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2))

    P_RH_even_t_kx = -np.real(1j*kx[None,:]*RH_fluxes_even_t_kx*np.conj(RH_phi_I_t_kx)) / np.abs(RH_inertia_kx[None,:])**2 * (1-Gamma0_vals)[None,:]
    P_RH_odd_t_kx  = -np.real(1j*kx[None,:]*RH_fluxes_odd_t_kx *np.conj(RH_phi_I_t_kx)) / np.abs(RH_inertia_kx[None,:])**2 * (1-Gamma0_vals)[None,:]

    return P_RH_even_t_kx, P_RH_odd_t_kx, time, kx


def plot_E_RH(run, fig=None, ax=None, time_min=0, time_max=1e10, idxs_kx=None, kx_max=1e5, colors=None):

    fig, ax = get_or_create_ax(fig, ax, nrows=1, ncols=1, figsize=(9,8))

    E_RH_t_kx, time, kx_vals = run.get_E_RH_t_kx(time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)

    # kx indices
    kx_all, ky, zed = run.get_kx_ky_zed()
    if idxs_kx is None:
        idxs_kx = np.arange(len(kx_vals))
    idxs_kx = idxs_kx[ np.abs(kx_all[idxs_kx]) <= kx_max ]
    kx_vals = kx_all[idxs_kx]

    # Evaluate phiZ
    time_idx_min = run.get_time_idx(time_min)
    time_idx_max = run.get_time_idx(time_max)
    phiZ_t_zed_kx_ri = run.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max-1,0,:,idxs_kx,0,:]
    phiZ_t_zed_kx = phiZ_t_zed_kx_ri[:,:,:,0]+1j*phiZ_t_zed_kx_ri[:,:,:,1]

    # Evaluate RH inertia
    RH_inertia_zed_kx, zed, kx = run.get_RH_inertia(species_idx="sum", kx_max=kx_max, idxs_kx=idxs_kx)

    dl_over_B_avg = run.dl_over_B_avg()

    phiZ_IRH_t_kx = np.sum((RH_inertia_zed_kx*dl_over_B_avg[:,None])[None,:,:]*phiZ_t_zed_kx, axis=1)
    RH_inertia_kx = np.sum(dl_over_B_avg[:,None]*RH_inertia_zed_kx, axis=0)

    # Evaluate 1-Gamma0 (single species!)
    Gamma0_vals = np.zeros_like(kx)
    for i_kx, kx_val in enumerate(kx):
        shat   = run.ncdata.variables['shat'].getValue()
        gds22  = run.ncdata.variables['gds22'][:,0]/shat**2 # |nabla(x)|^2
        bmag   = run.ncdata.variables['bmag'][:,0]
        kperp2 = (kx_val/bmag)**2 * gds22
        Gamma0_vals[i_kx] = np.sum(dl_over_B_avg * specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2))

    E_Z_t_kx = np.abs(phiZ_IRH_t_kx)**2 / (2*np.abs(RH_inertia_kx[None,:])**2) * (1-Gamma0_vals)

    if colors is None:
        colors = sns.color_palette("coolwarm", len(kx_vals))

    for i_kx, kx in enumerate(kx_vals):

        if kx <= 0:
            continue

        ax.plot(time, E_RH_t_kx[:,i_kx], 
                label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx], lw=2)
        ax.plot(time, E_Z_t_kx[:,i_kx], 
                                                       c=colors[i_kx], ls='--')

        #diff = np.abs(E_RH_t_kx[:,i_kx] - E_Z_t_kx[:,i_kx])/E_RH_t_kx[:,i_kx]
        #axs[1].plot(time, diff, c=colors[i_kx])

    ax.set_ylabel(r"$E_{RH}$ (solid), $E^Z_\varphi$ (dashed)")
    #axs[1].set_ylabel(r"$|E_{RH}-E^Z_\varphi|/E_{RH}$")
    ax.set_xlabel(r"$t v_{T}/a$")

    #for ax in axs:
    ax.grid(True)

    return fig, ax, E_RH_t_kx, time, kx


def plot_RH_phi_I(run, fig=None, axs=None, time_min=0, time_max=1e10, idxs_kx=None, kx_max=1e5, colors=None, colors_sim=None):

    fig, axs = get_or_create_ax(fig, axs, nrows=3, ncols=1, figsize=(6,16))
        #fig, axs = plt.subplots(nrows=3,ncols=1, figsize=(9,16))

    RH_phi_I_t_kx, time, kx_vals = run.get_RH_phi_I_t_kx(time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)

    if colors is None:
        if len(idxs_kx) > 1:
            colors = sns.color_palette("coolwarm", len(kx_vals))
        else:
            colors = ["crimson"]
    if colors_sim is None:
        if len(idxs_kx) > 1:
            colors_sim = colors
        else:
            colors_sim = ["mediumblue"]

    # Evaluate RH inertia
    RH_inertia_zed_kx, zed, kx_vals = run.get_RH_inertia(species_idx="sum", kx_max=kx_max, idxs_kx=idxs_kx)

    dl_over_B_avg = run.dl_over_B_avg()

    # Evaluate phiZ
    if idxs_kx is None:
        idxs_kx = np.arange(len(kx_vals))

    time_idx_min = run.get_time_idx(time_min)
    time_idx_max = run.get_time_idx(time_max)
    phiZ_t_zed_kx_ri = run.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max-1,0,:,idxs_kx,0,:]
    phiZ_t_zed_kx_all = phiZ_t_zed_kx_ri[:,:,:,0]+1j*phiZ_t_zed_kx_ri[:,:,:,1]
    kx_all, _, _ = run.get_kx_ky_zed()
    kx_all = kx_all[idxs_kx]

    phiZ_t_zed_kx = phiZ_t_zed_kx_all[:,:,np.abs(kx_all)<=kx_max]
    kx = kx_all[np.abs(kx_all)<=kx_max]

    phiZ_IRH_t_kx = np.sum((RH_inertia_zed_kx*dl_over_B_avg[:,None])[None,:,:]*phiZ_t_zed_kx, axis=1)

    rel_diff_t_kx = (RH_phi_I_t_kx - phiZ_IRH_t_kx)/np.abs(RH_phi_I_t_kx)

    for i_kx, kx in enumerate(kx_vals):

        axs[0].plot(time, np.real(RH_phi_I_t_kx[:,i_kx]), 
                    label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
        axs[0].plot(time, np.imag(RH_phi_I_t_kx[:,i_kx]), 
                    ls="--", c=colors[i_kx])

        axs[1].plot(time, np.real(phiZ_IRH_t_kx[:,i_kx]), 
                    label=r"$k_x \rho = %.3f$" % (kx), c=colors_sim[i_kx])
        axs[1].plot(time, np.imag(phiZ_IRH_t_kx[:,i_kx]), 
                    ls="--", c=colors_sim[i_kx])

        axs[2].plot(time, np.abs(RH_phi_I_t_kx[:,i_kx]), 
                    label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
        axs[2].plot(time, np.abs(phiZ_IRH_t_kx[:,i_kx]), 
                    label=r"$k_x \rho = %.3f$" % (kx), c=colors_sim[i_kx], alpha=0.5)

        #axs[2].plot(time, np.real(rel_diff_t_kx[:,i_kx]), 
        #            label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
        #axs[2].plot(time, np.imag(rel_diff_t_kx[:,i_kx]), 
        #            ls="--", c=colors[i_kx])

    axs[0].set_ylabel(r"$\varphi_\mathrm{RH} I_\mathrm{RH}$")
    axs[1].set_ylabel(r"$\varphi I_{RH}$")
    axs[2].set_ylabel(r"Both")
    #axs[2].set_ylabel(r"Relative diff")
    axs[2].set_xlabel(r"$t v_{T}/a$")

    for ax in axs:
        ax.grid(True)

    return fig, axs, RH_phi_I_t_kx, time, kx


def plot_RH_fluxes(run, fig=None, axs=None, time_min=0, time_max=1e10, species_idx="sum", passing_trapped="both", idxs_kx=None, kx_max=1e5, colors=None, fphi=1, fapar=1, fbpar=1, fcoll=1):

    fig, axs = get_or_create_ax(fig, axs, nrows=3, ncols=1, figsize=(9,16))

    RH_fluxes_even_t_kx, RH_fluxes_odd_t_kx, time, kx_vals = \
            run.get_RH_fluxes_t_kx(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll)

    if colors is None:
        colors = sns.color_palette("coolwarm", len(kx_vals))

    for i_kx, kx in enumerate(kx_vals):

        axs[0].plot(time, np.real(RH_fluxes_even_t_kx[:,i_kx]), 
                    label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
        axs[0].plot(time, np.imag(RH_fluxes_even_t_kx[:,i_kx]), 
                    ls="--", c=colors[i_kx])
        axs[1].plot(time, np.real(RH_fluxes_odd_t_kx[:,i_kx]),  
                    label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
        axs[1].plot(time, np.imag(RH_fluxes_odd_t_kx[:,i_kx]), 
                    ls="--", c=colors[i_kx])
        axs[2].plot(time, np.real(RH_fluxes_even_t_kx[:,i_kx]+RH_fluxes_odd_t_kx[:,i_kx]),  
                    label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
        axs[2].plot(time, np.imag(RH_fluxes_even_t_kx[:,i_kx]+RH_fluxes_odd_t_kx[:,i_kx]), 
                    ls="--", c=colors[i_kx])

    axs[0].set_ylabel(r"$F_\mathrm{RH}^+$")
    axs[1].set_ylabel(r"$F_\mathrm{RH}^-$")
    axs[2].set_ylabel(r"$F_\mathrm{RH}^+ + F_\mathrm{RH}^-$")
    axs[2].set_xlabel(r"$t v_{T}/a$")

    for ax in axs:
        ax.grid(True)

    return fig, axs, RH_fluxes_even_t_kx, RH_fluxes_odd_t_kx, time, kx


def plot_P_RH(run, fig=None, axs=None, time_min=0, time_max=1e10, species_idx="sum", passing_trapped="both", idxs_kx=None, kx_max=1e5, colors=None, fphi=1, fapar=1, fbpar=1, fcoll=1, D_hyper=None):

    fig, axs = get_or_create_ax(fig, axs, nrows=3, ncols=1, figsize=(9,16))

    # Evaluate from NL fluxes
    P_RH_phi_even_t_kx, P_RH_phi_odd_t_kx, time, kx_vals = \
                run.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=0, fbpar=0, fcoll=0)
    P_RH_phi_t_kx  = P_RH_phi_even_t_kx + P_RH_phi_odd_t_kx
    P_RH_even_t_kx = np.copy(P_RH_phi_even_t_kx )
    P_RH_odd_t_kx  = np.copy(P_RH_phi_odd_t_kx  )

    if fapar != 0:
        P_RH_apar_even_t_kx, P_RH_apar_odd_t_kx, time, kx_vals = \
                run.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx, fphi=0, fapar=fapar, fbpar=0, fcoll=0)
        P_RH_apar_t_kx  = P_RH_apar_even_t_kx + P_RH_apar_odd_t_kx
        P_RH_even_t_kx += P_RH_apar_even_t_kx
        P_RH_odd_t_kx  += P_RH_apar_odd_t_kx
    else:
        P_RH_apar_even_t_kx = np.zeros_like(P_RH_phi_t_kx) #None
        P_RH_apar_odd_t_kx  = np.zeros_like(P_RH_phi_t_kx) #None
        P_RH_apar_t_kx      = np.zeros_like(P_RH_phi_t_kx) #None

    if fbpar != 0:
        P_RH_bpar_even_t_kx, P_RH_bpar_odd_t_kx, time, kx_vals = \
                run.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx, fphi=0, fapar=0, fbpar=fbpar, fcoll=0)
        P_RH_bpar_t_kx  = P_RH_bpar_even_t_kx + P_RH_bpar_odd_t_kx
        P_RH_even_t_kx += P_RH_bpar_even_t_kx
        P_RH_odd_t_kx  += P_RH_bpar_odd_t_kx
    else:
        P_RH_bpar_even_t_kx = np.zeros_like(P_RH_phi_t_kx) #None
        P_RH_bpar_odd_t_kx  = np.zeros_like(P_RH_phi_t_kx) #None
        P_RH_bpar_t_kx      = np.zeros_like(P_RH_phi_t_kx) #None

    if fcoll != 0:
        P_RH_coll_even_t_kx, P_RH_coll_odd_t_kx, time, kx_vals = \
                run.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx, fphi=0, fapar=0, fbpar=0, fcoll=fcoll)
        P_RH_coll_t_kx  = P_RH_coll_even_t_kx + P_RH_coll_odd_t_kx
        P_RH_even_t_kx += P_RH_coll_even_t_kx
        P_RH_odd_t_kx  += P_RH_coll_odd_t_kx
    else:
        P_RH_coll_even_t_kx = np.zeros_like(P_RH_phi_t_kx) #None
        P_RH_coll_odd_t_kx  = np.zeros_like(P_RH_phi_t_kx) #None
        P_RH_coll_t_kx      = np.zeros_like(P_RH_phi_t_kx) #None


    # Evaluate numerically from time trace
    E_RH_t_kx, time, kx_vals = run.get_E_RH_t_kx(species_idx=species_idx, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)
    P_RH_t_kx_num = np.gradient(E_RH_t_kx, time, axis=0)
    #P_RH_t_kx_num = np.gradient(E_RH_t_kx, axis=0)/np.gradient(time)[:,None]

    # Evaluate hyperdissipation contribution if desired
    if D_hyper is not None:
        kperp2 = run.ncdata.variables['kperp2'][:][:,0,:,:]
        kmax = np.sqrt( kperp2.max() )
        P_RH_hyper_t_kx = -2*D_hyper * E_RH_t_kx * (kx_vals[None,:]/kmax)**4
    else:
        P_RH_hyper_t_kx = None

    for i_kx, kx in enumerate(kx_vals):

        if kx <= 0:
            continue

        if colors is None:
            #colors = sns.color_palette("coolwarm", len(kx_vals))
            c_num   = '0.5'
            c_tot   = 'k'
            c_phi   = 'mediumblue'
            c_apar  = 'crimson'
            c_bpar  = 'forestgreen'
            c_coll  = 'orange'
        else:
            c_num   = colors[i_kx] 
            c_tot   = colors[i_kx] 
            c_phi   = colors[i_kx] 
            c_apar  = colors[i_kx] 
            c_bpar  = colors[i_kx] 
            c_coll  = colors[i_kx] 

        axs[0].plot(time, P_RH_even_t_kx[:,i_kx], 
                    label=r"$k_x \rho = %.3f$" % (kx), c=c_tot, lw=2)
        axs[1].plot(time, P_RH_odd_t_kx[:,i_kx],  
                    label=r"$k_x \rho = %.3f$" % (kx), c=c_tot, lw=2)
        P_RH_tot_t_kx = P_RH_even_t_kx + P_RH_odd_t_kx
        ylabel = r"$P_\mathrm{RH}^+ + P_\mathrm{RH}^-$"
        if D_hyper is not None:
            P_RH_tot_t_kx += P_RH_hyper_t_kx
            ylabel += r"$+ P_\mathrm{RH}^\mathrm{hyper}$"

        axs[2].plot(time, P_RH_tot_t_kx[:,i_kx], c=c_tot, lw=2)
        axs[2].plot(time, P_RH_t_kx_num[:,i_kx], ls=(0, (3, 5, 1, 5, 1, 5)), lw=2,c=c_num)

        if fphi != 0:
            axs[0].plot(time, P_RH_phi_even_t_kx[:,i_kx], c=c_phi, ls='--')
            axs[1].plot(time, P_RH_phi_odd_t_kx[ :,i_kx], c=c_phi, ls='--')
            axs[2].plot(time, P_RH_phi_t_kx[     :,i_kx], c=c_phi, ls='--')

        if fapar != 0:
            axs[0].plot(time, P_RH_apar_even_t_kx[:,i_kx], c=c_apar, ls='-.')
            axs[1].plot(time, P_RH_apar_odd_t_kx[ :,i_kx], c=c_apar, ls='-.')
            axs[2].plot(time, P_RH_apar_t_kx[     :,i_kx], c=c_apar, ls='-.')

        if fbpar != 0:
            axs[0].plot(time, P_RH_bpar_even_t_kx[:,i_kx], c=c_bpar, ls=':')
            axs[1].plot(time, P_RH_bpar_odd_t_kx[ :,i_kx], c=c_bpar, ls=':')
            axs[2].plot(time, P_RH_bpar_t_kx[     :,i_kx], c=c_bpar, ls=':')

        if fcoll != 0:
            axs[0].plot(time, P_RH_coll_even_t_kx[:,i_kx], c=c_coll, ls=':')
            axs[1].plot(time, P_RH_coll_odd_t_kx[ :,i_kx], c=c_coll, ls=':')
            axs[2].plot(time, P_RH_coll_t_kx[     :,i_kx], c=c_coll, ls=':')

    axs[0].set_ylabel(r"$P_\mathrm{RH}^+$")
    axs[1].set_ylabel(r"$P_\mathrm{RH}^-$")
    axs[2].set_ylabel(ylabel)
    axs[2].set_xlabel(r"$t v_{T}/a$")

    for ax in axs:
        ax.grid(True)

    return fig, axs, P_RH_even_t_kx,      P_RH_odd_t_kx, \
                     P_RH_phi_even_t_kx,  P_RH_phi_odd_t_kx, \
                     P_RH_apar_even_t_kx, P_RH_apar_odd_t_kx,\
                     P_RH_bpar_even_t_kx, P_RH_bpar_odd_t_kx,\
                     P_RH_coll_even_t_kx, P_RH_coll_odd_t_kx,\
                     P_RH_hyper_t_kx, time, kx


def get_RH_integrand_mu_vpa_zed_kx(run, species_idx=0):
    RH_integrand_even_mu_vpa_zed_kx_ri = run.ncdata.variables['RH_integrand_even'][:,:,species_idx,0,:,:,:]
    RH_integrand_even_mu_vpa_zed_kx = RH_integrand_even_mu_vpa_zed_kx_ri[:,:,:,:,0] + 1j*RH_integrand_even_mu_vpa_zed_kx_ri[:,:,:,:,1]

    RH_integrand_odd_mu_vpa_zed_kx_ri = run.ncdata.variables['RH_integrand_odd'][:,:,species_idx,0,:,:,:]
    RH_integrand_odd_mu_vpa_zed_kx = RH_integrand_odd_mu_vpa_zed_kx_ri[:,:,:,:,0] + 1j*RH_integrand_odd_mu_vpa_zed_kx_ri[:,:,:,:,1]

    kx, _, zed = run.get_kx_ky_zed()

    vpa    = run.ncdata.variables['vpa']
    mu     = run.ncdata.variables['mu']

    return RH_integrand_even_mu_vpa_zed_kx, RH_integrand_odd_mu_vpa_zed_kx, mu, vpa, zed, kx
