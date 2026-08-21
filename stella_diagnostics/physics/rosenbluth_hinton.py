"""Rosenbluth-Hinton (RH) residual zonal-flow test: computes and plots the RH inertia/flux/energy quantities used to benchmark zonal-flow damping against theory.

``species_idx`` throughout this module is dual-typed: either an integer
index for one species, or the literal string ``"sum"`` (the default) to
sum/include all species. Not indicated by the name alone -- flagged here
since it isn't obvious from any single call site."""

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
from stella_diagnostics.io.codes import get_nspecies, get_rho_label, get_vt_label


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


def get_P_RH_coll_over_vnew_E_RH_t(run, vnew=None, species_idx="sum", time_min=0, time_max=1e10, kx_max=1e5, idxs_kx=None):
    """Collisional RH power transfer normalized by vnew*E_RH, vs time (summed
    over kx) -- a dimensionless proxy for the instantaneous collisional
    damping rate of the zonal flow: P_RH_coll ~ dE_RH/dt from collisions
    alone, so P_RH_coll/E_RH is ~ a damping rate, and dividing by vnew
    expresses it in units of the collision frequency (compare against
    get_P_RH_coll_normalized_vs_kx in scan/rh_collisional_kx.py, which
    does the analogous normalization for a single time-window mean vs kx
    instead of this function's time-resolved, kx-summed view).

    vnew defaults to this run's own species-summed collision frequency
    (the 'vnew' netCDF variable written by stella), not a value from a
    separate collisionality scan -- pass vnew explicitly to override.
    """
    if vnew is None:
        vnew = float(np.sum(run.ncdata.variables['vnew'][:]))

    E_RH_t_kx, time, kx_vals = run.get_E_RH_t_kx(species_idx=species_idx, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)
    P_RH_coll_even_t_kx, P_RH_coll_odd_t_kx, time_p, kx_vals_p = run.get_P_RH(species_idx=species_idx, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx, fphi=0, fapar=0, fbpar=0, fcoll=1)

    E_RH_t = E_RH_t_kx.sum(axis=1)
    P_RH_coll_t = (P_RH_coll_even_t_kx + P_RH_coll_odd_t_kx).sum(axis=1)

    return P_RH_coll_t / (vnew * E_RH_t), time


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
                label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx), c=colors[i_kx], lw=2)
        ax.plot(time, E_Z_t_kx[:,i_kx], 
                                                       c=colors[i_kx], ls='--')

        #diff = np.abs(E_RH_t_kx[:,i_kx] - E_Z_t_kx[:,i_kx])/E_RH_t_kx[:,i_kx]
        #axs[1].plot(time, diff, c=colors[i_kx])

    ax.set_ylabel(r"$E_{RH}$ (solid), $E^Z_\varphi$ (dashed)")
    #axs[1].set_ylabel(r"$|E_{RH}-E^Z_\varphi|/E_{RH}$")
    ax.set_xlabel(r"$t %s/a$" % get_vt_label(run.ncdata))

    #for ax in axs:
    ax.grid(True)

    return fig, ax, E_RH_t_kx, time, kx


def plot_RH_phi_I(run, fig=None, axs=None, time_min=0, time_max=1e10, idxs_kx=None, kx_max=1e5, colors=None, colors_sim=None):

    fig, axs = get_or_create_ax(fig, axs, nrows=3, ncols=1, figsize=(6,16))
        #fig, axs = plt.subplots(nrows=3,ncols=1, figsize=(9,16))

    RH_phi_I_t_kx, time, kx_vals = run.get_RH_phi_I_t_kx(time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)

    # idxs_kx must be resolved before it's used (by len()) just below --
    # previously read before this None-check ran, so calling this with
    # the documented default (idxs_kx=None) always raised TypeError.
    if idxs_kx is None:
        idxs_kx = np.arange(len(kx_vals))

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
                    label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx), c=colors[i_kx])
        axs[0].plot(time, np.imag(RH_phi_I_t_kx[:,i_kx]), 
                    ls="--", c=colors[i_kx])

        axs[1].plot(time, np.real(phiZ_IRH_t_kx[:,i_kx]), 
                    label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx), c=colors_sim[i_kx])
        axs[1].plot(time, np.imag(phiZ_IRH_t_kx[:,i_kx]), 
                    ls="--", c=colors_sim[i_kx])

        axs[2].plot(time, np.abs(RH_phi_I_t_kx[:,i_kx]), 
                    label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx), c=colors[i_kx])
        axs[2].plot(time, np.abs(phiZ_IRH_t_kx[:,i_kx]), 
                    label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx), c=colors_sim[i_kx], alpha=0.5)

        #axs[2].plot(time, np.real(rel_diff_t_kx[:,i_kx]), 
        #            label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx), c=colors[i_kx])
        #axs[2].plot(time, np.imag(rel_diff_t_kx[:,i_kx]), 
        #            ls="--", c=colors[i_kx])

    axs[0].set_ylabel(r"$\varphi_\mathrm{RH} I_\mathrm{RH}$")
    axs[1].set_ylabel(r"$\varphi I_{RH}$")
    axs[2].set_ylabel(r"Both")
    #axs[2].set_ylabel(r"Relative diff")
    axs[2].set_xlabel(r"$t %s/a$" % get_vt_label(run.ncdata))

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
                    label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx), c=colors[i_kx])
        axs[0].plot(time, np.imag(RH_fluxes_even_t_kx[:,i_kx]), 
                    ls="--", c=colors[i_kx])
        axs[1].plot(time, np.real(RH_fluxes_odd_t_kx[:,i_kx]),  
                    label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx), c=colors[i_kx])
        axs[1].plot(time, np.imag(RH_fluxes_odd_t_kx[:,i_kx]), 
                    ls="--", c=colors[i_kx])
        axs[2].plot(time, np.real(RH_fluxes_even_t_kx[:,i_kx]+RH_fluxes_odd_t_kx[:,i_kx]),  
                    label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx), c=colors[i_kx])
        axs[2].plot(time, np.imag(RH_fluxes_even_t_kx[:,i_kx]+RH_fluxes_odd_t_kx[:,i_kx]), 
                    ls="--", c=colors[i_kx])

    axs[0].set_ylabel(r"$F_\mathrm{RH}^+$")
    axs[1].set_ylabel(r"$F_\mathrm{RH}^-$")
    axs[2].set_ylabel(r"$F_\mathrm{RH}^+ + F_\mathrm{RH}^-$")
    axs[2].set_xlabel(r"$t %s/a$" % get_vt_label(run.ncdata))

    for ax in axs:
        ax.grid(True)

    return fig, axs, RH_fluxes_even_t_kx, RH_fluxes_odd_t_kx, time, kx


def get_P_RH_breakdown(run, time_min=0, time_max=1e10, time_idx_skip=1, species_idx="sum", passing_trapped="both", idxs_kx=None, kx_max=1e5, fphi=1, fapar=1, fbpar=1, fcoll=1, D_hyper=None):
    """No-plot equivalent of plot_P_RH's computation (same 4 isolated calls
    into get_P_RH -- phi/apar/bpar/coll channels -- combined into totals,
    plus the numerical dE_RH/dt cross-check and optional hyperdissipation
    term). Extracted so this computation can be called from inside a
    @cached function (stella_diagnostics.scan.rh_per_kx_scan), which must
    not call plotting code. plot_P_RH's own body is left untouched (not
    refactored to call this), to avoid touching an existing, working code
    path as a side effect.

    Returns the same 11 data arrays + time + kx that plot_P_RH returns,
    minus fig/axs: (P_RH_even_t_kx, P_RH_odd_t_kx, P_RH_phi_even_t_kx,
    P_RH_phi_odd_t_kx, P_RH_apar_even_t_kx, P_RH_apar_odd_t_kx,
    P_RH_bpar_even_t_kx, P_RH_bpar_odd_t_kx, P_RH_coll_even_t_kx,
    P_RH_coll_odd_t_kx, P_RH_hyper_t_kx, time, kx).
    """
    P_RH_phi_even_t_kx, P_RH_phi_odd_t_kx, time, kx_vals = \
            run.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=0, fbpar=0, fcoll=0)
    P_RH_phi_t_kx  = P_RH_phi_even_t_kx + P_RH_phi_odd_t_kx
    P_RH_even_t_kx = np.copy(P_RH_phi_even_t_kx )
    P_RH_odd_t_kx  = np.copy(P_RH_phi_odd_t_kx  )

    if fapar != 0:
        P_RH_apar_even_t_kx, P_RH_apar_odd_t_kx, time, kx_vals = \
                run.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx, fphi=0, fapar=fapar, fbpar=0, fcoll=0)
        P_RH_apar_t_kx  = P_RH_apar_even_t_kx + P_RH_apar_odd_t_kx
        P_RH_even_t_kx += P_RH_apar_even_t_kx
        P_RH_odd_t_kx  += P_RH_apar_odd_t_kx
    else:
        P_RH_apar_even_t_kx = np.zeros_like(P_RH_phi_t_kx)
        P_RH_apar_odd_t_kx  = np.zeros_like(P_RH_phi_t_kx)
        P_RH_apar_t_kx      = np.zeros_like(P_RH_phi_t_kx)

    if fbpar != 0:
        P_RH_bpar_even_t_kx, P_RH_bpar_odd_t_kx, time, kx_vals = \
                run.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx, fphi=0, fapar=0, fbpar=fbpar, fcoll=0)
        P_RH_bpar_t_kx  = P_RH_bpar_even_t_kx + P_RH_bpar_odd_t_kx
        P_RH_even_t_kx += P_RH_bpar_even_t_kx
        P_RH_odd_t_kx  += P_RH_bpar_odd_t_kx
    else:
        P_RH_bpar_even_t_kx = np.zeros_like(P_RH_phi_t_kx)
        P_RH_bpar_odd_t_kx  = np.zeros_like(P_RH_phi_t_kx)
        P_RH_bpar_t_kx      = np.zeros_like(P_RH_phi_t_kx)

    if fcoll != 0:
        P_RH_coll_even_t_kx, P_RH_coll_odd_t_kx, time, kx_vals = \
                run.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx, fphi=0, fapar=0, fbpar=0, fcoll=fcoll)
        P_RH_coll_t_kx  = P_RH_coll_even_t_kx + P_RH_coll_odd_t_kx
        P_RH_even_t_kx += P_RH_coll_even_t_kx
        P_RH_odd_t_kx  += P_RH_coll_odd_t_kx
    else:
        P_RH_coll_even_t_kx = np.zeros_like(P_RH_phi_t_kx)
        P_RH_coll_odd_t_kx  = np.zeros_like(P_RH_phi_t_kx)
        P_RH_coll_t_kx      = np.zeros_like(P_RH_phi_t_kx)

    # Evaluate numerically from time trace.
    # NOTE: get_E_RH_t_kx has no time_idx_skip of its own (always evaluates
    # every timestep), so its own (time_E_RH, kx_E_RH) can have a different
    # length than the phi/apar/bpar/coll arrays above whenever
    # time_idx_skip != 1 -- keep them local, don't let them clobber the
    # time/kx_vals this function returns (a real bug when this function
    # didn't accept time_idx_skip at all, since callers always got 1;
    # exposed now that get_RH_power_time_averages calls this with
    # time_idx_skip=10). plot_P_RH has the same E_RH-vs-time_idx_skip
    # mismatch potential in its own independent computation, but doesn't
    # expose time_idx_skip as a parameter, so it isn't reachable there.
    E_RH_t_kx, time_E_RH, kx_E_RH = run.get_E_RH_t_kx(species_idx=species_idx, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)
    P_RH_t_kx_num = np.gradient(E_RH_t_kx, time_E_RH, axis=0)

    # Evaluate hyperdissipation contribution if desired.
    # NOTE: still on the undecimated (time_E_RH, kx_E_RH) grid, so
    # P_RH_hyper_t_kx's shape won't match the other returned P_RH_*_t_kx
    # arrays' shape whenever time_idx_skip != 1 -- not exercised by any
    # current caller (none pass both D_hyper and time_idx_skip != 1), so
    # left as-is rather than guessed at; a real fix would need to resample
    # onto the decimated time grid.
    if D_hyper is not None:
        kperp2 = run.ncdata.variables['kperp2'][:][:,0,:,:]
        kmax = np.sqrt( kperp2.max() )
        P_RH_hyper_t_kx = -2*D_hyper * E_RH_t_kx * (kx_E_RH[None,:]/kmax)**4
    else:
        P_RH_hyper_t_kx = None

    return (P_RH_even_t_kx, P_RH_odd_t_kx,
            P_RH_phi_even_t_kx, P_RH_phi_odd_t_kx,
            P_RH_apar_even_t_kx, P_RH_apar_odd_t_kx,
            P_RH_bpar_even_t_kx, P_RH_bpar_odd_t_kx,
            P_RH_coll_even_t_kx, P_RH_coll_odd_t_kx,
            P_RH_hyper_t_kx, time, kx_vals)


def plot_P_RH(run, fig=None, axs=None, time_min=0, time_max=1e10, species_idx="sum", passing_trapped="both", idxs_kx=None, kx_max=1e5, colors=None, fphi=1, fapar=1, fbpar=1, fcoll=1, D_hyper=None, combine_fields=False, combine_even_odd=False):
    """...
    By default (combine_fields=False, combine_even_odd=False) shows the
    full breakdown: phi/apar/bpar/coll each get their own color, and
    even-/odd-parity contributions each get their own panel.

    combine_even_odd=True drops the separate even/odd panels, keeping
    only the always-present total+numerical-cross-check panel; each
    field's even+odd sum is drawn there instead.

    combine_fields=True drops the per-field phi/apar/bpar breakdown --
    coll is kept separate regardless of this flag, since it's a
    different physical mechanism (collisional dissipation), not a field
    -- only each panel's own field-summed total is drawn.

    Both True collapses to the single grand-total + numerical panel with
    no other lines (the pre-breakdown view).

    axs (returned) is always a flat list of the axes actually created:
    length 1 (just the total/cross-check panel) if combine_even_odd,
    else length 3 ([even panel, odd panel, total panel]) -- axs[-1] is
    always the total/cross-check panel, regardless of mode.
    """
    nrows = 1 if combine_even_odd else 3
    fig, axs = get_or_create_ax(fig, axs, nrows=nrows, ncols=1, figsize=(12, 6 if combine_even_odd else 16))
    axs = list(np.atleast_1d(axs))

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

    ax_total = axs[-1]
    ylabel = r"$P_\mathrm{RH}$" if combine_even_odd else r"$P_\mathrm{RH}^+ + P_\mathrm{RH}^-$"
    if D_hyper is not None:
        ylabel += r"$+ P_\mathrm{RH}^\mathrm{hyper}$"

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

        P_RH_tot_t_kx = P_RH_even_t_kx + P_RH_odd_t_kx
        if D_hyper is not None:
            P_RH_tot_t_kx = P_RH_tot_t_kx + P_RH_hyper_t_kx

        ax_total.plot(time, P_RH_tot_t_kx[:,i_kx], c=c_tot, lw=2, label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx))
        ax_total.plot(time, P_RH_t_kx_num[:,i_kx], ls=(0, (3, 5, 1, 5, 1, 5)), lw=2, c=c_num)

        if not combine_even_odd:
            axs[0].plot(time, P_RH_even_t_kx[:,i_kx],
                        label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx), c=c_tot, lw=2)
            axs[1].plot(time, P_RH_odd_t_kx[:,i_kx],
                        label=r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx), c=c_tot, lw=2)

        if not combine_fields:
            if fphi != 0:
                if combine_even_odd:
                    ax_total.plot(time, P_RH_phi_t_kx[:,i_kx], c=c_phi, ls='--')
                else:
                    axs[0].plot(time, P_RH_phi_even_t_kx[:,i_kx], c=c_phi, ls='--')
                    axs[1].plot(time, P_RH_phi_odd_t_kx[ :,i_kx], c=c_phi, ls='--')
                    ax_total.plot(time, P_RH_phi_t_kx[    :,i_kx], c=c_phi, ls='--')

            if fapar != 0:
                if combine_even_odd:
                    ax_total.plot(time, P_RH_apar_t_kx[:,i_kx], c=c_apar, ls='-.')
                else:
                    axs[0].plot(time, P_RH_apar_even_t_kx[:,i_kx], c=c_apar, ls='-.')
                    axs[1].plot(time, P_RH_apar_odd_t_kx[ :,i_kx], c=c_apar, ls='-.')
                    ax_total.plot(time, P_RH_apar_t_kx[    :,i_kx], c=c_apar, ls='-.')

            if fbpar != 0:
                if combine_even_odd:
                    ax_total.plot(time, P_RH_bpar_t_kx[:,i_kx], c=c_bpar, ls=':')
                else:
                    axs[0].plot(time, P_RH_bpar_even_t_kx[:,i_kx], c=c_bpar, ls=':')
                    axs[1].plot(time, P_RH_bpar_odd_t_kx[ :,i_kx], c=c_bpar, ls=':')
                    ax_total.plot(time, P_RH_bpar_t_kx[    :,i_kx], c=c_bpar, ls=':')

            if fcoll != 0:
                if combine_even_odd:
                    ax_total.plot(time, P_RH_coll_t_kx[:,i_kx], c=c_coll, ls=':')
                else:
                    axs[0].plot(time, P_RH_coll_even_t_kx[:,i_kx], c=c_coll, ls=':')
                    axs[1].plot(time, P_RH_coll_odd_t_kx[ :,i_kx], c=c_coll, ls=':')
                    ax_total.plot(time, P_RH_coll_t_kx[    :,i_kx], c=c_coll, ls=':')
        elif fcoll != 0:
            # coll is never folded into combine_fields (it's not a field) --
            # always shown separately, even when phi/apar/bpar are combined.
            if combine_even_odd:
                ax_total.plot(time, P_RH_coll_t_kx[:,i_kx], c=c_coll, ls=':')
            else:
                axs[0].plot(time, P_RH_coll_even_t_kx[:,i_kx], c=c_coll, ls=':')
                axs[1].plot(time, P_RH_coll_odd_t_kx[ :,i_kx], c=c_coll, ls=':')
                ax_total.plot(time, P_RH_coll_t_kx[    :,i_kx], c=c_coll, ls=':')

    if not combine_even_odd:
        axs[0].set_ylabel(r"$P_\mathrm{RH}^+$")
        axs[1].set_ylabel(r"$P_\mathrm{RH}^-$")
    ax_total.set_ylabel(ylabel)
    ax_total.set_xlabel(r"$t %s/a$" % get_vt_label(run.ncdata))

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
