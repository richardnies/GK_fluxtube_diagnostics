"""Particle/momentum/heat flux and energy readers: raw flux spectra from netCDF, time-integrated fluxes and energies, and flux normalisation."""

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
from stella_diagnostics.grid import nearest_index
from stella_diagnostics.io.codes import get_nspecies


def flux_norm(run):
    grho      = run.ncdata.variables['grho'][:,0]
    dl_over_b = run.dl_over_B_avg()

    flux_norm = np.sum(grho*dl_over_b)
    print("\n"+run.filename_base+": flux_norm = %e" % (flux_norm))

    return flux_norm


def read_flux_spectra(run, species_idx=0, tube=0):#, zed_slice=None, kx_slice=None, ky_slice=None, t_slice=None):

	# qflx_kxky(t, species, tube, zed, kx, ky)
    if run.code == "stella":
        qflx_t_zed_kx_ky = run.ncdata.variables['qflx_kxky'][:,species_idx, tube, :, :, :]
    elif run.code == "GX":
        qflx_t_zed_kx_ky = np.transpose(run.ncdata['Diagnostics']['HeatFlux_kxkyzst'][:,species_idx, :, :, :], axes=(0,1,3,2))

    time = run.get_time_array()
    kx, ky, zed = run.get_kx_ky_zed()

    # Check shape
    assert len(time) == np.shape(qflx_t_zed_kx_ky)[0]
    assert len(zed)  == np.shape(qflx_t_zed_kx_ky)[1]
    assert len(kx)   == np.shape(qflx_t_zed_kx_ky)[2]
    assert len(ky)   == np.shape(qflx_t_zed_kx_ky)[3]

    return qflx_t_zed_kx_ky, time, zed, kx, ky


def read_phi2_spectra(run, time_min=0, time_max=1e10, time_idx_skip=1):

    time = run.get_time_array()
    time_idx_min = run.get_time_idx(time_min)
    time_idx_max = run.get_time_idx(time_max)
    time = time[time_idx_min:time_idx_max:time_idx_skip]
    kx, ky, zed = run.get_kx_ky_zed()

    if run.code == "stella":
        # phi2_vs_kxky(t, kx, ky)
        phi2_vs_kxky = run.ncdata.variables['phi2_vs_kxky'][time_idx_min:time_idx_max:time_idx_skip]

    elif run.code == "GS2":
        phi2_vs_kxky = np.transpose(run.ncdata['phi2_by_mode'][time_idx_min:time_idx_max:time_idx_skip], axes=(0,2,1))

    elif run.code == "GX":
        if run.GX_old_version:
            phi2_vs_kxky = np.transpose(run.ncdata['Spectra']['Akxkyst'][time_idx_min:time_idx_max:time_idx_skip,:,:], axes=(0,2,1))
        else:
            phi2_vs_kxky = np.transpose(run.ncdata['Diagnostics']['Wphi_kxkyst'][time_idx_min:time_idx_max:time_idx_skip,0,:,:], axes=(0,2,1))

        # divide zonal by (1-Gamma0(kx(GX)**2))
        Gamma0 = specialfunc.iv(0, kx**2/2) * np.exp(-kx**2/2)
        #Gamma0 = specialfunc.iv(0, kx**2/4) * np.exp(-kx**2/4)
        phi2_vs_kxky[:,:,0] = phi2_vs_kxky[:,:,0]/(1-Gamma0)[None,:]

        # Factor from vT definition
        #phi2_vs_kxky = phi2_vs_kxky/4

    # Check shape
    assert len(time) == np.shape(phi2_vs_kxky)[0]
    assert len(kx)   == np.shape(phi2_vs_kxky)[1]
    assert len(ky)   == np.shape(phi2_vs_kxky)[2]

    # Make (0,0) mode NaN
#        for i_t in range(len(time)):
#            phi2_vs_kxky[i_t, 0, 0] = np.nan

    return phi2_vs_kxky, time, kx, ky


def read_W_spectra(run, time_min=0, time_max=1e10, time_idx_skip=1):

    kx, ky, _ = run.get_kx_ky_zed()
    if run.code == "stella":
        # Consider only temperature and parallel flow energy
        time = run.get_time_array()
        time_idx_min = run.get_time_idx(time_min)
        time_idx_max = run.get_time_idx(time_max)
        time = time[time_idx_min:time_idx_max:time_idx_skip]

        # temperature(t, species, tube, zed, kx, ky, ri)
        #temperature_t_s_zed_kx_ky_ri = run.ncdata.variables['temperature'][time_idx_min:time_idx_max:time_idx_skip,:,0,:,:,:,:]
        #temperature_t_s_zed_kx_ky = temperature_t_s_zed_kx_ky_ri[:,:,:,:,:,0]+1j*temperature_t_s_zed_kx_ky_ri[:,:,:,:,:,1]
        upar_t_s_zed_kx_ky_ri = run.ncdata.variables['upar'][time_idx_min:time_idx_max:time_idx_skip,:,0,:,:,:,:]
        upar_t_s_zed_kx_ky = upar_t_s_zed_kx_ky_ri[:,:,:,:,:,0]+1j*upar_t_s_zed_kx_ky_ri[:,:,:,:,:,1]
        dl_over_B_avg = run.dl_over_B_avg()
        W_vs_kxky = np.sum( np.abs(upar_t_s_zed_kx_ky)**2 * dl_over_B_avg[None,None,:,None,None], axis=(1,2))
        #W_vs_kxky = np.sum( (np.abs(temperature_t_s_zed_kx_ky)**2+np.abs(upar_t_s_zed_kx_ky)**2) * dl_over_B_avg[None,None,:,None,None], axis=(1,2))

    elif run.code == "GX":
        #if get_kx:
        #    W_vs_kx = run.ncdata['Spectra']['Wkxst'][:,0,:]
        #    W_vs_kxky = W_vs_kx[:,:,None]
        #else:
        #    W_vs_ky = run.ncdata['Spectra']['Wkyst'][:,0,:]
        #    W_vs_kxky = W_vs_ky[:,None,:]

        time = run.get_time_array()
        time_idx_min = run.get_time_idx(time_min)
        time_idx_max = run.get_time_idx(time_max)
        time = time[time_idx_min:time_idx_max:time_idx_skip]
        if run.GX_old_version:
            W_vs_kxky = np.transpose(run.ncdata['Spectra']['Wkxkyst'][time_idx_min:time_idx_max:time_idx_skip,0,:,:], axes=(0,2,1))
        else:
            W_vs_kxky = np.transpose(run.ncdata['Diagnostics']['Wg_kxkyst'][time_idx_min:time_idx_max:time_idx_skip,0,:,:], axes=(0,2,1))

    # Check shape
    #assert len(time) == np.shape(W_vs_kxky)[0]
    #assert len(kx)   == np.shape(W_vs_kxky)[1]
    #assert len(ky)   == np.shape(W_vs_kxky)[2]

    # Make (0,0) mode NaN
#        for i_t in range(len(time)):
#            phi2_vs_kxky[i_t, 0, 0] = np.nan

    return W_vs_kxky, time, kx, ky


def read_phi_zonal_spectra(run):

    # phi_vs_t(t, tube, zed, theta0, ky, ri)
    phiZF_vs_t  = run.ncdata.variables['phi_vs_t'][:,0,:,:,0,:] #(t, zed, kx, ri)
    time        = run.ncdata.variables['t'] 
    kx          = run.ncdata.variables['kx'][:] 
    gds22       = run.ncdata.variables['gds22'][:,0] # |nabla(x)|^2
    nablax      = np.sqrt(gds22)

    # Zonal flow derivatives
    phiZF_prime_vs_t  = 1j*kx[None,None,:,None]*nablax[None,:,None,None]*phiZF_vs_t
    phiZF_dprime_vs_t = - (kx[None,None,:,None]*nablax[None,:,None,None])**2 *phiZF_vs_t

    # Tube average and absolute value
    dl_over_B_avg = run.dl_over_B_avg()
    phiZF_vs_t        = np.sum( dl_over_B_avg[None,:,None] * np.abs(phiZF_vs_t[:,:,:,0]       +1j*phiZF_vs_t[:,:,:,1])       , axis=1)
    phiZF_prime_vs_t  = np.sum( dl_over_B_avg[None,:,None] * np.abs(phiZF_prime_vs_t[:,:,:,0] +1j*phiZF_prime_vs_t[:,:,:,1]) , axis=1)
    phiZF_dprime_vs_t = np.sum( dl_over_B_avg[None,:,None] * np.abs(phiZF_dprime_vs_t[:,:,:,0]+1j*phiZF_dprime_vs_t[:,:,:,1]), axis=1)

    return phiZF_vs_t, phiZF_prime_vs_t, phiZF_dprime_vs_t, time, kx


def read_phi2_vs_t_zed(run, tube=0, remove_zonal=False, only_zonal=False, kx_zonal=True, time_min=0, time_max=1e6):

    time = run.get_time_array()
    if time_min < 0:
        time_min = time[-1] - np.abs(time_min)
    time_idx_min = run.get_time_idx(time_min)
    time_idx_max = run.get_time_idx(time_max)
    time = time[time_idx_min:time_idx_max]
    # phi_vs_t(t, tube, zed, theta0, ky, ri) (ri=real,imaginary)
    phi_vs_t = run.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max,0,:,:,:]
    zed  = run.ncdata.variables['zed']
    kx   = run.ncdata.variables['kx'][:]

    if remove_zonal:
        phi_vs_t[:,:,:,0,:] = 0
    if only_zonal:
        phi_vs_t[:,:,:,1:,:] = 0
        if kx_zonal:
            phi_vs_t = phi_vs_t*kx[None,None,:,None,None]

    phi2 = phi_vs_t[:,:,:, :,0]**2 + phi_vs_t[:,:,:, :,1]**2

    phi2_vs_t_zed = np.sum(phi2, axis=(2,3))

    return phi2_vs_t_zed, time, zed


def read_phi2_vs_t(run, tube=0):

    # phi2(t)
    phi2_t = run.ncdata.variables['phi2'][:]
    time = run.ncdata.variables['t'][:]

    return time, phi2_t


def get_fluxes_over_time(run, species_idx=0, norm=True, configuration=None, delta_t=None, load_from_nc=False):

    time = run.get_time_array()
    if run.code == "stella":
        if load_from_nc:
            pflx = run.ncdata['pflux_vs_s'][:,species_idx]
            vflx = run.ncdata['vflux_vs_s'][:,species_idx]
            qflx = run.ncdata['qflux_vs_s'][:,species_idx]

        else:
            fluxes   = np.loadtxt(run.fluxes_file)
            nspecies = get_nspecies(run.ncdata)

            print(nspecies)
            # fluxes is in format [ #time pflx*ns vflx*ns qflx*ns ]
            time = fluxes[:,0]
            pflx = fluxes[:,1           +species_idx]
            vflx = fluxes[:,1+  nspecies+species_idx]
            qflx = fluxes[:,1+2*nspecies+species_idx]
    #np.nan_to_num(vflx)

    elif run.code == "GS2":
        pflx = run.ncdata['es_part_flux'][:,species_idx]
        vflx = run.ncdata['es_mom_flux'][:,species_idx]
        qflx = run.ncdata['es_heat_flux'][:,species_idx]
        norm = False

    elif run.code == "GX":
        if delta_t is not None:
            time_idx_min = nearest_index(time-(time[-1]-delta_t))
        else:
            time_idx_min = 0

        time = time[time_idx_min:]# * 2**(1/2)

        pflx = 0 #run.ncdata['Fluxes']['pflux'][:,species_idx]
        vflx = 0
        if run.GX_old_version:
            qflx = run.ncdata['Fluxes']['qflux'][time_idx_min:,species_idx] / (2**(3/2))
        else:
            qflx = run.ncdata['Diagnostics']['HeatFlux_st'][time_idx_min:,species_idx] / (2**(3/2))
        norm = False

    if norm:
        flux_norm = run.flux_norm()
        if configuration is not None:
            flux_norm = flux_norm / get_true_flux_norm(configuration)

        pflx = pflx/flux_norm
        vflx = vflx/flux_norm
        qflx = qflx/flux_norm

    return pflx, vflx, qflx, time


def get_energies_over_time(run, species_idx=0):

    if run.code == "stella":
        print("To be implemented.")
 

    elif run.code == "GS2":
        time = run.get_time_array()

        delfs2 = run.ncdata['heating_energy_delfs2'][:,species_idx]
        hs2    = run.ncdata['heating_energy_hs2'][:,species_idx]
        phis2  = run.ncdata['heating_energy_phis2'][:,species_idx]

    return delfs2, hs2, phis2, time


def get_moments2_over_time(run, species_idx=0, remove_zonal=True):

    if run.code == "stella":
        print("To be implemented.")

    elif run.code == "GS2":
        time = run.get_time_array()

        # Use moments
        if remove_zonal:
            phi2_by_ky    = run.ncdata['phi2_by_ky'][:,:]
            dens2_by_ky   = run.ncdata['density2_by_ky'][:,species_idx,:]
            upar2_by_ky   = run.ncdata['upar2_by_ky'][:   ,species_idx,:]
            tpar2_by_ky   = run.ncdata['tpar2_by_ky'][:   ,species_idx,:]
            tperp2_by_ky  = run.ncdata['tperp2_by_ky'][:  ,species_idx,:]

            phi2_by_ky[:,0]   = 0
            dens2_by_ky[:,0]  = 0
            upar2_by_ky[:,0]  = 0
            tpar2_by_ky[:,0]  = 0
            tperp2_by_ky[:,0] = 0

            phi2   = np.sum(phi2_by_ky,   axis=1)
            dens2  = np.sum(dens2_by_ky,  axis=1)
            upar2  = np.sum(upar2_by_ky,  axis=1)
            tpar2  = np.sum(tpar2_by_ky,  axis=1)
            tperp2 = np.sum(tperp2_by_ky, axis=1)
        else:
            phi2  = run.ncdata['phi2'][:]
            dens2  = run.ncdata['ntot2'][:,species_idx]
            upar2  = run.ncdata['upar2'][:,species_idx]
            tpar2  = run.ncdata['tpar2'][:,species_idx]
            tperp2 = run.ncdata['tperp2'][:,species_idx]

    return phi2, dens2, upar2, tpar2, tperp2, time


def get_true_flux_norm(configuration):

    if configuration == "precise QA":
        # from /scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2022-04-04_ITG_scan_precise_QA/configuration/evaluate_true_flux_surface_averages/precise_QA
        return 1.2911204260996358e+00

    elif configuration == "precise QH":
        # from /scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2022-04-04_ITG_scan_precise_QA/configuration/evaluate_true_flux_surface_averages/precise_QH
        return 1.3731390294068104e+00
