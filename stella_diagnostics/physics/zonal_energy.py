"""Zonal-flow energetics: Reynolds-stress spectra, zonal energy budget/time-derivative contributions, and related shearing-rate diagnostics."""

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
from stella_diagnostics.spectral.fft import get_fft_k
from stella_diagnostics.grid import nearest_index


def get_zonal_shearing_kx(run, time_min=0, time_max=1e5):

    time_idx_min = run.get_time_idx(time_min)
    time_idx_max = run.get_time_idx(time_max)
    time_all = run.get_time_array()
    time = time_all[time_idx_min:time_idx_max-1]
    dt = time_all[time_idx_min+1:time_idx_max]-time_all[time_idx_min:time_idx_max-1]
    kx, _, _ = run.get_kx_ky_zed()
    Gamma0 = specialfunc.iv(0, kx**2/2) * np.exp(-kx**2/2)

    if run.code == "stella":
        # phi_vs_t(t, tube, zed, theta0, ky, ri)
        #phiZ_t_kx_ri = run.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max-1,0,0,:,0,:]
        phiZ_t_kx = np.sqrt(run.ncdata.variables['phi2_vs_kxky'][time_idx_min:time_idx_max-1,:,0])
        phiZ_t_kx_ri = np.zeros( (np.shape(phiZ_t_kx)[0], np.shape(phiZ_t_kx)[1], 2) )
        phiZ_t_kx_ri[:,:,0] = phiZ_t_kx            

    elif run.code == "GX":
        if run.GX_old_version:
            # phi_vs_t(t, tube, zed, theta0, ky, ri)
            #phiZ_t_kx_ri = run.ncdata['Special']['Phi_z'][0,:,0,:]
            #phiZ_t_kx_ri = phiZ_t_kx_ri[None,:,:]

            phiZ_t_kx = np.sqrt(run.ncdata['Spectra']['Akxkyst'][time_idx_min:time_idx_max-1,0,:]/(1-Gamma0)[None,:])

        else:
            phiZ_t_kx = np.sqrt( run.ncdata['Diagnostics']['Wphi_kxkyst'][time_idx_min:time_idx_max-1,0,0,:] / (1-Gamma0)[None,:] )

        phiZ_t_kx_ri = np.zeros( (np.shape(phiZ_t_kx)[0], np.shape(phiZ_t_kx)[1], 2) )
        phiZ_t_kx_ri[:,:,0] = phiZ_t_kx*np.sqrt(2) #/(2*np.pi) # undo theta avg

    dx2phiZ_t_kx = -(1-Gamma0) * (phiZ_t_kx_ri[:,:,0]+1j*phiZ_t_kx_ri[:,:,1])

    dx2phiZ_stationary_kx_C = np.sum(dx2phiZ_t_kx*dt[:,None], axis=0)/np.sum(dt)

    gammaE2_stationary_kx = np.abs(dx2phiZ_stationary_kx_C)**2

    gammaE2_timevar_kx = np.sum(np.abs(dx2phiZ_t_kx-dx2phiZ_stationary_kx_C[None,:])**2 *dt[:,None], axis=0)/np.sum(dt)

    gammaE2_tot_kx = np.sum(np.abs(dx2phiZ_t_kx)**2 *dt[:,None], axis=0)/np.sum(dt)

    return gammaE2_tot_kx[kx>0], gammaE2_stationary_kx[kx>0], gammaE2_timevar_kx[kx>0], kx[kx>0]


def get_dt_par_mom_pressure_transport(run, time_min=0, time_max=1e10, time_idx_skip=1, nx=None, ny=None, kxmin_filter=np.inf, kymin_filter=np.inf, kxmax_filter=-1, kymax_filter=-1):

    time = run.get_time_array()
    time_max = min(time[-1], time_max)
    if time_min < 0:
        time_min = time_max - np.abs(time_min)
    time_idx_min = nearest_index(time-time_min)
    time_idx_max = nearest_index(time-time_max)
    time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)

    dE_par_mom_tr  = np.zeros(len(time_idx_eval))
    dE_meanP_tr    = np.zeros(len(time_idx_eval))
    dE_deltP_tr    = np.zeros(len(time_idx_eval))

    for i_time_idx, time_idx in enumerate(time_idx_eval):
        print("Evaluating par mom transport: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")

        x, dE_par_mom_tr_x, dE_meanP_tr_x, dE_deltP_tr_x = run.get_dt_par_mom_pressure_transport_x(time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
        dx = x[1]-x[0]
        dE_par_mom_tr[i_time_idx] = np.sum(dE_par_mom_tr_x)*dx
        dE_meanP_tr[i_time_idx]   = np.sum(dE_meanP_tr_x)*dx
        dE_deltP_tr[i_time_idx]   = np.sum(dE_deltP_tr_x)*dx

    return time[time_idx_eval], dE_par_mom_tr, dE_meanP_tr, dE_deltP_tr


def get_dt_par_mom_pressure_transport_x(run, time_idx=-1, nx=None, ny=None, kxmin_filter=np.inf, kymin_filter=np.inf, kxmax_filter=-1, kymax_filter=-1):

    uparZ_zed_x_y, zed, x, y, _  = run.get_quantity_zed_x_y("upar",           time_idx=time_idx, kx_order=0, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True, nx=nx, ny=ny)
    par_mom_transp_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y("par_mom_transport", time_idx=time_idx, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, nx=nx, ny=ny)

    presZ_zed_x_y, zed, x, y, _  = run.get_quantity_zed_x_y("pressure",       time_idx=time_idx, kx_order=0, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True, nx=nx, ny=ny)
    densZ_zed_x_y, zed, x, y, _  = run.get_quantity_zed_x_y("density",        time_idx=time_idx, kx_order=0, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True, nx=nx, ny=ny)
    tempZ_zed_x_y = presZ_zed_x_y-densZ_zed_x_y
    pressure_transp_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y("pressure_transport", time_idx=time_idx, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, nx=nx, ny=ny)

    dl_over_B_avg = run.dl_over_B_avg()
    mean_tempZ_x_y = np.sum(dl_over_B_avg[:,None,None]*tempZ_zed_x_y)
    mean_tempZ_zed_x_y = np.zeros_like(tempZ_zed_x_y)
    for i_zed in range(len(zed)):
        mean_tempZ_zed_x_y[i_zed] = mean_tempZ_x_y
    delt_tempZ_zed_x_y = tempZ_zed_x_y - mean_tempZ_zed_x_y

    dy = y[1]-y[0]
    dE_par_mom_tr_x  = np.sum(par_mom_transp_zed_x_y*uparZ_zed_x_y*dl_over_B_avg[:,None,None], axis=(0,2))*dy  * 2
    dE_mean_pressure_tr_x = np.sum(pressure_transp_zed_x_y*mean_tempZ_zed_x_y*dl_over_B_avg[:,None,None], axis=(0,2))*dy * 4/3#4/7 *2
    dE_delt_pressure_tr_x = np.sum(pressure_transp_zed_x_y*delt_tempZ_zed_x_y*dl_over_B_avg[:,None,None], axis=(0,2))*dy * 4/3#4/7 *2
    return x, dE_par_mom_tr_x, dE_mean_pressure_tr_x, dE_delt_pressure_tr_x


def get_dt_zonal_energy_contributions(run, time_min=0, time_max=1e10, time_idx_skip=1, nx=None, ny=None, kxmin_filter=np.inf, kymin_filter=np.inf, kxmax_filter=-1, kymax_filter=-1, separate_Reynolds=True):

    time = run.get_time_array()
    time_max = min(time[-1], time_max)
    if time_min < 0:
        time_min = time_max - np.abs(time_min)
    time_idx_min = nearest_index(time-time_min)
    time_idx_max = nearest_index(time-time_max)
    time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)

    EZ_t                        = np.zeros(len(time_idx_eval))
    EZ_t_deltaphi2              = np.zeros(len(time_idx_eval))
    dEZ_reynolds_phi_nablax2_t  = np.zeros(len(time_idx_eval))
    dEZ_reynolds_Pprp_nablax2_t = np.zeros(len(time_idx_eval))
    dEZ_reynolds_phi_nablaxy_t  = np.zeros(len(time_idx_eval))
    dEZ_reynolds_Pprp_nablaxy_t = np.zeros(len(time_idx_eval))
    dEZ_vDx_t                   = np.zeros(len(time_idx_eval))
    dEZ_upar_t                  = np.zeros(len(time_idx_eval))

    for i_time_idx, time_idx in enumerate(time_idx_eval):
        print("Evaluating zonal energy contributions: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")

        # Get energies
        x, EZ_x, EZ_deltaphi2_x, dEZ_reynolds_phi_nablax2_x, dEZ_reynolds_Pprp_nablax2_x, dEZ_reynolds_phi_nablaxy_x, dEZ_reynolds_Pprp_nablaxy_x, dEZ_vDx_x, dEZ_upar_x = run.get_dt_zonal_energy_contributions_x(time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
        dx = x[1]-x[0]

        EZ_t[i_time_idx]                        = np.sum(EZ_x               )*dx
        EZ_t_deltaphi2[i_time_idx]              = np.sum(EZ_deltaphi2_x     )*dx
        dEZ_reynolds_phi_nablax2_t[i_time_idx]  = np.sum(dEZ_reynolds_phi_nablax2_x )*dx
        dEZ_reynolds_Pprp_nablax2_t[i_time_idx] = np.sum(dEZ_reynolds_Pprp_nablax2_x)*dx
        dEZ_reynolds_phi_nablaxy_t[i_time_idx]  = np.sum(dEZ_reynolds_phi_nablaxy_x )*dx
        dEZ_reynolds_Pprp_nablaxy_t[i_time_idx] = np.sum(dEZ_reynolds_Pprp_nablaxy_x)*dx
        dEZ_vDx_t[   i_time_idx]                = np.sum(dEZ_vDx_x          )*dx
        dEZ_upar_t[    i_time_idx]              = np.sum(dEZ_upar_x         )*dx

    return time[time_idx_eval], EZ_t, dEZ_reynolds_phi_nablax2_t, dEZ_reynolds_Pprp_nablax2_t, dEZ_reynolds_phi_nablaxy_t, dEZ_reynolds_Pprp_nablaxy_t, dEZ_vDx_t, dEZ_upar_t, EZ_t_deltaphi2


def get_Reynolds_NZ_spectrum(run, time_min=0, time_max=99999, time_idx_skip=1):

    time = run.get_time_array()
    time_max = min(time[-1], time_max)
    if time_min < 0:
        time_min = time_max - np.abs(time_min)
    time_idx_min = nearest_index(time-time_min)
    time_idx_max = nearest_index(time-time_max)
    time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)
    dt = np.gradient(time)

    kx = run.ncdata['kx'][:]
    ky = run.ncdata['ky'][:]

    dEZ_dt_reynolds_phi_kx_ky  = np.zeros((len(kx),len(ky)), dtype='complex')
    dEZ_dt_reynolds_Pprp_kx_ky = np.zeros((len(kx),len(ky)), dtype='complex')

    # Geometric coefficients
    dl_over_B_avg = run.dl_over_B_avg()
    _, _, _, gds21, gds22, bmag = run.get_FLR()

    # Time-average
    for i_time_idx, time_idx in enumerate(time_idx_eval):
        print("Evaluating reynolds contributions: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")
    
        # Load phi and Pprp
        phi_zed_kx_ky,  zed, kx, ky, _  = run.get_quantity_zed_kx_ky("phi",           time_idx=time_idx)
        Pprp_zed_kx_ky,   _,  _,  _, _  = run.get_quantity_zed_kx_ky("pressure_perp", time_idx=time_idx)
        phiZ_zed_kx = phi_zed_kx_ky[:,:,0]

        nablaxnablaphi_zed_kx_ky = 1j* (ky[None,None,:]*phi_zed_kx_ky*(gds21/bmag**2)[:,None,None] + kx[None,:,None]*phi_zed_kx_ky*(gds22/bmag**2)[:,None,None])

        # Inner loop over kx of phi_Z that contribute to the energy exchange
        for i_kx in range(len(kx)):
            mult_fac_zed_kx_ky = 0.5*dl_over_B_avg[:,None,None] * ((1j*kx[i_kx])**2 * np.conj(phiZ_zed_kx[:,i_kx]))[:,None,None] * nablaxnablaphi_zed_kx_ky

            delta_kx_vals = (np.arange(len(kx)) - i_kx)%(len(kx))
            dEZ_dt_reynolds_phi_kx_ky  += np.sum(mult_fac_zed_kx_ky * np.conj( 1j*ky[None,None,:] *phi_zed_kx_ky[:,delta_kx_vals]) , axis=0) * dt[time_idx]
            dEZ_dt_reynolds_Pprp_kx_ky += np.sum(mult_fac_zed_kx_ky * np.conj( 1j*ky[None,None,:]*Pprp_zed_kx_ky[:,delta_kx_vals]) , axis=0) * dt[time_idx]

    # Correct time-normalisation to get average
    dEZ_dt_reynolds_phi_kx_ky  = dEZ_dt_reynolds_phi_kx_ky  / np.sum(dt)
    dEZ_dt_reynolds_Pprp_kx_ky = dEZ_dt_reynolds_Pprp_kx_ky / np.sum(dt)
    
    return kx, ky, dEZ_dt_reynolds_phi_kx_ky, dEZ_dt_reynolds_Pprp_kx_ky


def get_Reynolds_kz_kxNZ_spectrum(run, time_min=0, time_max=99999, time_idx_skip=1):

    time = run.get_time_array()
    time_max = min(time[-1], time_max)
    if time_min < 0:
        time_min = time_max - np.abs(time_min)
    time_idx_min = nearest_index(time-time_min)
    time_idx_max = nearest_index(time-time_max)
    time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)
    dt = np.gradient(time)

    kx = run.ncdata['kx'][:]
    ky = run.ncdata['ky'][:]

    dEZ_dt_reynolds_phi_kz_kx  = np.zeros((len(kx),len(kx)), dtype='complex')
    dEZ_dt_reynolds_Pprp_kz_kx = np.zeros((len(kx),len(kx)), dtype='complex')

    # Geometric coefficients
    dl_over_B_avg = run.dl_over_B_avg()
    _, _, _, gds21, gds22, bmag = run.get_FLR()

    # Time-average
    for i_time_idx, time_idx in enumerate(time_idx_eval):
        print("Evaluating reynolds contributions: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")
    
        # Load phi and Pprp
        phi_zed_kx_ky,  zed, kx, ky, _  = run.get_quantity_zed_kx_ky("phi",           time_idx=time_idx)
        Pprp_zed_kx_ky,   _,  _,  _, _  = run.get_quantity_zed_kx_ky("pressure_perp", time_idx=time_idx)
        phiZ_zed_kz = phi_zed_kx_ky[:,:,0]

        nablaxnablaphi_zed_kx_ky = 1j* (ky[None,None,:]*phi_zed_kx_ky*(gds21/bmag**2)[:,None,None] + kx[None,:,None]*phi_zed_kx_ky*(gds22/bmag**2)[:,None,None])

        # Sum over zed and ky of nonzonal terms that contribute to the energy exchange
        for i_kz in range(len(kx)):
            for i_kx in range(len(kx)):
                delta_i_kx = (i_kz - i_kx)%(len(kx))
                dEZ_dt_reynolds_phi_kz_kx[ i_kz,i_kx] = np.sum(0.5*dl_over_B_avg[:,None]*(1j*kx[i_kz])**2*phiZ_zed_kz[:,i_kz,None]* nablaxnablaphi_zed_kx_ky[:,i_kx,:]*np.conj(1j*ky[None,None,:]* phi_zed_kx_ky[:,delta_i_kx,:])) * dt[time_idx]
                dEZ_dt_reynolds_Pprp_kz_kx[i_kz,i_kx] = np.sum(0.5*dl_over_B_avg[:,None]*(1j*kx[i_kz])**2*phiZ_zed_kz[:,i_kz,None]* nablaxnablaphi_zed_kx_ky[:,i_kx,:]*np.conj(1j*ky[None,None,:]*Pprp_zed_kx_ky[:,delta_i_kx,:])) * dt[time_idx]

    # Correct time-normalisation to get average
    dEZ_dt_reynolds_phi_kz_kx  = dEZ_dt_reynolds_phi_kz_kx  / np.sum(dt)
    dEZ_dt_reynolds_Pprp_kz_kx = dEZ_dt_reynolds_Pprp_kz_kx / np.sum(dt)
    
    return kx, dEZ_dt_reynolds_phi_kz_kx, dEZ_dt_reynolds_Pprp_kz_kx


def get_time_avg_zonal_energy_contributions_kx(run, time_min=0, time_max=1e10, time_idx_skip=1, alt_slow_eval=False, omega_min=None, omega_max=None):

    time = run.get_time_array()
    time_max = min(time[-1], time_max)
    if time_min < 0:
        time_min = time_max - np.abs(time_min)
    time_idx_min = nearest_index(time-time_min)
    time_idx_max = nearest_index(time-time_max)
    time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)
    time_eval = time[time_idx_eval]

    # Geometric quantities we will need
    dl_over_B_avg = run.dl_over_B_avg()
    shat     = run.ncdata.variables['shat'].getValue()
    vdriftx = run.ncdata.variables['gbdrift0'][:,0]/(2*shat)
    _, _, _, _, gds22, bmag = run.get_FLR()
    nablax2 = gds22/bmag**2
    _, _, zed = run.get_kx_ky_zed()
    dl_costheta = run.get_zed_weight("cos", zed)

    for i_time_idx, time_idx in enumerate(time_idx_eval):
        print("Evaluating zonal energy contributions: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")

        # Get energy contributions
        if not alt_slow_eval:
            #if omega_min is not None or omega_max is not None:
            #    print("No implementation yet of omega filter for alt_slow_eval=False. Returning.")
            #    return

            # Evaluate everything directly in k-space
            phi_zed_kx_ky, zed, kx, ky, _ = run.get_quantity_zed_kx_ky(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx)
            dxphi_zed_kx_ky, zed, kx, ky, _ = run.get_quantity_zed_kx_ky(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1)
            reynolds_phi_nablax2_zed_kx_ky, _, _, _, _  = run.get_quantity_zed_kx_ky("Reynolds_phi_nablax2",  time_idx=time_idx, kx_order=2)
            reynolds_Pprp_nablax2_zed_kx_ky, _, _, _, _ = run.get_quantity_zed_kx_ky("Reynolds_Pprp_nablax2", time_idx=time_idx, kx_order=2)
            reynolds_phi_nablaxy_zed_kx_ky, _, _, _, _  = run.get_quantity_zed_kx_ky("Reynolds_phi_nablaxy",  time_idx=time_idx, kx_order=2)
            reynolds_Pprp_nablaxy_zed_kx_ky, _, _, _, _ = run.get_quantity_zed_kx_ky("Reynolds_Pprp_nablaxy", time_idx=time_idx, kx_order=2)
            upar_zed_kx_ky, _, _, _, _ = run.get_quantity_zed_kx_ky(quantity="upar", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=0)
            dxP_zed_kx_ky, _, _, _, _ = run.get_quantity_zed_kx_ky(quantity="pressure", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1)
            pressure_tr_zed_kx_ky, _, _, _, _ = run.get_quantity_zed_kx_ky(quantity="pressure_transport", time_idx=time_idx, kx_order=1)
            par_mom_tr_zed_kx_ky, _, _, _, _ = run.get_quantity_zed_kx_ky(quantity="par_mom_transport", time_idx=time_idx, kx_order=1)
            dxupar_zed_kx_ky, _, _, _, _ = run.get_quantity_zed_kx_ky(quantity="upar", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1)

            reynolds_phi_nablax2_zed_kx  = reynolds_phi_nablax2_zed_kx_ky[:,:,0]
            reynolds_Pprp_nablax2_zed_kx = reynolds_Pprp_nablax2_zed_kx_ky[:,:,0]
            reynolds_phi_nablaxy_zed_kx  = reynolds_phi_nablaxy_zed_kx_ky[:,:,0]
            reynolds_Pprp_nablaxy_zed_kx = reynolds_Pprp_nablaxy_zed_kx_ky[:,:,0]
            phi_zed_kx                   = phi_zed_kx_ky[ :,kx>=0,0]
            dxphi_zed_kx                 = dxphi_zed_kx_ky[ :,kx>=0,0]
            upar_zed_kx                  = upar_zed_kx_ky[:,kx>=0,0]
            dxP_zed_kx                   = dxP_zed_kx_ky[ :,kx>=0,0]
            pressure_tr_zed_kx           = pressure_tr_zed_kx_ky[ :,:,0]
            par_mom_tr_zed_kx            = par_mom_tr_zed_kx_ky[ :,:,0]
            dxupar_zed_kx                = dxupar_zed_kx_ky[:,kx>=0,0]
            kx = kx[kx>=0]

            # Obtain parallel derivative of upar term
            dupar_zed_kx = np.zeros_like(upar_zed_kx)
            uparB_zed_kx = upar_zed_kx / bmag[:,None]
            gradpar  = run.ncdata.variables['gradpar'][:]
            dzed = zed[1]-zed[0]
            for i_zed in range(len(zed)-1):
                if i_zed == 0:
                    dupar_zed_kx[0] = (uparB_zed_kx[1]-uparB_zed_kx[-1]) / dzed
                else:
                    dupar_zed_kx[i_zed] = 0.5*(uparB_zed_kx[i_zed+1]-uparB_zed_kx[i_zed-1]) / dzed
            dupar_zed_kx[-1] = (uparB_zed_kx[0]-uparB_zed_kx[-2]) / dzed
            dupar_zed_kx = dupar_zed_kx * (gradpar*bmag)[:,None]

        else:
            # Evaluate everything in real space and then transform back to k-space
            reynolds_phi_nablax2_zed_x_y,  zed, x, y, _ = run.get_quantity_zed_x_y("Reynolds_phi_nablax2",  time_idx=time_idx, kx_order=2)
            reynolds_Pprp_nablax2_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y("Reynolds_Pprp_nablax2", time_idx=time_idx, kx_order=2)
            reynolds_phi_nablaxy_zed_x_y,  zed, x, y, _ = run.get_quantity_zed_x_y("Reynolds_phi_nablaxy",  time_idx=time_idx, kx_order=2)
            reynolds_Pprp_nablaxy_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y("Reynolds_Pprp_nablaxy", time_idx=time_idx, kx_order=2)
            dxP_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="pressure", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1)
            phi_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx)
            dxphi_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1)
            upar_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="upar", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=0) 
            dxupar_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="upar", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1) 
            pressure_tr_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="pressure_transport", time_idx=time_idx)
            par_mom_tr_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="par_mom_transport", time_idx=time_idx)
    
            # Obtain parallel derivative of upar term
            dupar_zed_x_y = np.zeros_like(upar_zed_x_y)
            uparB_zed_x_y = upar_zed_x_y / bmag[:,None,None]
            gradpar  = run.ncdata.variables['gradpar'][:]
            dzed = zed[1]-zed[0]
            for i_zed in range(len(zed)-1):
                if i_zed == 0:
                    dupar_zed_x_y[0] = (uparB_zed_x_y[1]-uparB_zed_x_y[-1]) / dzed
                else:
                    dupar_zed_x_y[i_zed] = 0.5*(uparB_zed_x_y[i_zed+1]-uparB_zed_x_y[i_zed-1]) / dzed
            dupar_zed_x_y[-1] = (uparB_zed_x_y[0]-uparB_zed_x_y[-2]) / dzed
            dupar_zed_x_y = dupar_zed_x_y * (gradpar*bmag)[:,None,None]
    
            # Take y-averages
            dy = y[1]-y[0]
            reynolds_phi_nablax2_zed_x  = np.sum(reynolds_phi_nablax2_zed_x_y,  axis=2)*dy
            reynolds_Pprp_nablax2_zed_x = np.sum(reynolds_Pprp_nablax2_zed_x_y, axis=2)*dy
            reynolds_phi_nablaxy_zed_x  = np.sum(reynolds_phi_nablaxy_zed_x_y,  axis=2)*dy
            reynolds_Pprp_nablaxy_zed_x = np.sum(reynolds_Pprp_nablaxy_zed_x_y, axis=2)*dy
            dxP_zed_x                   = np.sum(dxP_zed_x_y,                   axis=2)*dy
            phi_zed_x                   = np.sum(phi_zed_x_y,                   axis=2)*dy
            dxphi_zed_x                 = np.sum(dxphi_zed_x_y,                 axis=2)*dy
            dupar_zed_x                 = np.sum(dupar_zed_x_y,                 axis=2)*dy
            dxupar_zed_x                = np.sum(dxupar_zed_x_y,                axis=2)*dy
            pressure_tr_zed_x           = np.sum(pressure_tr_zed_x_y,           axis=2)*dy
            par_mom_tr_zed_x            = np.sum(par_mom_tr_zed_x_y,            axis=2)*dy

            # FFT for each zed value
            for i_zed in range(len(zed)):
                reynolds_phi_nablax2_kx, kx  = get_fft_k(reynolds_phi_nablax2_zed_x[i_zed],    x)
                reynolds_Pprp_nablax2_kx, kx = get_fft_k(reynolds_Pprp_nablax2_zed_x[i_zed],   x)
                reynolds_phi_nablaxy_kx, kx  = get_fft_k(reynolds_phi_nablaxy_zed_x[i_zed],    x)
                reynolds_Pprp_nablaxy_kx, kx = get_fft_k(reynolds_Pprp_nablaxy_zed_x[i_zed],   x)
                dxP_kx,             _        = get_fft_k(dxP_zed_x[i_zed]            , x)
                phi_kx,             _        = get_fft_k(phi_zed_x[i_zed]            , x)
                dxphi_kx,           _        = get_fft_k(dxphi_zed_x[i_zed]          , x)
                dupar_kx,           _        = get_fft_k(dupar_zed_x[i_zed]          , x)
                dxupar_kx,          _        = get_fft_k(dxupar_zed_x[i_zed]         , x)
                pressure_tr_kx,     _        = get_fft_k(pressure_tr_zed_x[i_zed]    , x)
                par_mom_tr_kx,      _        = get_fft_k(par_mom_tr_zed_x[i_zed]     , x)

                if i_zed == 0:
                    reynolds_phi_nablax2_zed_kx  = np.zeros((len(zed),len(kx)), dtype='complex')
                    reynolds_Pprp_nablax2_zed_kx = np.zeros((len(zed),len(kx)), dtype='complex')
                    reynolds_phi_nablaxy_zed_kx  = np.zeros((len(zed),len(kx)), dtype='complex')
                    reynolds_Pprp_nablaxy_zed_kx = np.zeros((len(zed),len(kx)), dtype='complex')
                    dxP_zed_kx                   = np.zeros((len(zed),len(kx)), dtype='complex')
                    phi_zed_kx                   = np.zeros((len(zed),len(kx)), dtype='complex')
                    dxphi_zed_kx                 = np.zeros((len(zed),len(kx)), dtype='complex')
                    dupar_zed_kx                 = np.zeros((len(zed),len(kx)), dtype='complex')
                    dxupar_zed_kx                = np.zeros((len(zed),len(kx)), dtype='complex')
                    pressure_tr_zed_kx           = np.zeros((len(zed),len(kx)), dtype='complex')
                    par_mom_tr_zed_kx            = np.zeros((len(zed),len(kx)), dtype='complex')
                
                reynolds_phi_nablax2_zed_kx[i_zed]  = reynolds_phi_nablax2_kx
                reynolds_Pprp_nablax2_zed_kx[i_zed] = reynolds_Pprp_nablax2_kx
                reynolds_phi_nablaxy_zed_kx[i_zed]  = reynolds_phi_nablaxy_kx
                reynolds_Pprp_nablaxy_zed_kx[i_zed] = reynolds_Pprp_nablaxy_kx
                dxP_zed_kx[i_zed]                   = dxP_kx
                phi_zed_kx[i_zed]                   = phi_kx
                dxphi_zed_kx[i_zed]                 = dxphi_kx
                dupar_zed_kx[i_zed]                 = dupar_kx
                dxupar_zed_kx[i_zed]                = dxupar_kx
                pressure_tr_zed_kx[i_zed]           = pressure_tr_kx
                par_mom_tr_zed_kx[i_zed]            = par_mom_tr_kx

        if i_time_idx == 0:
            EZ_t_kx                        = np.zeros((len(time_idx_eval),len(kx)))
            dEZ_reynolds_phi_nablax2_t_kx  = np.zeros((len(time_idx_eval),len(kx)))
            dEZ_reynolds_Pprp_nablax2_t_kx = np.zeros((len(time_idx_eval),len(kx)))
            dEZ_reynolds_phi_nablaxy_t_kx  = np.zeros((len(time_idx_eval),len(kx)))
            dEZ_reynolds_Pprp_nablaxy_t_kx = np.zeros((len(time_idx_eval),len(kx)))
            dEZ_vDx_P_t_kx                 = np.zeros((len(time_idx_eval),len(kx)))
            dEZ_upar_t_kx                  = np.zeros((len(time_idx_eval),len(kx)))
            dE_mean_pressure_tr_t_kx       = np.zeros((len(time_idx_eval),len(kx)))
            dE_delt_pressure_tr_t_kx       = np.zeros((len(time_idx_eval),len(kx)))
            dE_par_mom_tr_t_kx             = np.zeros((len(time_idx_eval),len(kx)))
            du_par_mom_tr_t_kx             = np.zeros((len(time_idx_eval),len(kx)))
            du_cos_par_mom_tr_t_kx         = np.zeros((len(time_idx_eval),len(kx)))

        for i_kx in range(len(kx)):

            mean_dxP_zed = np.sum(dl_over_B_avg*dxP_zed_kx[:,i_kx])

            EZ_t_kx[i_time_idx, i_kx]                        =  np.sum(dl_over_B_avg * 2*np.abs(dxphi_zed_kx[:,i_kx])**2 * nablax2/2 )
            dEZ_reynolds_phi_nablax2_t_kx[i_time_idx, i_kx]  =  np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(reynolds_phi_nablax2_zed_kx[:,i_kx])) )
            dEZ_reynolds_Pprp_nablax2_t_kx[i_time_idx, i_kx] =  np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(reynolds_Pprp_nablax2_zed_kx[:,i_kx])) )
            dEZ_reynolds_phi_nablaxy_t_kx[i_time_idx, i_kx]  =  np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(reynolds_phi_nablaxy_zed_kx[:,i_kx])) )
            dEZ_reynolds_Pprp_nablaxy_t_kx[i_time_idx, i_kx] =  np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(reynolds_Pprp_nablaxy_zed_kx[:,i_kx])) )
            dEZ_vDx_P_t_kx[i_time_idx, i_kx]                 = -np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(dxP_zed_kx[:,i_kx])) * vdriftx) *2
            dEZ_upar_t_kx[i_time_idx, i_kx]                  = -np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(dupar_zed_kx[:,i_kx]))) *2
            dE_mean_pressure_tr_t_kx[i_time_idx, i_kx]       =  np.sum(dl_over_B_avg * 2*np.real(pressure_tr_zed_kx[:,i_kx]*np.conj(mean_dxP_zed))) *2 * 4/7 # ~ (dP/dt)_{NL}
            dE_delt_pressure_tr_t_kx[i_time_idx, i_kx]       =  np.sum(dl_over_B_avg * 2*np.real(pressure_tr_zed_kx[:,i_kx]*np.conj((dxP_zed_kx[:,i_kx]-mean_dxP_zed)))) *2 * 4/7 # ~ (dP/dt)_{NL}
            dE_par_mom_tr_t_kx[i_time_idx, i_kx]             =  np.sum(dl_over_B_avg * 2*np.real(dxupar_zed_kx[:,i_kx]     *np.conj(par_mom_tr_zed_kx[:,i_kx]))) *2 # = (dU/dt)_{NL}
            du_par_mom_tr_t_kx[i_time_idx, i_kx]             =  np.sum(dl_over_B_avg * 2*np.real(dxphi_zed_kx[:,i_kx]     *np.conj(par_mom_tr_zed_kx[:,i_kx]))) *2 # = vE*(dU/dt)_{NL}
            du_cos_par_mom_tr_t_kx[i_time_idx, i_kx]         =  np.sum(dl_costheta   * 2*np.real(dxphi_zed_kx[:,i_kx]     *np.conj(par_mom_tr_zed_kx[:,i_kx]))) *2 # = vE*(dU/dt)_{NL}*cos(theta)

    # Time-average (note dt may vary over time)
    dt = np.gradient(time_eval)
    EZ_kx            = np.sum(EZ_t_kx*dt[:,None],           axis=0)/(time_eval[-1]-time_eval[0])
    dEZ_reynolds_phi_nablax2_kx  = np.sum(dEZ_reynolds_phi_nablax2_t_kx*dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
    dEZ_reynolds_Pprp_nablax2_kx = np.sum(dEZ_reynolds_Pprp_nablax2_t_kx*dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
    dEZ_reynolds_phi_nablaxy_kx  = np.sum(dEZ_reynolds_phi_nablaxy_t_kx*dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
    dEZ_reynolds_Pprp_nablaxy_kx = np.sum(dEZ_reynolds_Pprp_nablaxy_t_kx*dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
    dEZ_vDx_P_kx     = np.sum(dEZ_vDx_P_t_kx   *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
    dEZ_upar_kx      = np.sum(dEZ_upar_t_kx    *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
    dE_mean_pressure_tr_kx = np.sum(dE_mean_pressure_tr_t_kx  *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
    dE_delt_pressure_tr_kx = np.sum(dE_delt_pressure_tr_t_kx  *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
    dE_par_mom_tr_kx = np.sum(dE_par_mom_tr_t_kx  *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
    du_par_mom_tr_kx = np.sum(du_par_mom_tr_t_kx  *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
    du_cos_par_mom_tr_kx = np.sum(du_cos_par_mom_tr_t_kx  *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])

    return kx, EZ_kx, dEZ_reynolds_phi_nablax2_kx, dEZ_reynolds_Pprp_nablax2_kx, dEZ_reynolds_phi_nablaxy_kx, dEZ_reynolds_Pprp_nablaxy_kx, dEZ_vDx_P_kx, dEZ_upar_kx, dE_mean_pressure_tr_kx, dE_delt_pressure_tr_kx, dE_par_mom_tr_kx, du_par_mom_tr_kx, du_cos_par_mom_tr_kx


def get_dt_zonal_energy_contributions_x(run, time_idx=-1, nx=None, ny=None, kxmin_filter=np.inf, kymin_filter=np.inf, kxmax_filter=-1, kymax_filter=-1):

    reynolds_phi_nablax2_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y("Reynolds_phi_nablax2", time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0) # without x-derivative in front, so need to multiply by dx2phiZ to get energy derivative
    reynolds_Pprp_nablax2_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y("Reynolds_Pprp_nablax2", time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0) # without x-derivative in front, so need to multiply by dx2phiZ to get energy derivative
    reynolds_phi_nablaxy_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y("Reynolds_phi_nablaxy", time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0) # without x-derivative in front, so need to multiply by dx2phiZ to get energy derivative
    reynolds_Pprp_nablaxy_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y("Reynolds_Pprp_nablaxy", time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0) # without x-derivative in front, so need to multiply by dx2phiZ to get energy derivative
    dxP_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="pressure", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
    phi_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
    dxphi_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
    dx2phi_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=2, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kymax_filter=kymax_filter)
    upar_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y(quantity="upar", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=0, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kymax_filter=kymax_filter)

    # Obtain deltaphi
    dl_over_B_avg = run.dl_over_B_avg()
    mean_phiZ_x_y = np.sum(phi_zed_x_y*dl_over_B_avg[:,None,None], axis=0)
    delta_phi_zed_x_y = np.zeros_like(phi_zed_x_y)
    for i_zed in range(len(zed)):
        delta_phi_zed_x_y[i_zed] = phi_zed_x_y[i_zed] - mean_phiZ_x_y
    deltaphiZ_2_zed_x_y = delta_phi_zed_x_y**2

    # Obtain parallel derivative of upar term
    _, _, _, _, gds22, bmag = run.get_FLR()
    #dupar_dzed_x_y = np.gradient(upar_zed_x_y/bmag[:,None,None], zed, axis=0) * bmag[:,None,None]
     #   f_zed_x_y = np.gradient(f_zed_x_y*gradpar[:,None,None], zed, axis=0)
    # Use periodicity
    dupar_zed_x_y = np.zeros_like(upar_zed_x_y)
    uparB_zed_x_y = upar_zed_x_y / bmag[:,None,None]
    gradpar  = run.ncdata.variables['gradpar'][:]
    dzed = zed[1]-zed[0]
    for i_zed in range(len(zed)-1):
        if i_zed == 0:
            dupar_zed_x_y[0] = (uparB_zed_x_y[1]-uparB_zed_x_y[-1]) / dzed
        else:
            dupar_zed_x_y[i_zed] = 0.5*(uparB_zed_x_y[i_zed+1]-uparB_zed_x_y[i_zed-1]) / dzed
    dupar_zed_x_y[-1] = (uparB_zed_x_y[0]-uparB_zed_x_y[-2]) / dzed

    dupar_zed_x_y = dupar_zed_x_y * (gradpar*bmag)[:,None,None]
#        print(dupar_zed_x_y[:,0,0])
    #print(np.shape(gradpar))
    #print(np.shape(bmag))
    #print(np.sum(dupar_zed_x_y[:,0,0]*dl_over_B_avg))
    #assert(np.abs(np.sum(dupar_zed_x_y[:,0,0]*dl_over_B_avg)) < 1e-14)

    # Get energies
    dy = y[1]-y[0]
    shat     = run.ncdata.variables['shat'].getValue()
    vdriftx = run.ncdata.variables['gbdrift0'][:,0]/(2*shat)
    nablax2 = gds22/bmag**2

    EZ_x           = np.sum(dl_over_B_avg[:,None,None]*0.5*dxphi_zed_x_y**2 *nablax2[:,None,None], axis=(0,2))*dy
    tau = 1
    EZ_deltaphi2_x =-np.sum(dl_over_B_avg[:,None,None]*tau*deltaphiZ_2_zed_x_y                             , axis=(0,2))*dy
    dEZ_reynolds_phi_nablax2_x  = np.sum(dl_over_B_avg[:,None,None]*dx2phi_zed_x_y*reynolds_phi_nablax2_zed_x_y              , axis=(0,2))*dy
    dEZ_reynolds_Pprp_nablax2_x = np.sum(dl_over_B_avg[:,None,None]*dx2phi_zed_x_y*reynolds_Pprp_nablax2_zed_x_y              , axis=(0,2))*dy
    dEZ_reynolds_phi_nablaxy_x  = np.sum(dl_over_B_avg[:,None,None]*dx2phi_zed_x_y*reynolds_phi_nablaxy_zed_x_y              , axis=(0,2))*dy
    dEZ_reynolds_Pprp_nablaxy_x = np.sum(dl_over_B_avg[:,None,None]*dx2phi_zed_x_y*reynolds_Pprp_nablaxy_zed_x_y              , axis=(0,2))*dy
    dEZ_vDx_x    =-np.sum(dl_over_B_avg[:,None,None]*2* phi_zed_x_y*dxP_zed_x_y  *vdriftx[:,None,None]     , axis=(0,2))*dy
    dEZ_upar_x   =-np.sum(dl_over_B_avg[:,None,None]*2*       phi_zed_x_y*dupar_zed_x_y                    , axis=(0,2))*dy

    return x, EZ_x, EZ_deltaphi2_x, dEZ_reynolds_phi_nablax2_x, dEZ_reynolds_Pprp_nablax2_x, dEZ_reynolds_phi_nablaxy_x, dEZ_reynolds_Pprp_nablaxy_x, dEZ_vDx_x, dEZ_upar_x


def get_EZ_omega_x(run, quantity, time_min=0, time_max=99999, time_idx_skip=1, nx=None):

    time = run.get_time_array()
    time_max = min(time[-1], time_max)
    if time_min < 0:
        time_min = time_max - np.abs(time_min)
    time_idx_min = nearest_index(time-time_min)
    time_idx_max = nearest_index(time-time_max)
    time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)
    time_eval = time[time_idx_eval]

    for i_time_idx, time_idx in enumerate(time_idx_eval):
        print("Evaluating zonal energy: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")
        kx_order = 0
        if quantity == "phi":
            kx_order = 1
        ### NOTE QUANTITY MUST NOT BE EQUAL PHI, BUT ASSUMED IN VARIABLE NAMING
        dxphi_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y(quantity=quantity, only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=kx_order, nx=nx)

        if i_time_idx == 0:
            dxphi_t_zed_x = np.zeros((len(time_eval),len(zed),len(x)))

        dxphi_t_zed_x[i_time_idx] = np.sum(dxphi_zed_x_y, axis=2)*(y[1]-y[0])

    # Resample to equal time-intervals
    dt = (np.gradient(time_eval)).max()
    time_interp = np.arange(time_eval[0], time_eval[-1], dt)
    dxphi_t_zed_x_interp_func = interp(time_eval, dxphi_t_zed_x, assume_sorted=True, axis=0)
    dxphi_t_zed_x_interp = dxphi_t_zed_x_interp_func(time_interp)

    # Fourier transform to omega
    dxphi_omega_zed_x = np.fft.fft(dxphi_t_zed_x_interp, axis=0)/len(time_interp)
    omega = np.fft.fftfreq(len(time_interp), d=dt)*(2*np.pi)

    idx_sort = np.argsort(omega)
    dxphi_omega_zed_x = dxphi_omega_zed_x[idx_sort]
    omega = omega[idx_sort]

    # Geometric quantities
    dl_over_B_avg = run.dl_over_B_avg()
    _, _, _, _, gds22, bmag = run.get_FLR()
    nablax2 = gds22/bmag**2

    # Sum over zed
    EZ_omega_x     = np.sum(dl_over_B_avg[None,:,None]*0.5*np.abs(dxphi_omega_zed_x)**2 *nablax2[None,:,None], axis=1)

    return omega, x, EZ_omega_x


def get_EZ_omega_kx(run, quantity, time_min=0, time_max=99999, time_idx_skip=1, nx=None):

    # if time min is negative, count as tmax - |tmin|
    time = run.get_time_array()
    time_max = min(time[-1], time_max)
    if time_min < 0:
        time_min = time_max - np.abs(time_min)
    time_idx_min = nearest_index(time-time_min)
    time_idx_max = nearest_index(time-time_max)
    time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)
    time_eval = time[time_idx_eval]

    for i_time_idx, time_idx in enumerate(time_idx_eval):
        print("Evaluating zonal energy: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")
        ### NOTE QUANTITY MUST NOT BE EQUAL PHI, BUT ASSUMED IN VARIABLE NAMING
        kx_order = 0
        if quantity == "phi":
            kx_order = 1
        dxphi_zed_kx_ky, zed, kx, ky, _ = run.get_quantity_zed_kx_ky(quantity=quantity, only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=kx_order)

        if i_time_idx == 0:
            dxphi_t_zed_kx = np.zeros((len(time_eval),len(zed),len(kx)), dtype='complex')

        #Ly = 2*np.pi/(ky[1]-ky[0])
        dxphi_t_zed_kx[i_time_idx] = dxphi_zed_kx_ky[:,:,0]# * Ly/2

    # Resample to equal time-intervals
    dt = (np.gradient(time_eval)).max()
    time_interp = np.arange(time_eval[0], time_eval[-1], dt)
    dxphi_t_zed_kx_interp_func = interp(time_eval, dxphi_t_zed_kx, assume_sorted=True, axis=0)
    dxphi_t_zed_kx_interp = dxphi_t_zed_kx_interp_func(time_interp)

    # Fourier transform to omega
    dxphi_omega_zed_kx = np.fft.fft(dxphi_t_zed_kx_interp, axis=0)/len(time_interp)
    omega = np.fft.fftfreq(len(time_interp), d=dt)*(2*np.pi)

    # Geometric quantities
    dl_over_B_avg = run.dl_over_B_avg()
    _, _, _, _, gds22, bmag = run.get_FLR()
    nablax2 = gds22/bmag**2

    # Sum over zed
    EZ_omega_kx    = np.sum(dl_over_B_avg[None,:,None]*0.5*np.abs(dxphi_omega_zed_kx)**2 *nablax2[None,:,None], axis=1)

    return omega, kx, EZ_omega_kx


def get_EZ_omega(run, quantity="phi", time_min=0, time_max=99999, time_idx_skip=1, nx=None):
    omega, x, EZ_omega_x   = run.get_EZ_omega_x(quantity=quantity, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, nx=nx)
    # Sum over x
    EZ_omega     = np.sum(EZ_omega_x, axis=1)*(x[1]-x[0])
    return omega, EZ_omega


def get_Wenergy_t_zed_kx_ky(run, time_idx_min=None, time_idx_max=None, time_min=0, time_max=10000, time_idx_skip=1, tite=1):

    if time_idx_min is None:
        time_idx_min = run.get_time_idx(time_min)
    if time_idx_max is None:
        time_idx_max = run.get_time_idx(time_max)
    time = run.ncdata.variables['t'][time_idx_min:time_idx_max:time_idx_skip]
    kx   = run.ncdata.variables['kx'][:]
    ky   = run.ncdata.variables['ky'][:]
    zed  = run.ncdata.variables['zed'][:]

    # Energy in phi
    # phi_vs_t(t, tube, zed, theta0, ky, ri)
    phi_t_zed_kx_ky_ri  = run.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max:time_idx_skip,0,:,:,:,:]
    phi_t_zed_kx_ky = phi_t_zed_kx_ky_ri[:,:,:,:,0] + 1j*phi_t_zed_kx_ky_ri[:,:,:,:,1]
    dl_over_B_avg = run.dl_over_B_avg()
    kperp2 = run.ncdata.variables['kperp2'][:,0,:,:]
    bmag   = run.ncdata.variables['bmag'][:,0]
    Gamma0_arg = kperp2/ ((2*bmag**2)[:,None,None])
    Gamma0 = specialfunc.iv(0, Gamma0_arg) * np.exp(-Gamma0_arg)
    phiZ_t_kx = np.sum(phi_t_zed_kx_ky[:,:,:,0]*dl_over_B_avg[None,:,None], axis=1)

    Wenergy_phi_e_t_zed_kx_ky = tite*(np.abs(phi_t_zed_kx_ky-phiZ_t_kx[:,None,:,None]*(1-np.heaviside(ky,0))[None,None,None,:] )**2) /2
    
    Wenergy_phi_i_t_zed_kx_ky = ( (1-Gamma0)[None,:,:,:]*np.abs(phi_t_zed_kx_ky)**2 )/2

    # Energy in g
    #double Wenergy_g(t, species, tube, zed, kx, ky) ;
    # NOTE: pre-existing bug (predates the restructure, confirmed against
    # real stella runs) -- 'Wenergy_g' isn't written by every stella
    # version; on runs without it this raises an IndexError from netCDF4.
    # See README "Known issues".
    Wenergy_g_t_zed_kx_ky  = run.ncdata['Wenergy_g'][time_idx_min:time_idx_max:time_idx_skip,0,0,:,:,:]

    return Wenergy_g_t_zed_kx_ky, Wenergy_phi_e_t_zed_kx_ky, Wenergy_phi_i_t_zed_kx_ky, time, zed, kx, ky
