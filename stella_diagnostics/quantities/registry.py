"""Single dispatch table for reading a named physical quantity (phi, density, upar, ...) in (zed, kx, ky) k-space, plus the coarser (kx, ky) reduction built on top of it."""

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


def get_omega_s_k(run, ky_idx=0, kx_idx=0):

    ky     = run.ncdata.variables['ky'][ky_idx]
    theta0 = run.ncdata.variables['theta0'][kx_idx,ky_idx]
    shat   = run.ncdata.variables['shat']
    iota   = 1/run.ncdata.variables['q'].getValue()
    tprim  = run.ncdata.variables['tprim'][0]
    cvdrift  = np.squeeze(run.ncdata.variables['cvdrift'][:] ) # drift * grad(y)
    cvdrift0 = np.squeeze(run.ncdata.variables['cvdrift0'][:]) # drift * grad(x) * shat

    omega_sT = ky*tprim #= ky*rho * a/L_T -> norm = vT/a

    #geom_quantities = np.loadtxt(run.geo_file, skiprows=2).T
    #zeta = geom_quantities[1]
    #theta = zeta*iota
    #Kx = np.squeeze( ky*shat*(theta0 - theta) )
    #omega_k = ky*cvdrift + Kx*cvdrift0
    omega_k = ky*(cvdrift + theta0*cvdrift0)

    omega_s_k = omega_sT / omega_k

    return omega_s_k, omega_sT, omega_k


def get_Gamma0(run, ky_idx=0, kx_idx=0):
    kperp2 = run.ncdata.variables['kperp2'][:][:,0,kx_idx,ky_idx] 
    Gamma0 = specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2)
    return Gamma0


def get_quantity_zed_kx_ky(run, quantity, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, alt_slow_eval=False):

    if remove_zonal and only_zonal:
        print("WARNING! Both only_zonal and remove_zonal were set to True, will thus return f_x_y = 0.")

    kx, ky, zed = run.get_kx_ky_zed()
    time_all    = run.get_time_array(GX_big=True)

    if time_avg is not None:
        time_eval = time_all[time_idx]
        time_min  = max(0,            time_eval-time_avg/2)
        time_max  = min(time_all[-1], time_eval+time_avg/2)
        time_idx_min = np.argmin( np.abs(time_all-time_min) )
        time_idx_max = np.argmin( np.abs(time_all-time_max) )
        time_idx = np.arange(time_idx_min,time_idx_max+1)
 
    if time_val is not None:
        time_idx = np.argmin( np.abs(time_all-time_val) )

    if not alt_slow_eval:
        if quantity=="phi":
            if run.code == "stella":
                # phi_vs_t(t, tube, zed, theta0, ky, ri)
                f_zed_kx_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
            elif run.code == "GS2":
                f_kx_ky_ri = np.transpose( run.ncdata.variables['phi_igomega_by_mode'][time_idx] , axes=[1,0,2] )
            elif run.code == "GX":
                if run.GX_old_version:
                    print("WARNING! You are loading Phi_z in GX, which had issues in early code versions.")
                    f_zed_kx_ky_ri = np.transpose( run.ncdata['Special']['Phi_z'], axes=[2,1,0,3] )
                else:
                    f_zed_kx_ky_ri = np.transpose( run.ncdata_big['Diagnostics']['Phi'][time_idx] , axes=[2,1,0,3] )

        elif quantity=="deltaphi":
            f_zed_kx_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
            zed_weight = run.dl_over_B_avg()
            f_zed_kx_ky_ri = f_zed_kx_ky_ri - np.sum(f_zed_kx_ky_ri*zed_weight[:,None,None,None], axis=0)

        elif quantity=="(1-Gamma0)phi":
            kperp2_zed_kx_ky = run.ncdata.variables['kperp2'][:][:,species_idx,:,:]
            Gamma0 = specialfunc.iv(0, kperp2_zed_kx_ky/2) * np.exp(-kperp2_zed_kx_ky/2)
            f_zed_kx_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]*(1-Gamma0)[:,:,:,None]
 
        elif quantity=="E_Z":
            if run.code == "stella":
                # phi_vs_t(t, tube, zed, theta0, ky, ri)
                f_zed_kx_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
                f_zed_kx_ky_ri[:,:,1:,:] = 0
                f_zed_kx_ky = (f_zed_kx_ky_ri[:,:,:,0] + 1j*f_zed_kx_ky_ri[:,:,:,1])*1j*kx[None,:,None]
                f_zed_kx_ky_ri[:,:,:,0] = np.abs(f_zed_kx_ky)**2 /2
                f_zed_kx_ky_ri[:,:,:,1] = 0

        elif quantity=="density":
            # density(t, species, tube, zed, kx, ky, ri)
            f_zed_kx_ky_ri = run.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
        elif quantity=="qpar":
            f_zed_kx_ky_ri = run.ncdata.variables['qpar'][time_idx,species_idx,0,:,:,:,:]
        elif quantity=="upar":
            # upar(t, species, tube, zed, kx, ky, ri)
            f_zed_kx_ky_ri = run.ncdata.variables['upar'][time_idx,species_idx,0,:,:,:,:]
        elif quantity=="temperature":
            if run.code == "stella":
                # temperature(t, species, tube, zed, kx, ky, ri)
                f_zed_kx_ky_ri = run.ncdata.variables['temperature'][time_idx,species_idx,0,:,:,:,:]
            elif run.code == "GX":
                Tpar_zed_kx_ky_ri = np.transpose( run.ncdata_big['Diagnostics']['Tpar'][time_idx, species_idx] , axes=[2,1,0,3] )
                Tprp_zed_kx_ky_ri = np.transpose( run.ncdata_big['Diagnostics']['Tperp'][time_idx, species_idx] , axes=[2,1,0,3] )
                f_zed_kx_ky_ri = Tpar_zed_kx_ky_ri + Tprp_zed_kx_ky_ri
        elif quantity=="temperature_par": #(xpa^2-1/2)
            if run.code == "stella":
                # temperature(t, species, tube, zed, kx, ky, ri)
                P_zed_kx_ky_ri    = run.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
                try:
                    Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
                except:
                    Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
                n_zed_kx_ky_ri    = run.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
                f_zed_kx_ky_ri = P_zed_kx_ky_ri - 0.5*Pprp_zed_kx_ky_ri - 0.5*n_zed_kx_ky_ri
            elif run.code == "GX":
                f_zed_kx_ky_ri = np.transpose( run.ncdata_big['Diagnostics']['Tpar'][time_idx, species_idx] , axes=[2,1,0,3] )
        elif quantity=="temperature_perp": #(xprp^2-1)
            if run.code == "stella":
                # temperature(t, species, tube, zed, kx, ky, ri)
                try:
                    Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
                except:
                    Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
                n_zed_kx_ky_ri    = run.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
                f_zed_kx_ky_ri = Pprp_zed_kx_ky_ri - n_zed_kx_ky_ri
            elif run.code == "GX":
                f_zed_kx_ky_ri = np.transpose( run.ncdata_big['Diagnostics']['Tperp'][time_idx, species_idx] , axes=[2,1,0,3] )
        elif quantity=="pressure":
            try:
                f_zed_kx_ky_ri = run.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
            except KeyError:
                # Some stella builds don't write a combined 'pressure'
                # variable -- reconstruct from density/temperature/
                # pressure_perp, matching quantities/realspace.py's
                # get_quantity_zed_x_y (a separate, independently
                # maintained copy of this same dispatch -- see the
                # ~40-branch quantity dispatch duplication noted in
                # README.md's "Known issues") which already had this
                # fallback.
                n_zed_kx_ky_ri = run.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
                T_zed_kx_ky_ri = run.ncdata.variables['temperature'][time_idx,species_idx,0,:,:,:,:]
                Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
                f_zed_kx_ky_ri = 1.5*(T_zed_kx_ky_ri+n_zed_kx_ky_ri) - 0.5*Pprp_zed_kx_ky_ri
        elif quantity=="chi":
            try:
                f_zed_kx_ky_ri = run.ncdata.variables['chi'][time_idx,species_idx,0,:,:,:,:]
            except:
                print("chi not found in NETCDF! Using pressure instead.")
                f_zed_kx_ky_ri = run.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]

        elif quantity=="pressure_par":
            P_zed_kx_ky_ri = run.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
            Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
            f_zed_kx_ky_ri = P_zed_kx_ky_ri-0.5*Pprp_zed_kx_ky_ri
        elif quantity=="pressure_perp":
            # pressure_perp(t, species, tube, zed, kx, ky, ri)
            try:
                f_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
            except:
                f_zed_kx_ky_ri = run.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
        elif quantity=="qflx":
           # qflx_kxky(t, species, tube, zed, kx, ky)
           f_zed_kx_ky = run.ncdata.variables['qflx_kxky'][time_idx,species_idx,0,:,:,:]
           f_zed_kx_ky_ri = np.zeros( (len(zed), len(kx), len(ky), 2))
           f_zed_kx_ky_ri[:,:,:,0] = f_zed_kx_ky

        else:
            alt_slow_eval = True

    if alt_slow_eval:
       # Evaluate in real space first and then Fourier transform
       f_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity=quantity, time_idx=time_idx, time_val=time_val, time_avg=time_avg, remove_zonal=remove_zonal, only_zonal=only_zonal)
       f_zed_x = np.sum(f_zed_x_y, axis=2)*2/len(y)
       for i_zed in range(len(zed)):
           f_kx, kx = get_fft_k(f_zed_x[i_zed], x)
           if i_zed == 0:
               f_zed_kx = np.zeros((len(zed), len(kx)), dtype='complex')

           f_zed_kx[i_zed] = f_kx

       ## CHECK NORMALISATION
       #dx = x[1]-x[0]
       #integral_x = np.sum(reynolds_stress_zed_x[0])*dx
       #print("\nIntegral_x: %e, kx=0: %e" % (integral_x, reynolds_stress_zed_kx[0,0]*(x[-1]-x[0])))

       return f_zed_kx[:,:,None]*(1j*kx[None,:,None])**kx_order, zed, kx, ky, time_eval

    time_eval = time_all[time_idx]
     
    if time_avg is not None:
        f_zed_kx_ky_ri = np.mean(f_zed_kx_ky_ri, axis=0)

    f_zed_kx_ky = f_zed_kx_ky_ri[:,:,:,0] + 1j*f_zed_kx_ky_ri[:,:,:,1]

    # x-derivatives
    f_zed_kx_ky = f_zed_kx_ky * (1j*kx[None,:,None])**kx_order
    #f_zed_kx_ky = f_zed_kx_ky * (1j*kx[None,:,None]/(kx[1]-kx[0]))**kx_order

    # y-derivatives
    f_zed_kx_ky = f_zed_kx_ky * (1j*ky[None,None,:])**ky_order
    #f_zed_kx_ky = f_zed_kx_ky * (1j*ky[None,None,:]/(ky[1]-ky[0]))**ky_order

    # Filter zonal if requested
    if remove_zonal:
        f_zed_kx_ky[:,:,0]= 0
    if only_zonal:
        f_zed_kx_ky[:,:,1:]= 0

    return f_zed_kx_ky, zed, kx, ky, time_eval


def get_quantity_kx_ky(run, quantity, zed_val = None, zed_idx=None, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, mult_zed=None, par_der_order=0, mean_delt_zed=None, alt_slow_eval=False, sort_kx=False):

    if remove_zonal and only_zonal:
        print("WARNING! Both only_zonal and remove_zonal were set to True, will thus return f_x_y = 0.")

    if quantity=="phi" and run.code == "GS2":
        f_kx_ky_ri = np.transpose( run.ncdata.variables['phi_igomega_by_mode'][time_idx] , axes=[1,0,2] )
        zed       = run.ncdata.variables['theta'][:]
        f_zed_kx_ky_ri = np.tile(f_kx_ky_ri[None,:,:,:].T, len(zed)).T
        f_zed_kx_ky = f_zed_kx_ky_ri[:,:,:,0] + 1j*f_zed_kx_ky_ri[:,:,:,1]
        ky        = run.ncdata.variables['ky'][:]
        kx        = run.ncdata.variables['kx'][:]
        time_eval = run.ncdata.variables['t'][time_idx]
    else:
        f_zed_kx_ky, zed, kx, ky, time_eval = run.get_quantity_zed_kx_ky(quantity, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, alt_slow_eval=alt_slow_eval)

    # if zed_val is not None, find zed_idx matching zed_val most closely
    if zed_val is not None:
        zed_idx = np.argmin( np.abs(zed - zed_val) )

    zed_weight = run.get_zed_weight(mult_zed=mult_zed, zed=zed)

    # Take zed derivatives if needed
    for i in range(par_der_order):
        gradpar  = run.ncdata.variables['gradpar'][:]
        # Use periodicity
        _, _, _, _, gds22, bmag = run.get_FLR()
        f_zed_kx_ky = np.gradient(f_zed_kx_ky*gradpar[:,None,None], zed, axis=0)
 
    try:
        tmp = f_kx_ky[0,0]
    except:
        f_zed_kx_ky = f_zed_kx_ky * zed_weight[:,None,None]

        if mean_delt_zed is not None:
            dl_over_B_avg = run.dl_over_B_avg()
            mean_f_kx_ky = np.sum(dl_over_B_avg[:,None,None]*f_zed_kx_ky, axis=0)
            mean_f_zed_kx_ky = np.zeros_like(f_zed_kx_ky)
            for i_zed in range(len(zed)):
                mean_f_zed_kx_ky[i_zed] = mean_f_kx_ky

            if mean_delt_zed == "mean":
                f_zed_kx_ky = mean_f_zed_kx_ky
            elif mean_delt_zed == "delt":
                f_zed_kx_ky = f_zed_kx_ky - mean_f_zed_kx_ky

        if zed_idx is None:
            f_kx_ky = np.sum(f_zed_kx_ky, axis=0)
        else:
            f_kx_ky = f_zed_kx_ky[zed_idx]

    # Sort kx if required
    if sort_kx:
        idx_sort = np.argsort(kx)
        kx = kx[idx_sort]
        f_kx_ky = f_kx_ky[idx_sort]

    return f_kx_ky, kx, ky, time_eval
