"""Time-averaged rho_i-normalised wavenumber statistics (avg ky*rhoi, kx*rhoi, kperp*rhoi) and generic time-trace average/stddev/convergence utilities."""

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


def read_avg_ky_rhoi(run, time_idx_jump=1, avg_qflx=False, normal_mean=False, take_max=False):
    time   = run.ncdata.variables['t'][::time_idx_jump]
    Ntime = len(time)
    ky     = run.ncdata.variables['ky'][1:]

    if avg_qflx:
        # Average over qflx
	    # qflx_kxky(t, species, tube, zed, kx, ky)
        qflx_t_zed_kx_ky = run.ncdata.variables['qflx_kxky'][::time_idx_jump,0, 0, :, :, 1:]
        dl_over_B_avg = run.dl_over_B_avg()
        phi2_vs_kxky = np.sum( qflx_t_zed_kx_ky*dl_over_B_avg[None,:,None,None], axis=1)
        
    else:
        # Average over phi^2
        # phi2_vs_kxky(t, kx, ky)
        phi2_vs_kxky = run.ncdata.variables['phi2_vs_kxky'][::time_idx_jump,:,1:]

    ky_rhoi_O = np.zeros(Ntime) 

    for i_time in range(Ntime):
        if take_max:
            phi2_ky = np.sum(phi2_vs_kxky[i_time], axis=0)
            ky_rhoi_O[i_time] = ky[np.argmax(phi2_ky)]
        else:
            denominator = np.sum(phi2_vs_kxky[i_time])
            if normal_mean:
                numerator   = np.sum(phi2_vs_kxky[i_time]*ky[None,:])
                ky_rhoi_O[i_time] = numerator/denominator
            else:
                numerator   = np.sum(phi2_vs_kxky[i_time]/ky[None,:])
                ky_rhoi_O[i_time] = 1. / (numerator/denominator)

    return ky_rhoi_O, np.asarray(time)


def read_avg_kx_rhoi(run, time_idx_jump=1, avg_qflx=False, normal_mean=False, take_max=False, only_zonal=False, remove_zonal=True):
    time   = run.ncdata.variables['t'][::time_idx_jump]
    Ntime = len(time)
    kx     = run.ncdata.variables['kx'][:]

    if avg_qflx and not only_zonal:
        # Average over qflx
	    # qflx_kxky(t, species, tube, zed, kx, ky)
        qflx_t_zed_kx_ky = run.ncdata.variables['qflx_kxky'][::time_idx_jump,0, 0, :, :, 1:]
        dl_over_B_avg = run.dl_over_B_avg()
        phi2_vs_kxky = np.sum( qflx_t_zed_kx_ky*dl_over_B_avg[None,:,None,None], axis=1)
        
    else:
        # Average over phi^2
        # phi2_vs_kxky(t, kx, ky)
        phi2_vs_kxky = run.ncdata.variables['phi2_vs_kxky'][::time_idx_jump]
        if only_zonal:
            phi2_vs_kxky[:,:,1:] = 0
        elif remove_zonal:
            phi2_vs_kxky[:,:,0] = 0

    kx_rhoi_O = np.zeros(Ntime) 

    for i_time in range(Ntime):
        if take_max:
            phi2_kx = np.sum(phi2_vs_kxky[i_time], axis=1)
            kx_rhoi_O[i_time] = np.abs(kx[np.argmax(phi2_kx)])
        else:
            denominator = np.sum(phi2_vs_kxky[i_time])
            if normal_mean:
                numerator   = np.sum(phi2_vs_kxky[i_time]*np.abs(kx[:,None]))
                kx_rhoi_O[i_time] = numerator/denominator
            else:
                numerator   = np.sum(phi2_vs_kxky[i_time]/np.abs(kx[:,None]))
                kx_rhoi_O[i_time] = 1. / (numerator/denominator)

    return kx_rhoi_O, np.asarray(time)


def read_avg_kperp_rhoi(run, exclude_zonal=True, only_zonal=False, time_idx_jump=1):

    print("\n"+run.filename_base+":")

    # phi_vs_t(t, tube, zed, theta0, ky, ri)
    phi2_vs_t = np.abs( run.ncdata.variables['phi_vs_t'][::time_idx_jump,0,:,:,:,0] + 1j*run.ncdata.variables['phi_vs_t'][::time_idx_jump,0,:,:,:,1])**2
    time      = run.ncdata.variables['t'][::time_idx_jump] 
    Ntime     = len(time)
    # kperp2(zed, alpha, kx, ky)
    kperp2    = run.ncdata.variables['kperp2'][:,0,:,:]

    dl_over_B_avg = run.dl_over_B_avg()

    if exclude_zonal:
        phi2_vs_t[:,:,:,0] = 0
    if only_zonal:
        phi2_vs_t[:,:,:,1:] = 0

    # Avoid division by zero
    phi2_vs_t[:,:,0,0] = 0
    kperp2[:,0,0] = 1e16

    # For all times, obtain energy-averaged kperp
    kperp2_O        = np.zeros(Ntime)
    kperp2_O_stddev = np.zeros(Ntime)
    for i_time in range(Ntime):
        print("Time index %i/%i" % (i_time+1, Ntime), end="\r")
        numerator   = np.sum(phi2_vs_t[i_time]/kperp2, axis=(1,2))
        denominator = np.sum(phi2_vs_t[i_time],        axis=(1,2))

        numerator_stddev   = np.sum(phi2_vs_t[i_time]**2 * (1/kperp2 - (numerator/denominator)[:,None,None])**2, axis=(1,2))

        # Tube-average
        kperp2_O[i_time]        = np.sum(               denominator/numerator * dl_over_B_avg)
        kperp2_O_stddev[i_time] = np.sum( np.sqrt(numerator_stddev)/numerator * dl_over_B_avg)

    return kperp2_O, kperp2_O_stddev, np.asarray(time)


def get_statistics(time, f_t, dt):

    # Make sure dt is not smaller than minimum timestep size
    dt_min = 2*np.min(time[1:]-time[:-1])
    if dt < dt_min:
        print("dt for statistics was taken to be too small.")
        dt = dt_min

    # Get data on equal time intervals
    time_intervalled = np.arange(time[0]+dt/2, time[-1]-dt/2, dt)
    f_t_intervalled = np.zeros_like(time_intervalled)
    for i_interval, time_interval in enumerate(time_intervalled):
        time_min_integrate = time_interval-dt/2
        time_max_integrate = time_interval+dt/2
        time_idx_min = nearest_index(time-time_min_integrate)
        time_idx_max = nearest_index(time-time_max_integrate)
        f_t_intervalled[i_interval] = np.mean(f_t[time_idx_min:time_idx_max])

    # Compute mean, rms, etc
    f_t_mean = np.mean(f_t_intervalled)
    f_t_rms = np.sqrt( np.mean(f_t_intervalled**2) )
    f_t_std = np.std(f_t_intervalled)

    return time_intervalled, f_t_intervalled, f_t_mean, f_t_rms, f_t_std
