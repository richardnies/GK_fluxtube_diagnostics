"""Core grid/time readers shared across all diagnostics: kx/ky/zed grids, the time array, nearest-time-index lookup, flux-tube bounce-average weighting, and finite-Larmor-radius (FLR) geometry factors."""

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


def read_basic_params(run):
    dict_params = dict()

    dict_params['theta0'] = run.ncdata['theta0'][0]
    dict_params['kx'] = run.ncdata['kx'][:]
    dict_params['ky'] = run.ncdata['ky'][:]

    return dict_params


def get_kx_ky_zed(run):
    if run.code == "stella":
        kx     = run.ncdata.variables['kx'][:]
        ky     = run.ncdata.variables['ky'][:] 
        zed    = run.ncdata.variables['zed'][:]
    elif run.code == "GX":
        if run.GX_old_version:
            kx     = np.array(run.ncdata.variables['kx'][:]) * np.sqrt(2) # sqrt(2) factor as we care about kx*rho_i !
            ky     = np.array(run.ncdata.variables['ky'][:]) * np.sqrt(2) 
            zed    = run.ncdata.variables['theta'][:]
        else:
            kx     = np.array(run.ncdata['Grids']['kx'][:])   * np.sqrt(2)
            ky     = np.array(run.ncdata['Grids']['ky'][:])   * np.sqrt(2)
            zed    = run.ncdata['Grids']['theta'][:]

    elif run.code == "GS2":
        kx     = run.ncdata.variables['kx'][:]
        ky     = run.ncdata.variables['ky'][:] 
        zed    = run.ncdata.variables['zed'][:]
    return kx, ky, zed


def get_time_array(run, GX_big=False):
    if run.code == "GX":
        if run.GX_old_version:
            time   = run.ncdata.variables['time'][:]
        elif GX_big:
            time = run.ncdata_big['Grids']['time'][:]
        else:
            time = run.ncdata['Grids']['time'][:]
    else:
        time   = run.ncdata.variables['t'][:]
    return time


def nearest_index(diff):
    """Index of the array entry closest to a reference value.

    Callers pass the *difference* array/expression (e.g. ``time - t_ref``,
    or just ``zed`` when the reference is zero); this collapses the
    ``np.argmin(np.abs(...))`` boilerplate repeated at ~40 call sites
    across the codebase into one place.
    """
    return np.argmin(np.abs(diff))


def get_time_idx(run, time_val):
    time = run.get_time_array()
    return nearest_index(time - time_val)


def get_zed_weight(run, mult_zed, zed=None):

    zed_weight  = run.dl_over_B_avg()
    if run.code=="stella":
        shat   = run.ncdata.variables['shat'].getValue()
    elif run.code=="GX":
        shat   = run.ncdata['Geometry']['shat']

    if mult_zed == 1:
        zed_weight[:] = 1
    elif mult_zed is None:
        zed_weight = zed_weight*1
    elif mult_zed == "nablax-nablax":
        _, _, _, _, gds22, bmag = run.get_FLR()
        zed_weight = zed_weight*gds22/bmag**2
    elif mult_zed == "nablax2-vdriftx":
        _, _, _, _, gds22, bmag = run.get_FLR()
        zed_weight = zed_weight * run.ncdata.variables['gbdrift0'][:,0]/(2*shat)*gds22/bmag**2
    elif mult_zed == "nablax-nablay":
        _, _, _, gds21, _, bmag = run.get_FLR()
        zed_weight = zed_weight*gds21/bmag**2
    elif mult_zed == "nablaxy-vdriftx":
        _, _, _, gds21, _, bmag = run.get_FLR()
        zed_weight = zed_weight * run.ncdata.variables['gbdrift0'][:,0]/(2*shat)*gds21/bmag**2
    elif mult_zed == "vdrifty":
        zed_weight = zed_weight * run.ncdata.variables['gbdrift'][:,0]
    elif mult_zed == "vdriftx-vdrifty":
        zed_weight = zed_weight * run.ncdata.variables['gbdrift'][:,0]* run.ncdata.variables['gbdrift0'][:,0]/(2*shat)
    elif mult_zed == "vdriftx":
        zed_weight = zed_weight * run.ncdata.variables['gbdrift0'][:,0]/(2*shat)
        if run.debug:
            print("Note: sum(vdriftx)/sum(|vdriftx|) = %e" % (np.sum(zed_weight)/np.sum(np.abs(zed_weight))))
    elif mult_zed == "vdriftx-B":
        _, _, _, _, gds22, bmag = run.get_FLR()
        zed_weight = zed_weight*run.ncdata.variables['gbdrift0'][:,0]/(2*shat)*bmag
    elif mult_zed == "B":
        _, _, _, _, _, bmag = run.get_FLR()
        zed_weight = zed_weight*bmag
    elif mult_zed == "vdriftx-abs":
        zed_weight = zed_weight*np.abs(run.ncdata.variables['gbdrift0'][:,0]/(2*shat))
    elif mult_zed == "vdriftx2":
        zed_weight = zed_weight * (run.ncdata.variables['gbdrift0'][:,0]/(2*shat))**2
    elif mult_zed == "pos":
        zed_weight[zed<0] = 0
    elif mult_zed == "neg":
        zed_weight[zed>0] = 0
    elif mult_zed == "sin":
        zed_weight = zed_weight * np.sin(zed)
    elif mult_zed == "cos":
        zed_weight = zed_weight * np.cos(zed)
    elif mult_zed == "hfs":
        zed_weight[np.abs(zed)<np.pi/2] = 0
    elif mult_zed == "lfs":
        zed_weight[np.abs(zed)>np.pi/2] = 0
    else:
        print("WARNING! The indicated mult_zed is not in the list of options.")

    return zed_weight


def dl_over_B_avg(run):

    if run.code == "stella" or run.code == "GS2":
        gradpar = run.ncdata.variables['gradpar'][:]
        bmag    = run.ncdata.variables['bmag'][:]
    elif run.code == "GX":
        gradpar  = run.ncdata['Geometry']['gradpar']
        bmag     = run.ncdata['Geometry']['bmag'][:]

    dl_over_b = np.squeeze(1/(gradpar*bmag.T))
    #dl_over_b[ 0] = 0.5*dl_over_b[ 0] # First and last points of tube are connected
    #dl_over_b[-1] = 0.5*dl_over_b[-1] # First and last points of tube are connected
    dl_over_b_avg = np.squeeze(dl_over_b/np.sum(dl_over_b))

    return dl_over_b_avg


def get_avg_kperp2(run, ky_idx=0, kx_idx=0):
    kperp2 = run.ncdata.variables['kperp2'][:][:,0,kx_idx,ky_idx] 
    dl_over_B_avg = run.dl_over_B_avg()

    return np.sum(kperp2*dl_over_B_avg)


def get_FLR(run, ky_idx=0, kx_idx=0):
    # FLR = finite-Larmor-radius: the Gamma0 gyroaveraging factor and its
    # underlying k_perp^2 at a single (kx, ky) point.
    if run.code in ["stella", "GS2"]:
        kperp2 = run.ncdata.variables['kperp2'][:][:,0,kx_idx,ky_idx] 
        Gamma0 = specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2)
        gds2   = run.ncdata.variables['gds2'][:,0]  # |nabla(y)|^2
        gds21  = run.ncdata.variables['gds21'][:,0] # nabla(x)*nabla(y)
        gds22  = run.ncdata.variables['gds22'][:,0] # |nabla(x)|^2
        shat   = run.ncdata.variables['shat'].getValue()
        bmag   = run.ncdata.variables['bmag'][:,0]
        gds21  = gds21/shat
        gds22  = gds22/shat**2

    elif run.code == "GX":
        gds2   = run.ncdata['Geometry']['gds2'][:]  # |nabla(y)|^2
        gds21  = run.ncdata['Geometry']['gds21'][:] # nabla(x)*nabla(y)
        gds22  = run.ncdata['Geometry']['gds22'][:] # |nabla(x)|^2
        shat   = run.ncdata['Geometry']['shat'].getValue()
        bmag   = run.ncdata['Geometry']['bmag'][:]
        gds21  = gds21/shat
        gds22  = gds22/shat**2

        kx, ky, _ = run.get_kx_ky_zed()
        kperp2 = (kx[kx_idx]**2*gds22 + 2*kx[kx_idx]*ky[ky_idx]*gds21 + ky[ky_idx]**2*gds2)/bmag**2
        Gamma0 = specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2)

    return kperp2, Gamma0, gds2, gds21, gds22, bmag
