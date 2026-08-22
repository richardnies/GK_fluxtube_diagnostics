"""Velocity-space (vpa, mu) distribution-function diagnostics and marker/orbit evolution.

NOTE: pre-existing (predates the restructure, confirmed against real
stella runs) -- the 'gvmus'/'gzvs' netCDF variable lookups below (in
read_g_vs_zed, get_gvpa_gmu, get_Evpa_Emu, get_n_T_vpa_mu) expect an
older stella output naming convention; some stella versions instead
write these under names like 'g2_vs_vpamus'/'g2_vs_zvpas', which makes
these functions raise KeyError. See README "Known issues".
"""

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
from stella_diagnostics.io.codes import get_vt_label
from stella_diagnostics.spectral.stats import dt_weighted_mean, dt_weights
from stella_diagnostics.plotting.mpl_helpers import resolve_vmin_vmax


def read_g_vs_zed(run, time_idx=-1, species_idx=0, vpa_index=None, normalise=True):

    # gzvs(t, species, vpa, zed, tube) -- older stella netCDF variable
    # name; some stella versions write g2_vs_zvpas instead (same
    # shape/dim order).
    try:
        gzvs = run.ncdata.variables['gzvs'][time_idx,species_idx,:,:,0]
    except KeyError:
        gzvs = run.ncdata.variables['g2_vs_zvpas'][time_idx,species_idx,:,:,0]

    # vpa?
    vpa = run.ncdata.variables['vpa']
    #print(vpa[vpa_index])
    if vpa_index is None:
        gz = np.sum(np.abs(gzvs), axis=0)
    else:
        gz = gzvs[vpa_index,:]
    #gz = np.sum(gzvs, axis=0)

    if normalise:
        max_g = np.max(gz)
        min_g = np.min(gz)
        if np.abs(max_g) > np.abs(min_g):
            gz = gz / max_g
        else:
            gz = gz / min_g

    zed      = run.ncdata.variables['zed'][:]

    return gz, zed


def get_gvpa_gmu(run, time_idx=-1, species_idx=0, remove_zonal=False, only_zonal=False):

    if run.code == "stella":
        # gvmus(t, species, mu, vpa) -- older stella netCDF variable name
        # (gvmus_Z/gvmus_NZ for the zonal-only/zonal-removed variants);
        # some stella versions write g2_vs_vpamus/g2nozonal_vs_vpamus
        # instead, with no direct zonal-only equivalent -- derive it as
        # total minus nozonal (same pattern as plot_contour_gvmu_vpa).
        if only_zonal:
            try:
                gvmus  = run.ncdata.variables['gvmus_Z'][time_idx,species_idx]
            except KeyError:
                gvmus_tot = run.ncdata.variables['g2_vs_vpamus'][time_idx,species_idx]
                gvmus_NZ  = run.ncdata.variables['g2nozonal_vs_vpamus'][time_idx,species_idx]
                gvmus = gvmus_tot - gvmus_NZ
        elif remove_zonal:
            try:
                gvmus  = run.ncdata.variables['gvmus_NZ'][time_idx,species_idx]
            except KeyError:
                gvmus  = run.ncdata.variables['g2nozonal_vs_vpamus'][time_idx,species_idx]
        else:
            try:
                gvmus  = run.ncdata.variables['gvmus'][time_idx,species_idx]
            except KeyError:
                gvmus  = run.ncdata.variables['g2_vs_vpamus'][time_idx,species_idx]

        vpa    = run.ncdata.variables['vpa'][:]
        mu     = run.ncdata.variables['mu'][:]
        time   = run.ncdata.variables['t'][time_idx]

        bmag = 1
        integrand = gvmus

    elif run.code == "GX":
        vpa = np.linspace(-3,3,50)
        mu  = np.linspace(0,5,100)
        time   = run.ncdata.variables['time'][time_idx]
        from scipy import special

        Wml  = run.ncdata['Spectra']['Wlmst'][time_idx,species_idx]
        nhermite  = run.ncdata['nhermite'].getValue()
        nlaguerre = run.ncdata['nlaguerre'].getValue()

        print("Wml(2,0)/Wml(0,0) = %e" % (Wml[2,0]/Wml[0,0]))
        print("Wml(0,1)/Wml(0,0) = %e" % (Wml[0,1]/Wml[0,0]))
        #Wml[:,:] = 0
        #Wml[0,1] = 0

        Wvmus = np.zeros((len(mu),len(vpa)))
        for i_hermite in range(nhermite):
            hermite_pol = special.hermitenorm(i_hermite)
            for i_laguerre in range(nlaguerre):
                laguerre_pol = special.laguerre(i_laguerre)

                Wvmus = Wvmus + Wml[i_hermite,i_laguerre] * hermite_pol(vpa)[None,:] * laguerre_pol(mu)[:,None] * (-1)**(i_laguerre) * np.exp(-mu)[:,None] * np.exp(-vpa**2/2)[None,:] * (np.math.factorial(i_hermite))**(-1/2)
                #Wvmus = Wvmus + Wml[i_hermite,i_laguerre] * hermite_pol(vpa)[None,:] * laguerre_pol(mu)[:,None] * (-1)**(i_laguerre) * np.exp(-mu)[:,None] * np.exp(-vpa**2/2)[None,:] * (np.math.factorial(i_hermite))**(-1/2)

                #Wvmus = Wvmus + Wml[ i_hermite, i_laguerre] * hermite_pol(vpa)[None,:] * laguerre_pol(mu)[:,None] * np.exp(-mu)[:,None] * np.exp(-vpa**2/2)[None,:] / np.sqrt( np.math.factorial(i_hermite) )
                #Wvmus = Wvmus + Wml[i_hermite,i_laguerre] * hermite_pol(vpa)[None,:] * laguerre_pol(mu)[:,None] * (-1)**(i_laguerre) * np.exp(-mu)[:,None] * np.exp(-vpa**2/2)[None,:] / np.sqrt( np.math.factorial(i_hermite) )

        integrand = Wvmus


    dmu  = np.abs( mu[1]- mu[0])
    dvpa = np.abs(vpa[1]-vpa[0])
    gmu  = np.sum( integrand, axis=1) * dvpa
    gvpa = np.sum( integrand, axis=0) * dmu


    return gmu, gvpa, mu, vpa, time


def get_Evpa_Emu(run, time_idx=-1, species_idx=0):

    # gvmus(t, species, mu, vpa) -- older stella netCDF variable name;
    # some stella versions write g2_vs_vpamus instead (same shape/dim
    # order).
    try:
        gvmus  = run.ncdata.variables['gvmus'][time_idx,species_idx]
    except KeyError:
        gvmus  = run.ncdata.variables['g2_vs_vpamus'][time_idx,species_idx]
    vpa    = run.ncdata.variables['vpa'][:]
    mu     = run.ncdata.variables['mu'][:]
    time   = run.ncdata.variables['t'][time_idx]

    bmag = 1
    integrand = gvmus * (vpa[None,:]**2 + 2*mu[:,None]*bmag)/2

    dmu  = np.abs( mu[1]- mu[0])
    dvpa = np.abs(vpa[1]-vpa[0])
    Emu  = np.sum( integrand, axis=1) * dvpa
    Evpa = np.sum( integrand, axis=0) * dmu

    return Emu, Evpa, mu, vpa, time


def get_n_T_vpa_mu(run, time_idx=-1, species_idx=0):

    # gvmus(t, species, mu, vpa) -- older stella netCDF variable name;
    # some stella versions write g2_vs_vpamus instead (same shape/dim
    # order).
    try:
        gvmus  = run.ncdata.variables['gvmus'][time_idx,species_idx]
    except KeyError:
        gvmus  = run.ncdata.variables['g2_vs_vpamus'][time_idx,species_idx]
    vpa    = run.ncdata.variables['vpa'][:]
    mu     = run.ncdata.variables['mu'][:]
    time   = run.ncdata.variables['t'][time_idx]

    bmag = 1
    integrand_n = gvmus
    integrand_T = gvmus * (vpa[None,:]**2 + 2*mu[:,None]*bmag - 3/2)

    dmu  = np.abs( mu[1]- mu[0])
    dvpa = np.abs(vpa[1]-vpa[0])
    nmu  = np.sum( integrand_n, axis=1) * dvpa
    nvpa = np.sum( integrand_n, axis=0) * dmu
    Tmu  = np.sum( integrand_T, axis=1) * dvpa
    Tvpa = np.sum( integrand_T, axis=0) * dmu

    return nmu, nvpa, Tmu, Tvpa, mu, vpa, time


def plot_contour_gvmu_vpa(run, fig=None, ax=None, time_idx=-1, vmin=None, vmax=None, logarithmic=False, cmap='inferno', plot_diff=False, zonal=False, nozonal=False, species_idx=0, kx_min=None, kx_max=None, time_avg=None):
    # time_avg: trailing-window width ending at time_idx's own time
    # (time_avg=None -> single snapshot at time_idx). Same trailing-window
    # convention as stella_diagnostics.scan.zonal_flow_scan/rh_flux_scan
    # (time_val_avg=None branch), just evaluated at this call's own
    # time_idx instead of the run's last sample.
    if time_avg is None:
        time_idx_eval = time_idx
    else:
        time_all = run.get_time_array()
        time_eval = time_all[time_idx]
        time_min = time_eval-time_avg#/2
        time_max = time_eval#+time_avg/2
        time_idx_min = run.get_time_idx(time_min)
        time_idx_max = run.get_time_idx(time_max)
        time_idx_eval = np.arange(time_idx_min, time_idx_max)
        dt_vals = dt_weights(time_all[time_idx_eval])

    # gvmus(t, species, mu, vpa)
    if kx_min is None and kx_max is None:

        if zonal:
            try:
                gvmus  = run.ncdata.variables['gvmus_Z'][time_idx_eval,species_idx]
            except:
                gvmus_tot  = run.ncdata.variables['g2_vs_vpamus'][time_idx_eval,species_idx]
                gvmus_NZ   = run.ncdata.variables['g2nozonal_vs_vpamus'][time_idx_eval,species_idx]
                gvmus = gvmus_tot - gvmus_NZ
        else:
            try:
                gvmus  = run.ncdata.variables['gvmus'][time_idx_eval,species_idx]
            except:
                if nozonal:
                    gvmus  = run.ncdata.variables['g2nozonal_vs_vpamus'][time_idx_eval,species_idx]
                else:
                    gvmus  = run.ncdata.variables['g2_vs_vpamus'][time_idx_eval,species_idx]

    else:
        if kx_min is None:
            kx_min = 0
        if kx_max is None:
            kx_max = 1e20

        kx = run.ncdata.variables['kx'][:]

        if zonal:
           gkxvmus_tot = run.ncdata.variables['g2_vs_kxvpamus'][time_idx_eval,species_idx]
           gkxvmus_NZ  = run.ncdata.variables['g2nozonal_vs_kxvpamus'][time_idx_eval,species_idx]
           gkxvmus = gkxvmus_tot - gkxvmus_NZ
        elif nozonal:
           gkxvmus     = run.ncdata.variables['g2nozonal_vs_kxvpamus'][time_idx_eval,species_idx]
        else:
           gkxvmus     = run.ncdata.variables['g2_vs_kxvpamus'][time_idx_eval,species_idx]

        # Sum over kx's within desired range 
        if time_avg is None:
            gvmus = np.sum(gkxvmus[  :,:, ( (np.abs(kx) >= kx_min) & (np.abs(kx) <= kx_max) )], axis=2)
        else:
            gvmus = np.sum(gkxvmus[:,:,:, ( (np.abs(kx) >= kx_min) & (np.abs(kx) <= kx_max) )], axis=3)

    vpa    = run.ncdata.variables['vpa']
    mu     = run.ncdata.variables['mu']
    time   = run.ncdata.variables['t'][time_idx]

    if time_avg is not None:
        gvmus = dt_weighted_mean(gvmus, weights=dt_vals, axis=0)

    if plot_diff:
        try:
            gvmus_init = run.ncdata.variables['gvmus'][0,species_idx]
        except KeyError:
            gvmus_init = run.ncdata.variables['g2_vs_vpamus'][0,species_idx]
        gvmus = gvmus - gvmus_init

    X, Y = np.meshgrid(vpa, mu)
    Z = gvmus

    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    if logarithmic:
        Z = np.abs(Z)

    # Was hand-rolled here (a Z.min()<1e-15 guard + a vmax/1e4 log floor)
    # duplicated with a different, inconsistent floor (1e-2*vmax) in
    # plot_contour_gzvs below and in every other 2D contour plot in the
    # package -- now shares the one convention in mpl_helpers.
    vmin, vmax = resolve_vmin_vmax(Z, vmin, vmax, logarithmic, default_vmax=Z.max())

    if logarithmic:
        try:
            im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
            #im = ax.contourf(X, Y, Z, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap, levels=100)
        except:
            logarithmic = False

    if not logarithmic:
        im = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)

    ax.set_xlabel(r"$v_\parallel/%s$" % get_vt_label(run.ncdata))
    ax.set_ylabel(r"$\mu B_\mathrm{max}/T$")
    fig.suptitle(r"$t %s/a = %.2f$" % (get_vt_label(run.ncdata), time))

    return fig, ax, im


def plot_contour_gzvs(run, fig=None, ax=None, time_idx=-1, vmin=None, vmax=None, logarithmic=False, cmap='inferno', plot_diff=False, zonal=False, nozonal=False):
    

    # gzvs(t, species, vpa, zed, tube)
    try:
        gzvs_tot = run.ncdata.variables['gzvs'][time_idx,0, :, :, 0]
    except:
        gzvs_tot = run.ncdata.variables['g2_vs_zvpas'][time_idx,0, :, :, 0]
        gzvs_NZ  = run.ncdata.variables['g2nozonal_vs_zvpas'][time_idx,0, :, :, 0]

    if zonal:
        gzvs = gzvs_tot-gzvs_NZ
    elif nozonal:
        gzvs = gzvs_NZ
    else:
        gzvs = gzvs_tot

    vpa  = run.ncdata.variables['vpa']
    zed       = run.ncdata.variables['zed'][:]
    time = run.ncdata.variables['t'][time_idx]

    if plot_diff:
        gzvs_init = run.ncdata.variables['gzvs'][0,0, :, :, 0]
        gzvs = gzvs - gzvs_init

    X, Y = np.meshgrid(vpa, zed)
    Z = gzvs.T

    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    if logarithmic:
        Z = np.abs(Z)

    # See plot_contour_gvmus above -- same shared floor now, not a
    # separately hand-rolled (and inconsistent) one.
    vmin, vmax = resolve_vmin_vmax(Z, vmin, vmax, logarithmic, default_vmax=Z.max())

    if logarithmic:
        im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
    else:
        im = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)

    ax.set_xlabel(r"$v_\parallel/%s$" % get_vt_label(run.ncdata))
    ax.set_ylabel(r"$\zeta$")
    ax.set_title(r"$t %s/a = %.2f$" % (get_vt_label(run.ncdata), time))

    return fig, ax, im


def evolve_markers_2D(run, t_min=0, t_max=np.inf, x0=[0], y0=[0], only_zonal_vEx=False, only_zonal_vEy=False, remove_zonal=False, zed_val=0, nx=None, ny=None, kx_highpass_cutoff=-1):

    # Time interval
    time_all = run.get_time_array()
    time_idx_min = nearest_index(time_all-t_min)
    time_idx_max = nearest_index(time_all-t_max)
    time_eval = time_all[time_idx_min:time_idx_max]

    Nmarkers = len(x0)
    assert(len(x0)==len(y0))
    x_t_vals = np.zeros((Nmarkers, len(time_eval)))
    y_t_vals = np.zeros((Nmarkers, len(time_eval)))
    x_t_vals[:,0] = x0[:]
    y_t_vals[:,0] = y0[:]

    # Evaluate velocity as a function of (x,y,t)
    for i_t in range(len(time_eval)-1):
        print("Marker evolution: time step %i/%i..." % (1+i_t, len(time_eval)), end="\r")

        # Evaluate velocity field
        vEx_x_y, x, y, _ = run.get_quantity_x_y(quantity="phi", zed_val=zed_val, time_idx=time_idx_min+i_t, remove_zonal=remove_zonal, only_zonal=only_zonal_vEx, ky_order=1, nx=nx, ny=ny, kx_highpass_cutoff=kx_highpass_cutoff)
        vEx_x_y = -vEx_x_y
        vEy_x_y, x, y, _ = run.get_quantity_x_y(quantity="phi", zed_val=zed_val, time_idx=time_idx_min+i_t, remove_zonal=remove_zonal, only_zonal=only_zonal_vEy, kx_order=1, nx=nx, ny=ny, kx_highpass_cutoff=kx_highpass_cutoff)

        # Shift for easier periodicity
        x = x-x[0]
        y = y-y[0]

        # Interpolating function 
        vEx_x_y_interp = interp2D(points=(x,y), values=vEx_x_y)
        vEy_x_y_interp = interp2D(points=(x,y), values=vEy_x_y)

        # Time step
        dt = time_eval[i_t+1]-time_eval[i_t]
        for i_m in range(Nmarkers):
            x_per = x_t_vals[i_m,i_t] % x[-1]
            y_per = y_t_vals[i_m,i_t] % y[-1]
            dx = dt*vEx_x_y_interp( (x_per, y_per) )
            dy = dt*vEy_x_y_interp( (x_per, y_per) )
            x_t_vals[i_m,i_t+1] = x_t_vals[i_m,i_t] + dx
            y_t_vals[i_m,i_t+1] = y_t_vals[i_m,i_t] + dy

    return time_eval, x_t_vals, y_t_vals
