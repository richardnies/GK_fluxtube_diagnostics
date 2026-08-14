"""Plots of quantities in k-space and k-space-derived representations (spectra, zonal-mode time traces, kx-omega spectrograms)."""

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
from stella_diagnostics.io.codes import get_rho_label, get_vt_label
from stella_diagnostics.plotting.mpl_helpers import get_or_create_ax


def plot_spectrum2(run, quantity, kx_or_ky, fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, c=None, lw=None, label=None, marker='.', scale_kmin=True, scale_CB=False, zed_val=None, zed_idx=None, ls='-', mult_zed=None):

    if quantity == "upar_over_phi":
        upar_kx_ky, kx, ky, time_eval = run.get_quantity_kx_ky(quantity="upar", zed_val=zed_val, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)
        phi_kx_ky, kx, ky, time_eval = run.get_quantity_kx_ky(quantity="phi", zed_val=zed_val, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)
        quantity_kx_ky = np.abs(upar_kx_ky)/np.abs(phi_kx_ky)
        quantity_zed_kx_ky = quantity_kx_ky[None,:,:]

    elif quantity == "temp_over_phi":
        temp_kx_ky, kx, ky, time_eval = run.get_quantity_kx_ky(quantity="temperature", zed_val=zed_val, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)
        phi_kx_ky, kx, ky, time_eval = run.get_quantity_kx_ky(quantity="phi", zed_val=zed_val, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)
        quantity_kx_ky = np.abs(temp_kx_ky)/np.abs(phi_kx_ky)

        quantity_zed_kx_ky = quantity_kx_ky[None,:,:]

    else:
        quantity_zed_kx_ky, zed, kx, ky, time_eval = run.get_quantity_zed_kx_ky(quantity=quantity, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)


    fig, ax = get_or_create_ax(fig, ax, nrows=1, ncols=1, figsize=(12,9))

    if kx_or_ky == "kx":
        kx = np.abs(kx)
        k = kx[1:]
        dk = ky[1]-ky[0]
        quantity_zed_k = np.mean(np.real(quantity_zed_kx_ky[:,1:]*np.conj(quantity_zed_kx_ky[:,1:])), axis=2)/dk

    if kx_or_ky == "ky":
        k  = ky[1:]
        dk = kx[1]-kx[0]
        quantity_zed_k = np.mean(np.real(quantity_zed_kx_ky[:,:,1:]*np.conj(quantity_zed_kx_ky[:,:,1:])), axis=1)/dk

    # Average over zed
    zed_weight = run.get_zed_weight(mult_zed=mult_zed, zed=zed)
    quantity_k = np.sum(quantity_zed_k*zed_weight[:,None], axis=0)

    if scale_kmin:
        quantity_k = quantity_k / np.abs(k[1]-k[0])**2

    ax.loglog(k, quantity_k, label=label, lw=lw, c=c, marker=marker, ls=ls)

    return fig, ax, time_eval


def plot_quantity_zonal(run, quantity="phi", species_idx=0, fig=None, axs=None, zed_idx=None, time_idx=-1, label=None, ls=None, color=None, marker=None, substract_background_temp=False, normalise=False, time_avg=None, nx=None, sum_nonzonal=False, mult_zed=None, kx_order_min=0, kx_lowpass_cutoff=1e5, mult=1):

    if not sum_nonzonal:
        f_Z,       x, _, time_eval = run.get_quantity_x_y(quantity=quantity, species_idx=species_idx, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=False, only_zonal=True, kx_order=kx_order_min+0, time_avg=time_avg, nx=nx, mult_zed=mult_zed, kx_lowpass_cutoff=kx_lowpass_cutoff)
        fprime_Z,  x, _, time_eval = run.get_quantity_x_y(quantity=quantity, species_idx=species_idx, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=False, only_zonal=True, kx_order=kx_order_min+1, time_avg=time_avg, nx=nx, mult_zed=mult_zed, kx_lowpass_cutoff=kx_lowpass_cutoff)
        fdprime_Z, x, _, time_eval = run.get_quantity_x_y(quantity=quantity, species_idx=species_idx, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=False, only_zonal=True, kx_order=kx_order_min+2, time_avg=time_avg, nx=nx, mult_zed=mult_zed, kx_lowpass_cutoff=kx_lowpass_cutoff)

        # Make 1D array
        f_Z       = mult*f_Z[:,0]
        fprime_Z  = mult*fprime_Z[:,0]
        fdprime_Z = mult*fdprime_Z[:,0]

    else:
        f_Z,       x, _, time_eval = run.get_quantity_x_y(quantity=quantity, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=True, only_zonal=False, kx_order=0, time_avg=time_avg, nx=nx)
        fprime_Z,  x, _, time_eval = run.get_quantity_x_y(quantity=quantity, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=True, only_zonal=False, kx_order=1, time_avg=time_avg, nx=nx)
        fdprime_Z, x, _, time_eval = run.get_quantity_x_y(quantity=quantity, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=True, only_zonal=False, kx_order=2, time_avg=time_avg, nx=nx)

        # Make 1D array
        f_Z       = mult*np.sqrt(np.mean(np.abs(f_Z)**2       , axis=1))
        fprime_Z  = mult*np.sqrt(np.mean(np.abs(fprime_Z)**2  , axis=1))
        fdprime_Z = mult*np.sqrt(np.mean(np.abs(fdprime_Z)**2 , axis=1))


    if normalise:
        f_Z       = f_Z       / np.abs(f_Z      ).max()
        fprime_Z  = fprime_Z  / np.abs(fprime_Z ).max()
        fdprime_Z = fdprime_Z / np.abs(fdprime_Z).max()

    if axs is None:
        fig, axs = plt.subplots(nrows=3,ncols=1, figsize=(8,14), sharex=True)
        plt.subplots_adjust(left=0.15,right=0.95, hspace=0.05)

    title = r"$t= %.2f$" % (time_eval if np.ndim(time_eval) == 0 else time_eval[-1])
    if time_avg is not None:
        title = title + r"$_{\Delta t = %.1f}$" % (time_avg)
    fig.suptitle(title)

    axs[0].plot(x, f_Z,       ls=ls, c=color, marker=marker, label=label)
    axs[1].plot(x, fprime_Z,  ls=ls, c=color, marker=marker, label=(r"$\partial_x $" + label) if label is not None else None)
    axs[2].plot(x, fdprime_Z, ls=ls, c=color, marker=marker, label=(r"$\partial^2_x $" + label) if label is not None else None)

    for ax in axs:
        ax.grid(True)
        ax.set_xlim(xmin=x[0],xmax=x[-1])

    if label is not None:
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

    axs[2].set_xlabel(r"$x/%s$" % get_rho_label(run.ncdata))
    #axs[0].set_ylabel(r"$f_Z$")
    #axs[1].set_ylabel(r"$f'_Z$")
    #axs[2].set_ylabel(r"$f''_Z$")

    return fig, axs


def plot_quantity1_quantity2(run, quantities, fig=None, ax=None, ls="--", c=None, marker='.', time_min=0, time_max=99999, time_idx_skip=1, remove_zonals=[False,False], only_zonals=[False,False], avg_norms=[None,None], nx=None, ny=None, species_idx=0, kx_orders=[0,0], ky_orders=[0,0], mult_zeds=[None, None], time_ders=[False, False], mult_vals=[1,1], all_xs=False):

    # Determine time over which to plot
    time_all   = run.ncdata.variables['t'][:]#[::time_idx_skip]
    time_idx_min = nearest_index(time_all-time_min)
    time_idx_max = nearest_index(time_all-time_max)
    time_plot    = time_all[time_idx_min:time_idx_max:time_idx_skip]
    time_idxs = range(time_idx_min, time_idx_max, time_idx_skip)
    assert(len(time_plot)==len(time_idxs))

    if nx is None:
        kx, _, _  = run.get_kx_ky_zed()
        nx = len(kx)

    # Load quantities
    if all_xs:
        f12_t = np.zeros((2, len(time_plot)*nx))
    else:
        f12_t = np.zeros((2, len(time_plot)))

    for i_quantity, quantity in enumerate(quantities):
        kx_order = kx_orders[i_quantity]
        ky_order = ky_orders[i_quantity]
        avg_norm = avg_norms[i_quantity]
        mult_zed = mult_zeds[i_quantity]
        only_zonal = only_zonals[i_quantity]
        remove_zonal = remove_zonals[i_quantity]

        for i_idx, time_idx in enumerate(time_idxs):
            print("Evaluating time_idx %.6i/%i..." % (i_idx+1, len(time_idxs)), end="\r")

            x_der_taken = False
            y_der_taken = False
            if quantity == "phi-phi":
                phi_x_y,  x, y, time_eval = run.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, remove_zonal=True, only_zonal=False, nx=nx, ny=ny, mult_zed=mult_zed)
                f_x_y = phi_x_y**2

            elif quantity == "phi-pressure_perp":
                phi_x_y,  x, y, time_eval = run.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, nx=nx, ny=ny, mult_zed=mult_zed)
                Pprp_x_y,  x, y, time_eval = run.get_quantity_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, nx=nx, ny=ny, mult_zed=mult_zed)
                f_x_y = phi_x_y * Pprp_x_y

            elif quantity == "dyphi-dyPprp":
                dyphi_x_y,  x, y, time_eval = run.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny)
                dyPprp_x_y,  x, y, time_eval = run.get_quantity_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny)
                f_x_y = dyphi_x_y * dyPprp_x_y

            elif quantity == "dxphi-dyPprp":
                dxphi_x_y,  x, y, time_eval = run.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, kx_order=1, nx=nx, ny=ny)
                dyPprp_x_y,  x, y, time_eval = run.get_quantity_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny)
                f_x_y = dxphi_x_y * dyPprp_x_y

            elif quantity == "dyphi-dyphi":
                dyphi_x_y,  x, y, time_eval = run.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny)
                f_x_y = dyphi_x_y**2

            elif quantity == "kx-avg":
                phi_kx_ky,  kx, ky, time_eval = run.get_quantity_kx_ky("phi", time_idx=time_idx, species_idx=species_idx)
                kx_avg = np.sum( kx[None,:,None] * np.abs(phi_kx_ky[:,:,1:])**2, axis=(1,2)) / np.sum( np.abs(phi_kx_ky[:,:,1:])**2, axis=(1,2))

                f_x_y = kx_avg[:,None,None]

            elif quantity == "dxphi-dyphi":
                dxphi_x_y,  x, y, time_eval = run.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, kx_order=1, nx=nx, ny=ny)
                dyphi_x_y,  x, y, time_eval = run.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny)
                f_x_y = dxphi_x_y * dyphi_x_y

            else:
                f_x_y,  x, y, time_eval = run.get_quantity_x_y(quantity=quantity, time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, nx=nx, ny=ny, mult_zed=mult_zed)
                x_der_taken = True
                y_der_taken = True


            # Take derivatives by finite differences if needed
            if not x_der_taken:
                for i in range(kx_order):
                    f_x_y = np.gradient(f_x_y, axis=0)/(x[1]-x[0])
            if not y_der_taken:
                for i in range(ky_order):
                    f_x_y = np.gradient(f_x_y, axis=1)/(y[1]-y[0])

            if all_xs:
                # Average over y
                if avg_norm == "abs":
                    f_x = np.sum( np.abs(f_x_y), axis=1)
                elif avg_norm == 2:
                    f_x = np.sqrt( np.sum( f_x_y**2, axis=1) )
                elif avg_norm == "center":
                    f_x = f_x_y[:,0]
                else:
                    f_x = np.sum( f_x_y , axis=1)
                f12_t[i_quantity, nx*i_idx:nx*(i_idx+1)] = f_x*mult_vals[i_quantity]
 
            else:
                # Average over x-y
                if avg_norm == "abs":
                    f = np.sum( np.abs(f_x_y))
                elif avg_norm == 2:
                    f = np.sqrt( np.sum( f_x_y**2) )
                elif avg_norm == "center":
                    f = f_x_y[0,0]
                elif avg_norm == "zonal_center":
                    f = np.sum(f_x_y[0])
                else:
                    f = np.sum( f_x_y )

                # Save to array
                f12_t[i_quantity, i_idx] = f*mult_vals[i_quantity]

        # Time derivative if required
        dt = np.gradient(time_plot)
        if time_ders[i_quantity]:
            f12_t[i_quantity] = np.gradient(f12_t[i_quantity])/dt
 
    fig, ax = get_or_create_ax(fig, ax, nrows=1, ncols=1, figsize=(12,10))


    if all_xs:
        ax.scatter(f12_t[0], f12_t[1],              marker=marker, s=100, cmap='inferno')
    else:
        ax.plot(f12_t[0], f12_t[1], ls=ls, c=c)
        ax.scatter(f12_t[0], f12_t[1], c=time_plot, marker=marker, s=100, cmap='inferno')
    ax.grid()

    return fig, ax


def plot_quantity_t_k(run, quantity="phi", fig=None, ax=None, remove_zonal=False, ky_idx=None, only_zonal=False, ls=None, lw=None, log_ax=True, t_min=0, t_max=1e6, ratio_zonal_nonzonal=False, kx_min=-1, kx_idxs=None, time_idx_skip=1, species_idx=0, kx_order=0, ky_order=0, eval_real=False, eval_imag=False, colors=None, marker=None, no_plot=False, norm_plot=False, sum_kx=False, labels=None):
    
    ky           = run.ncdata.variables['ky'][:]
    kx           = run.ncdata.variables['kx'][:] 
    time_all = run.get_time_array()
    time_idx_min = nearest_index(time_all-t_min)
    time_idx_max = nearest_index(time_all-t_max)
    time = time_all[time_idx_min:time_idx_max:time_idx_skip]
    dl_over_B_avg = run.dl_over_B_avg()

    if quantity=="phi":
        # phi_vs_t(t, tube, zed, theta0, ky, ri)
        f_t_zed_kx_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max:time_idx_skip,0,:,:,:,:]
    elif quantity=="phi2":
        # phi2_vs_kxky(t, kx, ky)
        phi2_t_kx_ky = run.ncdata.variables['phi2_vs_kxky'][time_idx_min:time_idx_max:time_idx_skip,:,:]
        f_t_zed_kx_ky = phi2_t_kx_ky[:,None,:,:]
    elif quantity=="density":
        # density(t, species, tube, zed, kx, ky, ri)
        f_t_zed_kx_ky_ri = run.ncdata.variables['density'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
    elif quantity=="upar":
        # upar(t, species, tube, zed, kx, ky, ri)
        f_t_zed_kx_ky_ri = run.ncdata.variables['upar'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
    elif quantity=="temperature":
        # temperature(t, species, tube, zed, kx, ky, ri)
        f_t_zed_kx_ky_ri = run.ncdata.variables['temperature'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
    elif quantity=="pressure_par":
        P_t_zed_kx_ky_ri = run.ncdata.variables['pressure'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
        try:
            Pprp_t_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
        except:
            Pprp_t_zed_kx_ky_ri = run.ncdata.variables['pressure_prp'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
        f_t_zed_kx_ky_ri = P_t_zed_kx_ky_ri-0.5*Pprp_t_zed_kx_ky_ri
    elif quantity=="pressure_perp":
        # pressure_perp(t, species, tube, zed, kx, ky, ri)
        try:
            f_t_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
        except:
            f_t_zed_kx_ky_ri = run.ncdata.variables['pressure_prp'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
    elif quantity=="qpar":
        # qpar(t, species, tube, zed, kx, ky, ri)
        f_t_zed_kx_ky_ri = run.ncdata.variables['qpar'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
    elif quantity=="qperp":
        # qperp(t, species, tube, zed, kx, ky, ri)
        f_t_zed_kx_ky_ri = run.ncdata.variables['qperp'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
    elif quantity=="qflx":
        # qflx_kxky(t, species, tube, zed, kx, ky)
        f_t_zed_kx_ky = run.ncdata.variables['qflx_kxky'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:]
    elif quantity=="pflx":
        # pflx_kxky(t, species, tube, zed, kx, ky)
        f_t_zed_kx_ky = np.abs(run.ncdata.variables['pflx_kxky'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:])

    elif quantity in ["Reynolds", "par_mom_transport", "dEZ_par_mom_transport", "pressure_transport"]:

        time_idx_min = nearest_index(time-t_min)
        time_idx_max = nearest_index(time-t_max)
        time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)

        f_t_kx = np.zeros((len(time_idx_eval), int((1+len(kx))/2)), dtype='complex')
        for i_time_idx, time_idx in enumerate(time_idx_eval):
            print("Time idx %4i/%4i" % (i_time_idx, len(time_idx_eval)), end="\r")

            f_zed_kx_ky, _, kx, _, _ = run.get_quantity_zed_kx_ky(quantity=quantity, time_idx=time_idx)

            # Recall int(dy) f(y) = L_y f_{ky=0}/2
            #reynolds_stress_t_kx[i_time_idx] = reynolds_stress_zed_kx_ky[0,:,0]*(y[-1]-y[0])/2
            f_t_kx[i_time_idx] = f_zed_kx_ky[0,:,0]/(2*len(y))

        time = time[time_idx_eval]
        f_t_zed_kx_ky = f_t_kx[:,None,:,None]
        dl_over_B_avg[:] = 1/len(dl_over_B_avg)
        ky_idx = 0

    else:
        print("Did not enter valid quantity to plot (" + str(quantity) + "). Returning")
        return


    # For some quantities, evaluate abs() or real part if desired
    if quantity in ["phi", "density", "upar", "temperature", "pressure_perp", "pressure_par", "qpar", "qperp"]:
        f_t_zed_kx_ky = f_t_zed_kx_ky_ri[:,:,:,:,0] + 1j*f_t_zed_kx_ky_ri[:,:,:,:,1]

    # Filter out ky's now if requested to avoid work in summing
    if only_zonal:
        ky_idx = 0
    if ky_idx is not None and not ky_idx == "abs" and not ky_idx == "SB":
        f_t_zed_kx_ky[:,:,:,:ky_idx] = 0
        f_t_zed_kx_ky[:,:,:,ky_idx+1:] = 0

    # x-derivatives
    f_t_zed_kx_ky = f_t_zed_kx_ky * np.abs(kx[None,None,:,None])**kx_order

    # y-derivatives
    f_t_zed_kx_ky = f_t_zed_kx_ky * np.abs(ky[None,None,None,:])**ky_order

    if kx_idxs is None:
        print("Will plot for all kx")
        kx_idxs = 1e15

    if sum_kx or len(np.shape(kx_idxs)) == 0:
        kx_idxs = [i for i in range(len(kx)) if (np.abs(kx[i]) < kx_idxs and np.abs(kx[i]) > kx_min)]

    kx_plot  = kx[kx_idxs]
    idx_sort = np.argsort(kx_plot)
    kx_sort = kx_plot[idx_sort]
    kx_idxs_sort = np.array(kx_idxs)[idx_sort.astype(int)]
    nx = len(kx_idxs_sort)

    if colors is None:
        colors = sns.color_palette("coolwarm", nx)
    elif len(np.shape(colors)) == 0:
        colors = sns.color_palette(colors, nx)

    f_t_zed_kx_ky = f_t_zed_kx_ky[:,:,kx_idxs_sort]

    ## Take zed average
    f_t_kx_ky = np.sum(dl_over_B_avg[None,:,None,None]*f_t_zed_kx_ky, axis=1)

    if ky_idx is None:
        if ratio_zonal_nonzonal:
            f_t_kx =  f_t_kx_ky[:,:,0] / np.sum(f_t_kx_ky[:,:,1:], axis=2)
        elif remove_zonal:
            f_t_kx = np.sum(f_t_kx_ky[:,:,1:], axis=2)
        elif only_zonal:
            f_t_kx = f_t_kx_ky[:,:,0]
        else:
            f_t_kx = np.sum(f_t_kx_ky, axis=2)
    elif ky_idx == "abs":
        f_t_kx = np.sum(np.abs(f_t_kx_ky), axis=2)
    elif ky_idx == "SB":
        f_t_kx = np.sum(f_t_kx_ky*np.exp(1j*np.pi/2* ky[None,None,:]/ky[-1]), axis=2)
    else:
        f_t_kx = f_t_kx_ky[:,:,ky_idx]

    if sum_kx:
        kx_idxs_sort = [0]
        f_t_kx[:,0]  = np.sum(f_t_kx, axis=1)
        f_t_kx[:,1:] = 0

 
    if not no_plot:
        fig, ax = get_or_create_ax(fig, ax, nrows=1, ncols=1, figsize=(12,9))

        for i_kx, kx_idx in enumerate(kx_idxs_sort):

            if eval_real:
                f_t_plot = np.real(f_t_kx[:,i_kx])
            elif eval_imag:
                f_t_plot = np.imag(f_t_kx[:,i_kx])
            else:
                f_t_plot = np.abs(f_t_kx[:,i_kx])

            if norm_plot:
                f_t_plot = f_t_plot/(np.abs(f_t_plot).max())
    
            if labels is not None:
                if len(labels) == len(kx_idxs_sort):
                    label = labels[i_kx]
                elif labels == "firstlast":
                    if i_kx == 0 or i_kx == len(kx_idxs_sort)-1:
                        label = r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx_sort[i_kx])
                    else:
                        label = None
                elif labels == "minlast":
                    #if kx_sort[i_kx] == np.abs(kx_sort).min() or i_kx == len(kx_idxs_sort)-1 or i_kx==0:
                    if kx_sort[i_kx] == np.abs(kx_sort[np.abs(kx_sort)>0]).min() or i_kx == len(kx_idxs_sort)-1:
                        label = r"$k_x %s = %.3f$" % (get_rho_label(run.ncdata), kx_sort[i_kx])
                    else:
                        label = None
                else:
                    label = None
            else:
                label = None

            if log_ax:
                ax.semilogy(time, np.abs(f_t_plot),label=label, ls=ls, c=colors[i_kx], lw=lw, marker=marker)
            else:
                ax.plot(time, f_t_plot, label=label, ls=ls, c=colors[i_kx], lw=lw, marker=marker)

        ax.set_xlabel(r"$t$")

        if labels is not None:
            ax.legend()

    return fig, ax, time, kx_sort, f_t_kx


def plot_phi_t_ky(run, fig=None, ax=None, zed_idx=None, remove_zonal=False, only_zonal=False, label=None, ls=None, c=None, lw=None, log_ax=True, norm_to_t0=False, plot_abs=True, t_max=np.inf, time_avg=1, norm_kperp2=False, ratio_zonal_nonzonal=False):
    

    # phi_vs_t(t, tube, zed, theta0, ky, ri)
    phi_vs_t  = run.ncdata.variables['phi_vs_t'][:,0,:,:,:,:]
    zed       = run.ncdata.variables['zed'][:]
    ky        = run.ncdata.variables['ky'] 
    kx        = run.ncdata.variables['kx'] 
    time      = run.ncdata.variables['t'][:]

    dl_over_B_avg = run.dl_over_B_avg()
     
    # if zed_idx = None, average over tube
    if zed_idx is None:
        phi_t_ky = np.zeros(shape=(len(time),len(ky)))
        for i_zed in range(len(zed)):
            if plot_abs:
                phi_t_ky = phi_t_ky + np.sum( np.abs(phi_vs_t[:,i_zed,:,:,0]+1j*phi_vs_t[:,i_zed,:,:,1])**2, axis=1)*dl_over_B_avg[i_zed]
            else:
                phi_t_ky = phi_t_ky + np.sum( phi_vs_t[:,i_zed,:,:,0], axis=1)*dl_over_B_avg[i_zed]
            #phi_t_ky = phi_t_ky + np.sum( np.real(phi_vs_t[:,i_zed,:,:,0]+1j*phi_vs_t[:,i_zed,:,:,1]), axis=1)*dl_over_B_avg[i_zed]
    else:
        if plot_abs:
            phi_t_ky = np.sum( np.abs(phi_vs_t[:,zed_idx,:,:,0]+1j*phi_vs_t[:,zed_idx,:,:,1])**2, axis=1)
        else:
            phi_t_ky = np.sum( dl_over_B_avg[None,:,None,None]*phi_vs_t[:,zed_idx,:,:,0], axis=1)
        #phi_t_ky = np.sum( np.real(phi_vs_t[:,zed_idx,:,:,0]+1j*phi_vs_t[:,zed_idx,:,:,1]), axis=1)

    # Filter zonal if requested
    if ratio_zonal_nonzonal:
        phi_t =  phi_t_ky[:,0] / np.sum(phi_t_ky[:,1:], axis=1)
    elif remove_zonal:
        phi_t = np.sum(phi_t_ky[:,1:], axis=1)
    elif only_zonal:
        phi_t = phi_t_ky[:,0]
    else:
        phi_t = np.sum(phi_t_ky, axis=1)

    # Only keep t<=tmax
    phi_t = phi_t[time < t_max]
    time  = time[time < t_max]

    phi_end = np.mean(phi_t[time > max(0, time[-1]-time_avg)])
    # Normalise by flux-tube averaged kperp2 if desired
    if norm_kperp2:
        kperp2 = run.get_avg_kperp2()
        print(run.input_file + ": <kperp2> = %e" % (kperp2))
        phi_t = phi_t/kperp2

    # Plot
    fig, ax = get_or_create_ax(fig, ax, nrows=1, ncols=1, figsize=(12,9))

    if norm_to_t0:
        phi_t = phi_t / phi_t[0]
    if log_ax:
        ax.semilogy(time, np.abs(phi_t),label=label, ls=ls, c=c, lw=lw)
    else:
        ax.plot(time, phi_t,label=label, ls=ls, c=c, lw=lw)

    ax.set_xlabel(r"$t %s/a$" % get_vt_label(run.ncdata))
    if norm_to_t0 and plot_abs:
        label = r"$|\varphi(t)/\varphi(t=0)|^2$"
    elif not norm_to_t0 and plot_abs:
        label = r"$|\varphi(t)|^2$"
    elif norm_to_t0 and not plot_abs:
        label = r"$\varphi(t)/\varphi(t=0)$"
    else:
        label = r"$\varphi(t)$"

    if norm_kperp2:
        label = label + r"$/\langle (k_\perp %s)^2\rangle$" % get_rho_label(run.ncdata)
    ax.set_ylabel(label)

    ax.legend()
    ax.grid()

    return fig, ax, phi_end
