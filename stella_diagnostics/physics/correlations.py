"""Parallel (field-line-following) correlation function diagnostics, plus generic 1D/2D correlation-function utilities."""

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
from stella_diagnostics.io.cache import cached
from stella_diagnostics.io.codes import get_rho_label
from stella_diagnostics.plotting.mpl_helpers import get_or_create_ax


def _periodic_shift_correlation(f_zed_x_y, dl_over_B_avg, axis, sum_other):
    """1D periodic-shift (O(N^2)) correlation of f_zed_x_y (shape
    zed,x,y) along `axis` (1 for x, 2 for y), dl_over_B_avg-weighted and
    summed over zed (and, if sum_other, also over the OTHER spatial
    axis) -- the double loop shared, up to an x<->y axis swap, between
    the x and y halves of get_perp_correlation_function.
    """
    N = f_zed_x_y.shape[axis]
    other_axis = 2 if axis == 1 else 1

    if sum_other:
        norm = np.sum(f_zed_x_y**2 * dl_over_B_avg[:, None, None])
    else:
        idx0 = [slice(None)] * 3
        idx0[other_axis] = 0
        norm = np.sum(f_zed_x_y[tuple(idx0)]**2 * dl_over_B_avg[:, None])

    f_corr = np.zeros(N)
    for i in range(N):
        for i_corr in range(N):
            idx2 = (i + i_corr) % N
            sl_i = [slice(None)] * 3
            sl_i[axis] = i
            sl_idx2 = [slice(None)] * 3
            sl_idx2[axis] = idx2
            if sum_other:
                f_corr[i_corr] += np.sum(f_zed_x_y[tuple(sl_i)] * f_zed_x_y[tuple(sl_idx2)] * dl_over_B_avg[:, None])
            else:
                sl_i[other_axis] = 0
                sl_idx2[other_axis] = 0
                f_corr[i_corr] += np.sum(f_zed_x_y[tuple(sl_i)] * f_zed_x_y[tuple(sl_idx2)] * dl_over_B_avg)

    return f_corr / norm


@cached(version=2)
def get_perp_correlation_function(run, quantity="phi", remove_zonal=True, time_idx=-1, sum_other=True):
    """(x_corr, f_corr_x, y_corr, f_corr_y): the real-space perpendicular
    (x and y separately) correlation functions of `quantity` at one time
    snapshot -- a periodic-shift correlation, dl_over_B_avg-weighted and
    summed over zed (and, if sum_other, also over the other of x/y).

    Extracted from example_plots/plot_correlation_func_perp.py's inline
    double loop -- a different computation from
    plot_parallel_correlation_function (field-line-following, k-space)
    and get_correlation_func_1D/2D (generic, take arbitrary arrays, not
    the specific zed+dl_over_B_avg-weighted average this one does) in
    this same module.
    """
    f_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y(quantity, remove_zonal=remove_zonal, time_idx=time_idx)
    dl_over_B_avg = run.dl_over_B_avg()

    x_corr = x - x[0]
    f_corr_x = _periodic_shift_correlation(f_zed_x_y, dl_over_B_avg, axis=1, sum_other=sum_other)

    y_corr = y - y[0]
    f_corr_y = _periodic_shift_correlation(f_zed_x_y, dl_over_B_avg, axis=2, sum_other=sum_other)

    return x_corr, f_corr_x, y_corr, f_corr_y


def plot_parallel_correlation_function(run, quantity="phi", time_idx=-1, time_avg=0, fig=None, ax=None, zeta_max=False, k_min=None, k_max=None, no_plot=False, kx_instead_of_ky=False, keep_only_zonal=False, vmin=None, vmax=None):

    time = run.get_time_array(GX_big=True)
    time_max = time[time_idx]
    time_idx_min = nearest_index(time-(time_max-time_avg))-3

    quantity_name = quantity
    if quantity_name == "phi":
        if run.code == "stella":
            # phi_vs_t(t, tube, zed, theta0, ky, ri)
            quantity_data = np.mean(run.ncdata.variables['phi_vs_t'][time_idx_min:time_idx,0,:,:,:,:],axis=0) # zed-kx-ky-ri
        elif run.code == "GX":
            if run.GX_old_version:
                print("WARNING! You are loading Phi_z in GX, which had issues in early code versions.")
                quantity_data = np.transpose( run.ncdata['Special']['Phi_z'], axes=[2,1,0,3] )
            else:
                quantity_data = np.transpose( np.mean(run.ncdata_big['Diagnostics']['Phi'][time_idx_min:time_idx], axis=0) , axes=[2,1,0,3])
        elif run.code == "GS2":
            quantity_data = np.transpose( run.ncdata.variables['phi'] , axes=[2,1,0,3] )

    elif quantity_name == "temperature":
        # temperature(t, species, tube, zed, kx, ky, ri)
        quantity_data = np.mean(run.ncdata.variables['temperature'][time_idx_min:time_idx,0,0,:,:,:,:],axis=0) # zed-kx-ky-ri

    elif quantity_name == "upar":
        quantity_data = np.mean(run.ncdata.variables['upar'][time_idx_min:time_idx,0,0,:,:,:,:],axis=0) # zed-kx-ky-ri
    else:
        print("ENTER VALID QUANTITY!")
        return

    if keep_only_zonal:
        quantity_data[:,:,1:,:] = 0
        quantity_data[:,0,0,:] = 0
    else:
        quantity_data[:,:,0,:] = 0

    kx, ky, zed = run.get_kx_ky_zed()

    if run.code == "GX" and run.GX_old_version:
        quantity_data[:,kx>0,:,:] = 0

    assert(np.shape(quantity_data)[0] == len(zed))
    assert(np.shape(quantity_data)[1] == len(kx))
    assert(np.shape(quantity_data)[2] == len(ky))
    assert(np.shape(quantity_data)[3] == 2)

    if zeta_max:
        # Find zed where phi peaks
        quantity_sum = np.sum(np.abs(quantity_data[:,:,:,0]+1j*quantity_data[:,:,:,1])**2, axis=(1,2))
        arg_zed_ctr = np.argmax(quantity_sum)
        print("zeta(phi=phi_max)/zeta_max = %e" % (zed[arg_zed_ctr]/np.max(zed)))
    else:
        # Find zed=0
        arg_zed_ctr = nearest_index(zed)

    if kx_instead_of_ky:
        idx_kx_sort = np.argsort(kx)
        k = kx[idx_kx_sort]
        k_other = ky
        quantity_tmp = quantity_data[:,idx_kx_sort]
        quantity_data = np.transpose(quantity_tmp, (0,2,1,3))
    else:
        k = ky
        k_other = kx
    correlation_func = np.zeros(shape=(len(zed), len(k)))

    for i_k in range(len(k)):
        f_C_k_ctr = quantity_data[arg_zed_ctr,:, i_k,0] + 1j*quantity_data[arg_zed_ctr,:,i_k,1]
        for i_zed in range(len(zed)):

            f_C_k = quantity_data[i_zed,:,i_k,0] + 1j*quantity_data[i_zed,:,i_k,1]

            if k_min is None:
                k_min = 0
            if k_max is None:
                k_max = np.inf

            idx_k = np.where( (np.abs(k_other)>k_min) & (np.abs(k_other)<k_max))
            #idx_k = np.abs(k_other) > k_min
            correlation_func[i_zed, i_k] = np.sum( np.real( f_C_k[idx_k] * np.conj(f_C_k_ctr[idx_k]))) / np.sum( np.abs(f_C_k_ctr[idx_k])**2 )
                #correlation_func[i_zed, i_k] = np.sum( np.real( f_C_k * np.conj(f_C_k_ctr))) / np.sum( np.abs(f_C_k_ctr)**2 )

    if not no_plot:
        fig, ax = get_or_create_ax(fig, ax, nrows=1, ncols=1, figsize=(12,9))

        X, Y = np.meshgrid(zed, k)
        Z = correlation_func.T

        im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)

        ax.set_xlabel(r"$\Delta \zeta$")
        if kx_instead_of_ky:
            ax.set_ylabel(r"$k_x %s$" % get_rho_label(run.ncdata))
        else:
            ax.set_ylabel(r"$k_y %s$" % get_rho_label(run.ncdata))
        ax.set_title(r"$\mathcal{C}$")
    else:
        fig = None
        ax = None
        im = None

    # Evaluate average delta-chi
    avg_delta_chi = np.zeros_like(k)
    for i_k in range(len(k)):
        avg_delta_chi[i_k] = np.mean(correlation_func[:, i_k])

    return fig, ax, im, avg_delta_chi, k


def get_parallel_correlation_function_kx_ky(run, quantity="phi", time_idx=-1, zeta_max=False, k_min=None):

    if quantity == "phi":
        # phi_vs_t(t, tube, zed, theta0, ky, ri)
        phi_vs_t = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:] # zed-kx-ky-ri
    elif quantity == "temperature":
        # temperature(t, species, tube, zed, kx, ky, ri)
        phi_vs_t = run.ncdata.variables['temperature'][time_idx,0,0,:,:,:,:]

    zed      = run.ncdata.variables['zed'][:]
    ky       = run.ncdata.variables['ky'][:]
    kx       = run.ncdata.variables['kx'][:] 

    if zeta_max:
        # Find zed where phi peaks
        phi_sum = np.sum(np.abs(phi_vs_t[:,:,:,0]+1j*phi_vs_t[:,:,:,1])**2, axis=(1,2))
        arg_zed_ctr = np.argmax(phi_sum)
        print("zeta(phi=phi_max)/zeta_max = %e" % (zed[arg_zed_ctr]/np.max(zed)))
    else:
        # Find zed=0
        arg_zed_ctr = nearest_index(zed)

    idx_kx_sort = np.argsort(kx)
    kx = kx[idx_kx_sort]
    phi_vs_t = phi_vs_t[:,idx_kx_sort]

    correlation_func_zed_kx_ky = np.zeros(shape=(len(zed), len(kx), len(ky)))

    phi_C_k_ctr  = phi_vs_t[arg_zed_ctr,:,:,0] + 1j*phi_vs_t[arg_zed_ctr,:,:,1]

    for i_zed in range(len(zed)):
        phi_C_k_delt = phi_vs_t[i_zed,:,:,0] + 1j*phi_vs_t[i_zed,:,:,1]
        correlation_func_zed_kx_ky[i_zed] = np.abs(phi_C_k_delt)/np.abs(phi_C_k_ctr)
        #correlation_func_zed_kx_ky[i_zed] = np.real(phi_C_k_ctr*np.conj(phi_C_k_delt))#/np.abs(phi_C_k_ctr)**2

    # Evaluate average delta-chi
    avg_delta_chi = np.mean(correlation_func_zed_kx_ky, axis=0)

    time  = run.ncdata.variables['t'][time_idx]
    return correlation_func_zed_kx_ky, avg_delta_chi, kx, ky, time


def get_correlation_func_1D(x, y, ref_point="middle", dx_max=None, Nr_dx=10, Nr_x_ref=10):
    assert(len(x)==len(y))

    if ref_point=="middle":
        corr_func = np.zeros_like(x)
        idx_mid = int(len(x)/2)
        y_mid   = y[idx_mid]
        mult_mid = np.conj(y_mid)/np.abs(y_mid)**2
        Delta_x   = x-x[idx_mid]

        for i in range(len(x)):
            corr_func[i] = np.real(y[i]*mult_mid)

    elif ref_point=="avg":
        # Interpolate in case data is not equally spaced
        y_interp_real = interp(x, np.real(y))
        y_interp_imag = interp(x, np.imag(y))

        # Determine dx_max and ensure it is lower than 1/2 length of data
        if dx_max is None:
            dx_max = (x[-1]-x[0])/2

        Delta_x = np.linspace(-dx_max, dx_max, Nr_dx, endpoint=True)
        corr_func = np.zeros_like(Delta_x)

        for i_dx, dx in enumerate(Delta_x):

            xmin_ref = max( x[0],  x[0] -dx)
            xmax_ref = min( x[-1], x[-1]-dx)
            xvals_ref = np.linspace(xmin_ref*1.001, xmax_ref*0.999, Nr_x_ref)

            y_interp_xvals_ref = y_interp_real(xvals_ref) + 1j*y_interp_imag(xvals_ref)
            y_interp_dx        = y_interp_real(xvals_ref+dx) + 1j*y_interp_imag(xvals_ref+dx)
            norm = np.mean( np.abs(y_interp_xvals_ref)**2 )
            corr_func[i_dx] = np.mean( np.real(y_interp_xvals_ref * np.conj(y_interp_dx))) / norm

    return Delta_x, corr_func


def get_correlation_func_2D(x1, x2, y, idx_ref1=None, idx_ref2=None, ref_point="middle", x2_window=None):
    assert(len(x1)==np.shape(y)[0])
    assert(len(x2)==np.shape(y)[1])
    corr_func = np.zeros_like(y)

    if ref_point=="single":
        if idx_ref1 is None:
            idx_ref1 = int(len(x1)/2)
        if idx_ref2 is None:
            idx_ref2 = int(len(x2)/2)
        y_ref   = y[idx_ref1, idx_ref2]
        mult_ref = np.conj(y_ref)/np.abs(y_ref)**2
        
        Delta_x1  = x1-x1[idx_ref1]
        Delta_x2  = x2-x2[idx_ref2]
        
        for i1 in range(len(x1)):
            for i2 in range(len(x2)):
                corr_func[i1, i2] = np.real(y[i1,i2]*mult_ref)

    elif ref_point=="avg1":
        # Note: assumes equally spaced data in first index, and periodic (e.g. x or y)
        if idx_ref2 is None:
            idx_ref2 = int(len(x2)/2)

        idx_mid1  = int(len(x1)/2)+1
        Delta_x1  = x1-np.mean(x1)
        Delta_x2  = x2-x2[idx_ref2]
        corr_func = np.zeros((len(x1), len(x2)))
        yvals_ref = y[:,idx_ref2]
        norm_ref  = np.mean( np.abs(yvals_ref)**2 )

        for i_Delta_x1 in range(len(Delta_x1)):
            idxs_1 = ( idx_mid1 + np.arange(len(Delta_x1)) + i_Delta_x1) % len(Delta_x1)
            for i_Delta_x2 in range(len(Delta_x2)):
                corr_func[i_Delta_x1, i_Delta_x2] = np.mean( np.real(y[idxs_1, i_Delta_x2]*np.conj(yvals_ref)) ) / norm_ref


    elif ref_point=="avg":
        # Note: assumes equally spaced data, and periodic in first index
        idx_mid1  = int(len(x1)/2)+1
        idx_mid2  = int(len(x2)/2)+1
        Delta_x1  = x1-np.mean(x1)
        Delta_x2  = x2-x2[0]
        idxs2 = np.arange(len(Delta_x2))
        idxs2_window = idxs2[Delta_x2<=x2_window]
        idxs2_ref    = idxs2[Delta_x2<=Delta_x2[-1]-x2_window]
        #print(len(idxs2))
        #print(len(idxs2_window))
        #print(len(idxs2_ref))
        assert(len(idxs2)>=len(idxs2_window)+len(idxs2_ref))
        corr_func = np.zeros((len(x1), len(idxs2_window)))


        for i_Delta_x1 in range(len(Delta_x1)):
            idxs_1 = ( idx_mid1 + np.arange(len(Delta_x1)) + i_Delta_x1) % len(Delta_x1)

            for i_Delta_x2 in range(len(idxs2_window)):
                corr_func[i_Delta_x1, i_Delta_x2] = np.sum( np.real(y[idxs_1[:,None],idxs2_ref[None,:]+i_Delta_x2]*np.conj(y[:,idxs2_ref])) ) / np.sum( np.abs(y[:,idxs2_ref])**2 )

        Delta_x2 = x2[idxs2_window]-x2[0]

    return Delta_x1, Delta_x2, corr_func
