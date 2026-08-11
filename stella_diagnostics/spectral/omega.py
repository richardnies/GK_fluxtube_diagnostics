"""Complex frequency (omega = omega_r + i*gamma) reading and growth-rate extraction, including Laplace-transform-based and omega-filtering approaches."""

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


def read_data_omega_k(run, timestep=-1, om_avg=True, check_convergence=True, nonconverged_to_none=True, delta_t_avg=None, t_val=None):
    kx   = run.ncdata.variables['kx'][:]
    ky   = run.ncdata.variables['ky'][:]
    dim_kx = len(kx)
    dim_ky = len(ky)

    # omega is in format [time ky kx Re[om] Im[om] Re[omavg] Im[omavg]]
    # omega_data has dim (N_time)*(N_ky)*(N_kx)*(7)
    if run.code == "stella":
        omega_data = np.loadtxt(run.omega_file, dtype='float').reshape(-1, dim_ky, dim_kx, 7)
        time_all = omega_data[:,0,0,0]

    elif run.code == "GX":
        #omega_v_time(time, ky, kx, ri)
        omega_v_time = run.ncdata['Special']['omega_v_time']

        time_all = run.get_time_array()
        kx, ky, zed = run.get_kx_ky_zed()

        omega_data = np.zeros( (len(time_all), len(ky), len(kx), 7) )
        omega_data[:,:,:,0] = time_all[:,None,None]
        omega_data[:,:,:,1] = ky[None,:,None]
        omega_data[:,:,:,2] = kx[None,None,:]
        omega_data[:,:,:,3] = omega_v_time[:,:,:,0]
        omega_data[:,:,:,4] = omega_v_time[:,:,:,1]

        om_avg = False

    # Make sure t_val is smaller or equal to the maximal time
    if t_val is not None:
        t_val = min(t_val, time_all[-1])

    if delta_t_avg is None:
        if t_val is None:
            omega_slice = omega_data[timestep]
        else:
            omega_slice = omega_data[np.argmin( np.abs(time_all - t_val) )]
    else:
        if t_val is None:
            omega_slice = np.mean(omega_data[ np.logical_and(time_all > time_all[timestep]-delta_t_avg, time_all <= time_all[timestep])], axis=0)
        else:
            omega_slice = np.mean(omega_data[ np.logical_and(time_all > t_val-delta_t_avg, time_all <= t_val)], axis=0)

 
    time     = omega_slice[:,:,0]
    ky       = omega_slice[:,:,1]
    kx       = omega_slice[:,:,2]
    if om_avg:
        omega_r = omega_slice[:,:,5]
        omega_i = omega_slice[:,:,6]
    else:
        omega_r = omega_slice[:,:,3]
        omega_i = omega_slice[:,:,4]


    if delta_t_avg is None and check_convergence and len(kx) == len(ky) == 1:
        omega_r_prev = omega_data[timestep-1][0][0][5]
        omega_i_prev = omega_data[timestep-1][0][0][6]

        diff_omega_r = np.abs( (omega_r - omega_r_prev)/omega_r )
        diff_omega_i = np.abs( (omega_i - omega_i_prev)/omega_i )

        threshold = 1e-1

        if diff_omega_r > threshold:
            print(run.filename_base + ": average omega_r = %e evolved by %.3f > threshold = %.3f in last step." % (omega_r, diff_omega_r, threshold))
            if nonconverged_to_none:
                omega_r[:] = np.nan

        if diff_omega_i > threshold:
            print(run.filename_base + ": average omega_i = %e evolved by %.3f > threshold = %.3f in last step." % (omega_i, diff_omega_i, threshold))
            if nonconverged_to_none:
                omega_i[:] = np.nan

    return time, ky, kx, omega_r, omega_i


def read_omega_t(run, delta_t_avg=None):
    omega_data = np.loadtxt(run.omega_file)
    Nr_timesteps = len(omega_data)

    time    = np.zeros(Nr_timesteps)
    omega_r = np.zeros(Nr_timesteps)
    omega_i = np.zeros(Nr_timesteps)

    for i in range(Nr_timesteps):

        time[i], _, _, omega_r[i], omega_i[i] = run.read_data_omega_k(timestep=i, check_convergence=False, delta_t_avg=delta_t_avg)

    return time, omega_r, omega_i


def get_quantity_omega_zed_kx(run, quantity, time_min, time_max, time_idx_skip=1, species_idx=0, remove_zonal=False, only_zonal=False, kx_order=0, omega_min=-np.inf, omega_max=np.inf, alt_slow_eval=True): 

    time_all  =  run.ncdata.variables['t'][:]
    time_idx_min = nearest_index(time_all - time_min)
    time_idx_max = nearest_index(time_all - time_max)
    time_idxs = range(time_idx_min, time_idx_max, time_idx_skip)
    time = time_all[time_idxs]
    assert(len(time) > 0)
    assert(len(time_idxs) == len(time))

    # Obtain quantity as a function of time
    for i_idx, time_idx in enumerate(time_idxs):
        print("Quantity = " + quantity + ": evaluating time_idx %.6i/%i..." % (i_idx+1, len(time_idxs)), end="\r")
        f_zed_kx_ky, zed, kx, ky, time_eval = run.get_quantity_zed_kx_ky(quantity=quantity, time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, alt_slow_eval=alt_slow_eval)
        if i_idx == 0:
            f_t_zed_kx = np.zeros((len(time),len(zed),len(kx)), dtype='complex')
        f_t_zed_kx[i_idx] = f_zed_kx_ky[:,:,0]

    # Resample to equal time-intervals
    dt = (np.gradient(time)).max()
    time_interp = np.arange(time[0], time[-1], dt)
    f_interp = interp(time, f_t_zed_kx, assume_sorted=True, axis=0)
    f_t_zed_kx_interp = f_interp(time_interp)

    # Fourier transform time to omega
    f_omega_zed_kx = np.fft.fft(f_t_zed_kx_interp, axis=0)
    omega = np.fft.fftfreq(len(time_interp), d=dt)*(2*np.pi)

    # Filter out omega outside range
    f_omega_zed_kx = f_omega_zed_kx[(omega<omega_max) & (omega>omega_min)]
    omega  = omega[(omega<omega_max) & (omega>omega_min)]

#        idx_sort = np.argsort(omega)
#        f_omega_zed_kx = f_omega_zed_kx[idx_sort]
#        omega = omega[idx_sort]

    return f_omega_zed_kx, omega, zed, kx


def get_quantity_filtered_in_omega(run, f_t, time, omega_min=-np.inf, omega_max=np.inf):

    # Resample to equal time-intervals
    dt = np.max(np.gradient(time))
    time_interp = np.arange(time[0], time[-1], dt)
    f_interp = interp(time, f_t, assume_sorted=True, axis=0)
    f_t_interp = f_interp(time_interp)

    # Fourier transform time to omega
    f_omega = np.fft.fft(f_t_interp, axis=0)
    omega = np.fft.fftfreq(len(time_interp), d=dt)*(2*np.pi)

    # Filter out omega outside range
    f_omega = f_omega[(omega<=omega_max) & (omega>=omega_min)]
    omega   =   omega[(omega<=omega_max) & (omega>=omega_min)]
    #return f_omega, omega

    # Transform back to real space
    f_t_filtered = np.fft.ifft(f_omega, axis=0)
    time_new = np.linspace(time[0], time[0]+2*np.pi/omega[1], len(omega))

    return f_t_filtered, time_new


def plot_quantity_kx_omega(run, quantity, time_min, time_max, time_idx_skip=1, fig=None, ax=None, vmin=None, vmax=None, species_idx=0, logarithmic=False, remove_zonal=False, only_zonal=False, cmap='inferno', kx_order=0, par_der_order=0, mult_zed=None, zed_val=None, no_plot=False, omega_min=-np.inf, omega_max=np.inf, time_der=False, plot_omega2_kx2=False, mean_delt_zed=None, alt_slow_eval=False, append_mirror=False, normalise_each_kx=False, omega_norm=1, scale_eps=1):

    kx, ky, zed = run.get_kx_ky_zed()
    time_all    = run.get_time_array(GX_big=True)
    dl_over_B_avg = run.dl_over_B_avg()
    time_idx_min = nearest_index(time_all - time_min)
    time_idx_max = nearest_index(time_all - time_max)
    time_idxs = range(time_idx_min, time_idx_max, time_idx_skip)
    time = time_all[time_idxs]

    assert(len(time) > 0)
    assert(len(time_idxs) == len(time))

    for i_idx, time_idx in enumerate(time_idxs):
        print("Evaluating time_idx %.6i/%i..." % (i_idx+1, len(time_idxs)), end="\r")
        f_kx_ky, kx, ky, _ = run.get_quantity_kx_ky(quantity=quantity, remove_zonal=remove_zonal, only_zonal=only_zonal, time_idx=time_idx, kx_order=kx_order, mult_zed=mult_zed, par_der_order=par_der_order, mean_delt_zed=mean_delt_zed, alt_slow_eval=alt_slow_eval, zed_val=zed_val)
        if i_idx == 0:
            f_kx_ky_t = np.zeros((len(kx), len(ky), len(time)), dtype=np.complex128)
        f_kx_ky_t[:,:,i_idx] = f_kx_ky

    # Add mirror of sample if required
    if append_mirror:
        time = np.concatenate((time, time+time[-1]+time[1]-time[0]))
        f_kx_ky_t = np.concatenate((f_kx_ky_t, f_kx_ky_t[:,:,::-1]), axis=2)

    # Resample to equal time-intervals
    dt = (np.gradient(time)).max()
    time_interp = np.arange(time[0], time[-1], dt)
    f_interp = interp(time, f_kx_ky_t, assume_sorted=True, axis=2)
    f_kx_ky_t_interp = f_interp(time_interp)

    # Take time derivative if required
    if time_der:
        f_kx_ky_t_interp = np.gradient(f_kx_ky_t_interp, axis=2)/dt

    # Fourier transform time to omega
    #f_kx_ky_omega = np.fft.fft(f_kx_ky_t, axis=2)
    #omega = np.fft.fftfreq(len(time))
    f_kx_ky_omega = np.fft.fft(f_kx_ky_t_interp, axis=2)
    omega = -np.fft.fftfreq(len(time_interp), d=dt)*(2*np.pi)

    # Take care of ky
    if only_zonal:
        f_kx_omega = f_kx_ky_omega[:,0,:]
    else:
        # Summing over ky
        print("Note! Summing over ky")
        f_kx_omega = np.sum(f_kx_ky_omega, axis=1)

    f_kx_omega     = f_kx_omega[:, (omega<=omega_max) & (omega>=omega_min)]
    omega  = omega[(omega<=omega_max) & (omega>=omega_min)]

 
    if normalise_each_kx:
        for i_kx in range(len(kx)):
            norm = (np.abs(f_kx_omega[i_kx,:])).max()
            if norm != 0:
                f_kx_omega[i_kx,:] = f_kx_omega[i_kx,:]/norm

    ## Find peak at larger kx
    #f_kx_omega_subset = f_kx_omega[np.abs(kx) > 0.4]
    #kx_subset = kx[np.abs(kx) > 0.4]
    #idx_max = np.argmax( f_kx_omega_subset )
    #idx_max = np.unravel_index(idx_max, f_kx_omega_subset.shape)
    #print("kx = %.2e, omega = %.2e, omega/kx = %.2e at maximum (kx > 0.4)." % (kx_subset[idx_max[0]], omega[idx_max[1]], omega[idx_max[1]]/kx_subset[idx_max[0]]))

    if not no_plot:
        #Ascending order
        idx_omega = np.argsort(omega)
        idx_kx    = np.argsort(kx)
        omega = omega[idx_omega]
        kx    = kx[idx_kx]
        f_kx_omega = f_kx_omega[idx_kx,:][:,idx_omega]

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,10))
            plt.subplots_adjust(left=0.15,right=0.95)

        # Rescale with R/a fac
        omega_norm = omega_norm*scale_eps
        f_kx_omega = f_kx_omega/scale_eps
        
        # Plot
        Z    = np.abs(f_kx_omega)
        #Z    = f_kx_omega

        if vmax is None:
            vmax = Z[1:,:].max()
        if logarithmic and vmin is not None:
            vmin = vmin*vmax
        if vmin is None:
            vmin = Z[1:,:].min()
        print("vmin = %e" % (vmin))
        print("vmax = %e" % (vmax))


        if not plot_omega2_kx2:
            X, Y = np.meshgrid(kx, omega/omega_norm)
            dkx = kx[1]-kx[0]
            dom = (omega[1]-omega[0])/omega_norm
            if logarithmic:
                im = ax.imshow(Z.T, norm=colors.LogNorm(vmin=vmin, vmax=vmax), interpolation='nearest', cmap=cmap, extent=[kx.min()-dkx/2, kx.max()-dkx/2, omega.min()/omega_norm-dom/2, omega.max()/omega_norm-dom/2], aspect='auto', origin='lower')
                ax.set_xlim([kx.min()-dkx/2, kx.max()+dkx/2])
                ax.set_ylim([omega.min()/omega_norm-dom/2, omega.max()/omega_norm-dom/2])
                #im = ax.pcolormesh(X, Y, Z.T, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
            else:
                im = ax.pcolormesh(X, Y, Z.T, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
            ax.set_xlabel(r"$k_x \rho_i$")
            if scale_eps == 1:
                ax.set_ylabel(r"$\omega a/v_{Ti}$")
            else:
                ax.set_ylabel(r"$\omega R/v_{Ti}$")
        else:
            X, Y = np.meshgrid(kx**2, (omega/omega_norm)**2)
            if logarithmic:
                im = ax.pcolormesh(X, Y, Z.T, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
            else:
                im = ax.pcolormesh(X, Y, Z.T, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
            ax.set_xlabel(r"$(k_x \rho_i)^2$")
            if scale_eps:
                ax.set_ylabel(r"$(\omega R/v_T)^2$")
            else:
                ax.set_ylabel(r"$(\omega a/v_T)^2$")

    else:
        im = None

    return fig, ax, im, kx, omega, f_kx_omega


def get_avg_stddev_timetrace(time, quantity, timeavg, timemax=None):
    if timemax is None:
        timemax = max(time)
    timemin = max(0, timemax-timeavg)

    quantity_to_avg = quantity[(timemin<time) & (time<timemax)]
    time_to_avg     = time[(timemin<time) & (time<timemax)]
    if len(time_to_avg) < 2:
        time_to_avg     = time[-2:]
        quantity_to_avg = quantity[-2:]

    Delta_t = np.gradient(time_to_avg)

    quantity_mean = np.mean(quantity_to_avg*Delta_t)/np.mean(Delta_t)
    quantity_std  = np.sqrt( np.mean( Delta_t*(quantity_to_avg-quantity_mean)**2 )/np.mean(Delta_t) )

    return quantity_mean, quantity_std


def get_convergence_quantity(time, quantity, timeavg_array, timemax=None):
    quantity_avg = np.zeros_like(timeavg_array)
    quantity_std = np.zeros_like(timeavg_array)
    for i_avg, timeavg in enumerate(timeavg_array):
        quantity_avg[i_avg], quantity_std[i_avg] = get_avg_stddev_timetrace(time, quantity, timeavg, timemax)

    return quantity_avg, quantity_std


def plot_convergence_quantity(time, quantity, timeavg_array, timemax=None, fig=None, ax=None, c=None, marker=None, ls=None, label=None):

    if ax is None:
       fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    quantity_avg, quantity_std = get_convergence_quantity(time, quantity, timeavg_array, timemax)

    ax.errorbar(timeavg_array, quantity_avg, yerr=quantity_std, label=label, c=c, marker=marker, ls=ls)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_xlim(xmin=0)

    return fig, ax


def extract_growth_rate(time, quantity):

    assert(len(time)==len(quantity))

    # Instantaneous growth rate, assuming f~f0*exp(i*omega*t)
    gamma = np.zeros(len(time), dtype='complex')
    gamma[0] = 1/(time[1]-time[0])*np.log(quantity[1]/quantity[0])
    gamma[1:] = 1/(time[1:]-time[:-1])*np.log(quantity[1:]/quantity[:-1])

    return gamma


def Laplace_transform(times, f_t, omega_r, omega_i):

    X, Y = np.meshgrid(omega_r, omega_i)

    f_t_interp_real = interp(times, np.real(f_t))
    f_t_interp_imag = interp(times, np.imag(f_t))

    def Laplace_integrand_real(t, omega):
        return np.real( (f_t_interp_real(t)+1j*f_t_interp_imag(t)) * np.exp(1j*omega*t) )
    def Laplace_integrand_imag(t, omega):
        return np.imag( (f_t_interp_real(t)+1j*f_t_interp_imag(t)) * np.exp(1j*omega*t) )

    omega = X + 1j*Y
    Z = np.zeros_like(omega)
    
    N_r = len(omega_r)
    N_i = len(omega_i)
    for i_r in range(N_r):
        for i_i in range(N_i):
            print("Laplace: evaluating omega %i/%i..." % (1+i_i+i_r*N_i, N_i*N_r), end="\r")
            Z_real = integrate.quad(Laplace_integrand_real, times[0], times[-1], args=(omega[i_i,i_r],))[0]
            Z_imag = integrate.quad(Laplace_integrand_imag, times[0], times[-1], args=(omega[i_i,i_r],))[0]
            #print(Z_real)
            Z[i_i, i_r] = (Z_real + 1j*Z_imag)#*np.exp(np.imag(omega[i_i,i_r])*times[-1])

    return X, Y, Z


def estimate_omega_gamma_signal(f_t, t, ignore_omega=False):
    #argmaxs = argrelextrema(np.abs((f_t)), np.greater)[0]
    argmaxs = argrelextrema(np.abs(np.real(f_t)), np.greater)[0]
    if len(argmaxs)<=1 or ignore_omega:
        omega = 0
        gamma = np.log(np.abs(f_t)[-1]/np.abs(f_t)[0])/(t[-1]-t[0])
        omega_stddev = 0
        gamma_stddev = 0
    else:
        t_maxs = t[argmaxs]
        f_maxs = f_t[argmaxs]
        avg_period = np.mean(np.gradient(t_maxs))
        period_stddev = np.std(np.gradient(t_maxs))
        omega = 2*np.pi/avg_period/2
        omega_stddev = omega * period_stddev/avg_period
        gamma_all = np.log(np.abs(f_maxs[1:])/np.abs(f_maxs[:-1]))/np.diff(t_maxs)
        print(gamma_all)
        gamma = np.mean(gamma_all)
        gamma_stddev = np.std(gamma_all)

    return omega, gamma, omega_stddev, gamma_stddev
