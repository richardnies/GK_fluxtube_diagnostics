"""Cross-run comparison plots of phi/Q k-spectra, plus a directory-scanning heat-flux-vs-gradient utility."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import scipy.special as specialfunc
from scipy.interpolate import interp1d as interp
from scipy.interpolate import RegularGridInterpolator as interp2D
from scipy import integrate
from scipy.interpolate import interpn
from scipy.signal import argrelextrema
import seaborn as sns
from glob import glob
from os.path import exists
from stella_diagnostics.spectral.omega import get_avg_stddev_timetrace
from stella_diagnostics.io.codes import get_rho_label
from stella_diagnostics.io.run import StellaRun, FALLBACK_ASPECT_RATIO
from stella_diagnostics.io.cache import cached


def _style_from_list(style_list, i, default=None):
    """`style_list[i]` if available, else `default`.

    Collapses the ``try: x = x_list[i] except: x = <default>`` per-series
    style-fallback pattern duplicated between plot_phi_k_spectrum and
    plot_Q_k_spectrum (their `marker` fallback has an extra only_zonal
    special case at one of the two call sites, so `marker` is left as its
    own try/except at each site rather than folded in here).
    """
    try:
        return style_list[i]
    except Exception:
        return default


def plot_qflx_tprim_qinp_dir(dirname, filename, rundir_str_exclude=None, rundir_str_beg="run_", rundir_str_end="", species_idx=0, time_max=1e10, time_avg=None, norm=True, configuration=None, fig=None, ax=None, label=None, ls=None, c=None, lw=None, marker='.', code="stella", scale_tprim=1, scale_Q=1, tprim_qinp="both", load_from_file=False, tprim_max=None):

    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(9,5))

    rundirs = glob(dirname+"/"+rundir_str_beg+"*"+rundir_str_end)

    if rundir_str_exclude is not None:
        rundirs_new = []
        for i_dir, rundir in enumerate(rundirs):
            if rundir_str_exclude not in rundir:
                rundirs_new.append(rundir)
        rundirs = rundirs_new
    
    Nr_dirs = len(rundirs)
    tprim_qinp_vals = np.zeros(Nr_dirs)
    qflx_vals  = np.zeros(Nr_dirs)
    qflx_stddev_vals  = np.zeros(Nr_dirs)

    for i_dir in range(Nr_dirs):
        filename_data = rundirs[i_dir]+"/data_qflx_" + tprim_qinp + ".dat"

        if load_from_file and exists(filename_data):
            # Load from file if desired
            tprim_qinp_vals[i_dir], qflx_vals[i_dir], qflx_stddev_vals[i_dir] = np.loadtxt(filename_data)
        else:

            try:
                diagObj = StellaRun(rundirs[i_dir]+"/"+filename, code=code)
                _, _, qflx, time = diagObj.get_fluxes_over_time(species_idx=species_idx, norm=norm, configuration=configuration)#, delta_t=1.5*time_avg)
                qflx = qflx[time<time_max]
                time = time[time<time_max]
    
                if time_avg == "auto":
                    # Time average between t=2*t(Qmax) and t_end
                    idx_Qmax = np.argmax(qflx)
                    double_t_Qmax = 2*time[idx_Qmax]
                    time_start_avg = min( double_t_Qmax, time[-2])
                    time_avg = time[-1]-time_start_avg
                    print("Automatically determine t_avg for " + rundirs[i_dir] +" = %e" % (time_avg))
    
                if time_avg is not None:
                    qflx, qflx_stddev = get_avg_stddev_timetrace(time, qflx, time_avg)
                else:
                    qflx = qflx[-1]
                    qflx_stddev = 0
    
                qflx        = qflx       *scale_Q
                qflx_stddev = qflx_stddev*scale_Q
        
                if code == "GX":
                    tprim = diagObj.ncdata['Inputs']['Species']['T0_prime'][species_idx]
                    geo_data = np.loadtxt(diagObj.geo_file, max_rows=1, skiprows=1)
                    qinp  = geo_data[-1]
                else:
                    tprim = diagObj.ncdata.variables['tprim'][species_idx]
                    if code == "GS2":
                        qinp  = diagObj.ncdata.variables['qval'].getValue()
                    else:
                        qinp  = diagObj.ncdata.variables['q'].getValue()
        
                tprim = tprim*scale_tprim
    
                qflx_vals[i_dir]  = qflx
                qflx_stddev_vals[i_dir]  = qflx_stddev
    
                if tprim_qinp == "tprim":
                    tprim_qinp_vals[i_dir] = tprim
                elif tprim_qinp == "qinp":
                    tprim_qinp_vals[i_dir] = qinp
                elif tprim_qinp == "both":
                    tprim_qinp_vals[i_dir] = qinp*tprim

                np.savetxt(filename_data, (tprim_qinp_vals[i_dir], qflx_vals[i_dir], qflx_stddev_vals[i_dir]))
    
            except:
                print("Failed reading " + rundirs[i_dir])
                continue

    try:
        idx_sort = np.argsort(tprim_qinp_vals)
        tprim_qinp_vals = tprim_qinp_vals[idx_sort]
        qflx_vals  =  qflx_vals[idx_sort]
        qflx_stddev_vals  =  qflx_stddev_vals[idx_sort]
    
        if tprim_max is not None:
            qflx_vals        = qflx_vals[       tprim_qinp_vals<tprim_max*scale_tprim]
            qflx_stddev_vals = qflx_stddev_vals[tprim_qinp_vals<tprim_max*scale_tprim]
            tprim_qinp_vals  = tprim_qinp_vals[ tprim_qinp_vals<tprim_max*scale_tprim]

        if len(qflx_vals)>=1:
            ax.errorbar(tprim_qinp_vals, qflx_vals, yerr=qflx_stddev_vals, ls=ls, label=label, c=c, lw=lw, marker=marker)
        #ax.plot(tprim_vals, qflx_vals, ls=ls, label=label, c=c, lw=lw, marker=marker)
    except:
        print("It seems like none of the directories in " + dirname + " could be read.")

    return fig, ax


def plot_contour_phi_vs_zed_theta0(run, fig=None, ax=None, normalise_phi=False, logarithmic=False, vmin=None, vmax=None):

    # Load values
    list_phi, list_zed = run.load_phi_vs_zed()
    array_phi = np.asarray(list_phi)

    list_theta0 = []
    for i, single_run in enumerate(run.list_runs):
        theta0 = single_run.read_basic_params()['theta0'][0]
        list_theta0.append(theta0)

    # Normalise phi if desired
    if normalise_phi:
        for i_t in range(len(list_theta0)):
            array_phi[i_t, :] = np.abs(array_phi[i_t, :]) / np.max(np.abs(array_phi[i_t, :]))

    # Convert to plottable arrays
    X, Y = np.meshgrid(np.asarray(list_theta0), np.asarray(list_zed[0]) )
    Z = array_phi.T

    # Plot
    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    if vmin is None:
        vmin = Z.min()
    if vmax is None:
        vmax = Z.max()

    if logarithmic:
        im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap='plasma')
    else:
        im = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading='auto', cmap='plasma')

    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\zeta$")

    return fig, ax, im


@cached(version=1)
def get_phi_k_spectrum(run, plot_kx, time_idx=-1, time_avg=None, only_zonal=False, remove_zonal=False, scale_kmin=False, k_exp=0, W_instead_of_phi=False, zonal_stationary=False, plot_RH_phi_spectrum=False):
    """(k, phi2_k, phi2_k_stddev) for one run -- the data half of
    plot_phi_k_spectrum, extracted so it's cached instead of hand-rolled
    to "<filename_base>_Ephi_*.dat"/"..._stddev.dat" files by each caller.
    """
    time = run.get_time_array()
    time_max = time[time_idx]
    if time_avg is not None:
        time_min = time_max-time_avg
    else:
        time_min = time_max-10

    if plot_RH_phi_spectrum:
        E_RH_t_kx, RH_time, RH_kx = run.get_E_RH_t_kx(time_min=time_min, time_max=time_max)
        phi2_k = np.average(E_RH_t_kx[:,RH_kx>0], weights=np.gradient(RH_time), axis=0)*2
        k      = RH_kx[      RH_kx>0]
        phi2_k_stddev = np.zeros_like(phi2_k)

    elif not zonal_stationary:
        if W_instead_of_phi:
            phi2_t_kx_ky, time, kx, ky = run.read_W_spectra(time_min=time_min, time_max=time_max)
        else:
            phi2_t_kx_ky, time, kx, ky = run.read_phi2_spectra(time_min=time_min, time_max=time_max)

        phi2_t_kx_ky[np.isnan(phi2_t_kx_ky)]=0

        if time_avg is None:
            phi2_kx_ky = phi2_t_kx_ky[-1]
            phi2_kx_ky_stddev = np.zeros_like(phi2_kx_ky)
            print("Evaluating at t = %.2f" % (time[-1]))
        else:
            phi2_kx_ky = np.average(    phi2_t_kx_ky, weights=np.gradient(time), axis=0)
            phi2_kx_ky_stddev = np.std( phi2_t_kx_ky,                            axis=0)

        if plot_kx:

            if only_zonal:
                phi2_kx_ky[:,1:] = 0
                phi2_kx_ky[0,0]  = 0
                phi2_kx_ky_stddev[:,1:] = 0
                phi2_kx_ky_stddev[0,0]  = 0
                kx = np.array(kx)
                phi2_kx_ky = phi2_kx_ky[kx > 0, :]
                phi2_kx_ky_stddev = phi2_kx_ky_stddev[kx > 0, :]
                kx = kx[kx>0]

            if remove_zonal:
                phi2_kx_ky[:,0]  = 0
                phi2_kx_ky_stddev[:,0]  = 0

            phi2_k = np.sum( phi2_kx_ky, axis=1)
            phi2_k_stddev = np.sum( phi2_kx_ky_stddev, axis=1)
            idx_sort = np.argsort(kx)
            k = kx[idx_sort]
            phi2_k = phi2_k[idx_sort]
            phi2_k_stddev = phi2_k_stddev[idx_sort]
            k = np.abs(k)

        else:
            phi2_k = np.sum( phi2_kx_ky[:,1:], axis=0)
            phi2_k_stddev = np.sum( phi2_kx_ky_stddev[:,1:], axis=0)
            k = ky[1:]

    # Stationary zonal flows
    else:
        omega, kx, EZ_omega_kx = run.get_EZ_omega_kx(quantity="phi", time_min=-time_avg)
        k = kx[kx>0]
        phi2_k = EZ_omega_kx[0, kx>0]*2/k**2
        # Not a time-average, so there's no real notion of a stddev here
        # -- zero, matching the plot_RH_phi_spectrum branch's convention
        # for its own single-snapshot case above.
        phi2_k_stddev = np.zeros_like(phi2_k)

    # multiply phi2_k with k power if desired
    phi2_k = phi2_k * k**(k_exp)
    phi2_k_stddev = phi2_k_stddev * k**(k_exp)

    if scale_kmin:
        print("Rescaling phi2 with kmin (to be able to compare sims with different x0,y0)")
        phi2_k = phi2_k / np.abs(k[1]-k[0])
        phi2_k_stddev = phi2_k_stddev / np.abs(k[1]-k[0])

    return k, phi2_k, phi2_k_stddev


def plot_phi_k_spectrum(run, plot_kx, fig=None, ax=None, time_idx=-1, ls_list=None, marker_list=None, color_list=None, tprim_norm_list=None, qinp_norm_list=None, xdrift_norm_list=None, time_avg=None, only_zonal=False, remove_zonal=False, scale_kmin=False, k_exp=0, alpha_kx_O=1, beta_kx_O=0, lw=None, no_label=False, scaling_theory="GCB", W_instead_of_phi=False, scale_fac_vals=None, zonal_stationary=False, load_from_file=False, mult_k=False, plot_alpha_spectrum=False, plot_RH_phi_spectrum=False, alpha_plot=1, markersize=3):
    # time_avg (renamed from delta_t_avg): trailing-window width ending at
    # time[time_idx] -- same trailing-window convention as
    # stella_diagnostics.scan.zonal_flow_scan/rh_flux_scan and
    # physics.velocity_space.plot_contour_gvmu_vpa. time_idx default
    # changed from -2 to -1 (matching every other function's "last
    # sample" default, including this file's own plot_Q_k_spectrum,
    # which already defaulted to -1) -- no rationale was found anywhere
    # for -2; flagged here in case this default mattered for a reason
    # not visible from the code alone.
    #
    # load_from_file: kept only for backward compatibility with existing
    # configs -- now a no-op, since get_phi_k_spectrum is @cached and
    # transparently reuses a prior computation whenever nothing relevant
    # changed, replacing the old hand-rolled
    # "<filename_base>_Ephi_*.dat"/"..._stddev.dat" file cache this
    # function used to read/write itself.

    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(9,6))


    for i, single_run in enumerate(run.list_runs):

        k, phi2_k, phi2_k_stddev = get_phi_k_spectrum(
            single_run, plot_kx, time_idx=time_idx, time_avg=time_avg, only_zonal=only_zonal,
            remove_zonal=remove_zonal, scale_kmin=scale_kmin, k_exp=k_exp,
            W_instead_of_phi=W_instead_of_phi, zonal_stationary=zonal_stationary,
            plot_RH_phi_spectrum=plot_RH_phi_spectrum,
        )

        ### RESCALE DATA
        if tprim_norm_list is not None:
            try:
                aspectratio = single_run.aspect_ratio
                print("Aspect ratio = %.2f" % (aspectratio))
            except AttributeError:
                # No geometry file was readable at all for this run (see
                # io.run.StellaRun.__init__) -- same unverified placeholder
                # used there for the Miller-geometry-but-unparseable case,
                # reused here so this doesn't silently drift to a
                # different guess.
                aspectratio = FALLBACK_ASPECT_RATIO
                print("Setting aspect ratio to placeholder %.2f (no geometry file readable)" % (aspectratio))

            try:
                safetyfactor = qinp_norm_list[i]
            except:
                print("Setting safety factor to 1")
                safetyfactor = 1

            phi2_k = phi2_k* 2*aspectratio**2

            kappa  = tprim_norm_list[i]*aspectratio
            print("A = %.4f, kappa = %.4f, q = %.4f" % (aspectratio, kappa, safetyfactor))

            if scaling_theory == "CB":
                k = k*kappa*safetyfactor
                phi2_k = phi2_k / (kappa**5 * safetyfactor**3)

            elif scaling_theory == "GCB":
                if plot_kx:
                     k = k*safetyfactor
                     phi2_k = phi2_k / (kappa**2 * safetyfactor**3)
                else:
                    k = k*kappa*safetyfactor
                    phi2_k = phi2_k / (kappa**3 * safetyfactor**3)

            elif scaling_theory == "MGCB":
                if plot_kx:
                    k = k*safetyfactor**0.5
                    phi2_k = phi2_k / (kappa**2 * safetyfactor**2.5)
                else:
                    k = k*kappa*safetyfactor
                    phi2_k = phi2_k / (kappa**3 * safetyfactor**3)

            elif scaling_theory == "zonal_diffusive":
                phi2_k = phi2_k / (kappa**2 * safetyfactor)

            elif scaling_theory == "heuristic_T":
                if plot_kx:
                    k = k*kappa*safetyfactor
                    phi2_k = phi2_k / (kappa**4 * safetyfactor**3)
                else:
                    k = k*kappa*safetyfactor
                    phi2_k = phi2_k / (kappa**4 * safetyfactor**3)



        if xdrift_norm_list is not None:
            print("Scaling with xdrift_norm")

            norm = xdrift_norm_list[i]

            if plot_kx:
                if only_zonal:
                    kexp_alphaD = 0
                    phi2exp_alphaD = 1
                else:
                    if scaling_theory == "GCB":
                        kexp_alphaD = 1
                        phi2exp_alphaD = 2
                    else:
                        kexp_alphaD = 1/2
                        phi2exp_alphaD = 3/2
            else:
                kexp_alphaD = 0
                phi2exp_alphaD = 1

            phi2_k = phi2_k / (norm**(phi2exp_alphaD))
            k = k*norm**(kexp_alphaD)

        if scale_fac_vals is not None:
            phi2_k = phi2_k*scale_fac_vals[i]


        ####### PLOT
        ls = _style_from_list(ls_list, i, default=None)
        try:
            marker = marker_list[i]
        except:
            marker = '.'
            if only_zonal:
                marker = 's'
        color = _style_from_list(color_list, i, default=None)

        if no_label:
            label = None
        else:
            label = run.list_labels[i]
 
        if only_zonal and not W_instead_of_phi and not plot_RH_phi_spectrum:
            Gamma0 = specialfunc.iv(0, k**2/2) * np.exp(-k**2/2)
            phi2_k = phi2_k*(1-Gamma0)
        elif mult_k:
            phi2_k = phi2_k*np.abs(k)

        if plot_alpha_spectrum:
            idx_k_0 = np.argmin(k)
            k[:idx_k_0] *= -1
            k_mid, alpha_k = get_alpha_spectrum(k, phi2_k)
            k_mid = np.abs(k_mid)
            ax.semilogx(k_mid, alpha_k, label=label, ls='None', marker=marker, color=color, lw=lw, alpha=alpha_plot)
            #ax.plot(k_mid, alpha_k, label=label, ls=ls, marker=marker, color=color, lw=lw)
            ax.set_ylabel(r"$\alpha$")
        else:
            ax.loglog(k[np.abs(k)>0], phi2_k[np.abs(k)>0], label=label, ls=ls, marker=marker, color=color, lw=lw, markersize=markersize, alpha=alpha_plot)

    # NOTE: the "Z_i e ... T_i" part of ylabel_base below stays hardcoded
    # to the ion symbol regardless of species -- a pre-existing,
    # out-of-scope notational assumption (this whole normalization is
    # implicitly ion-referenced); only the rho subscript (the actual
    # gyroradius normalization the axis is in) is made species-aware here.
    rho_label = get_rho_label(run.list_runs[0].ncdata) if run.list_runs else r"\rho"

    if plot_kx:
        if plot_alpha_spectrum:
            ylabel_base = r"$\alpha_{k_x}$"
        else:
            if W_instead_of_phi:
                ylabel_base = r"$W_{k_x}$"
            else:
                if remove_zonal:
                    ylabel_base = r"$\left(\frac{Z_i e\delta\varphi^\mathrm{NZ}_{k_x}}{T_i} \frac{R}{%s}\right)^2$" % rho_label
                    #ylabel_base = r"$(e_i\delta\varphi^\mathrm{NZ}_{k_x}/T_i\; R/\rho_i)^2$"
                elif only_zonal:
                    ylabel_base = r"$\left(\frac{Z_i e\delta\varphi^\mathrm{Z}_{k_x}}{T_i} \frac{R}{%s}\right)^2$" % rho_label
                    #ylabel_base = r"$(e_i\delta\varphi^\mathrm{Z}_{k_x}/T_i\; R/\rho_i)^2$"
                else:
                    ylabel_base = r"$\left(\frac{Z_i e\delta\varphi_{k_x}}{T_i} \frac{R}{%s}\right)^2$" % rho_label
                    #ylabel_base = r"$(e_i\delta\varphi_{k_x}/T_i\; R/\rho_i)^2$"
                #ylabel_base = r"$\Phi_{k_x}^2$"
        xlabel = r"$k_x %s$" % rho_label
        #xlabel = r"$|k_x| \rho_i$"
    else:
        if plot_alpha_spectrum:
            ylabel_base = r"$\alpha_{k_y}$"
        else:
            if W_instead_of_phi:
                ylabel_base = r"$W_{k_y}$"
            else:
                ylabel_base = r"$\left(\frac{Z_i e\delta\varphi_{k_y}}{T_i} \frac{R}{%s}\right)^2$" % rho_label
                #ylabel_base = r"$(e_i\varphi_{k_y}/T_i\; R/\rho_i)^2$"
                #ylabel_base = r"$\Phi_{k_y}^2$"
        xlabel = r"$k_y %s$" % rho_label

    #if delta_t_avg is not None:
    #    ylabel_base = r"$\langle$" + ylabel_base + r"$\rangle_{\Delta t = %i}$" % (delta_t_avg)

    if only_zonal and not W_instead_of_phi:
        ylabel = ylabel_base + r"$(1-\Gamma_0)$"
    elif mult_k:
        ylabel = ylabel_base + r"$k/k_\mathrm{scale}$"
    else:
        ylabel = ylabel_base

    if k_exp != 0:
        ylabel = ylabel + r"$k^{%i}$" % (k_exp)
    else:
        ylabel = ylabel# + r"$(t=%.1f)$" % (time[time_idx])

    if tprim_norm_list is not None:
        #xlabel = xlabel + r"$(q \kappa)^{%.1f}$" % (-alpha_kx_O)
        #ylabel = ylabel + r"$ / (q^3 \kappa^5 (q\kappa)^{7/3(%.1f-1)} )$" % (beta_kx_O)
        if scaling_theory == "CB":
            xlabel = xlabel + r"$q \kappa$"
            ylabel = ylabel + r"$ / q^3 \kappa^{5}$"
        elif scaling_theory == "GCB":
            if plot_kx:
                #xlabel = xlabel + r"$\kappa^{1/2}$"
                #ylabel = ylabel + r"$ / \kappa^{7/2}$"
                xlabel = xlabel + r"$q$"
                ylabel = ylabel + r"$ / q^3 \kappa^{2}$"
            else:
                xlabel = xlabel + r"$q \kappa$"
                ylabel = ylabel + r"$ / q^3 \kappa^{3}$"

        elif scaling_theory == "MGCB":
            if plot_kx:
                xlabel = xlabel + r"$q^{1/2}$"
                ylabel = ylabel + r"$ / q^2 \kappa^{5/2}$"
            else:
                xlabel = xlabel + r"$q \kappa$"
                ylabel = ylabel + r"$ / q^3 \kappa^{3}$"

        elif scaling_theory == "zonal_diffusive":
            ylabel = ylabel + r"$ / q \kappa^2$"

        elif scaling_theory == "heuristic_T":
            xlabel = xlabel + r"$q \kappa$"
            ylabel = ylabel + r"$ / q^4 \kappa^4$"

    add_arb_units = False
    #add_arb_units = True
    if add_arb_units:
        ylabel+=r" (arb. units)"

    if xdrift_norm_list is not None:
        xlabel = xlabel + r"$\alpha_D^{%.2f}$" % (kexp_alphaD)
        ylabel = ylabel + r"$ / \alpha_D^{%.2f}$" % (phi2exp_alphaD)

    ax.set_xlabel(xlabel)
    if plot_alpha_spectrum:
        ax.set_ylabel(ylabel_base)
    else:
        ax.set_ylabel(ylabel)
#        plt.gca().xaxis.grid(True, which='minor')
#        plt.gca().yaxis.grid(True, which='minor')

    # x is always log-scale here (loglog or semilogx) -- when the plotted
    # k range spans less than a decade, matplotlib's default LogLocator
    # labels every minor tick (2x, 3x, 4x, ... per decade), which overlap
    # into an unreadable smear on a narrow-aspect figure. Capping the
    # number of major ticks and dropping minor-tick labels keeps the axis
    # legible regardless of how narrow the k range or figure is.
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=6))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.grid()

    #return fig, ax, time
    return fig, ax


def plot_Q_k_spectrum(run, plot_kx, species_idx=0, tube=0, fig=None, ax=None, time_idx=-1, ls_list=None, marker_list=None, color_list=None, delta_t_avg=None, zed_val=None, scale_k=False, scale_kmin=True, kfilter_vals=None, plot_k_qk=False):

    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(10,8))

    for i, single_run in enumerate(run.list_runs):

        #if single_run.code == "stella":
        qflx_t_zed_kx_ky, time, zed, kx, ky = single_run.read_flux_spectra(species_idx=species_idx, tube=tube)
        delta_kx = kx[1]-kx[0]
        delta_ky = ky[1]-ky[0]

        if delta_t_avg is None:
            qflx_zed_kx_ky = qflx_t_zed_kx_ky[time_idx]
        else:
            qflx_zed_kx_ky = np.average( qflx_t_zed_kx_ky[time > time[time_idx]-delta_t_avg], axis=0)

        if zed_val is None:
            dl_over_B_avg = single_run.dl_over_B_avg()
            qflx_kx_ky = np.sum( dl_over_B_avg[:,None,None] * qflx_zed_kx_ky, axis=0)
        else:
            zed_idx = np.argmin( np.abs( zed[:] - zed_val ) )
            qflx_kx_ky = qflx_zed_kx_ky[zed_idx]

        if plot_kx:
            qflx_k = np.sum( qflx_kx_ky, axis=1)
            idx_sort = np.argsort(kx)
            k = kx[idx_sort]
            qflx_k = qflx_k[idx_sort]
        else:
            qflx_k = np.sum( qflx_kx_ky, axis=0)
            k = ky

        # NOTE: GX support existed here (reading single_run.ncdata['Spectra']
        # ['Qkxst'/'Qkyst']) and was disabled at some point -- this function
        # is dead code (not called by any driver) so it was left as-is
        # rather than restored.

        if scale_k and zed_idx is not None:
            _, _, gds2, _, gds22, _ = single_run.get_FLR()
            if plot_kx:
                k = k*np.sqrt(gds22[zed_idx])
            else:
                k = k*np.sqrt(gds2[zed_idx])

        if scale_kmin:
            print("Rescaling Q with kmin (to be able to compare sims with different x0,y0)")
            qflx_k = qflx_k / np.abs(k[1]-k[0])

        # Evaluate and print integrated heat flux for some kfilter_vals
        if kfilter_vals is not None:
            if plot_kx:
                str_k = "(k=kx)"
            else:
                str_k = "(k=ky)"
            print("\n"+run.filenames_base[i]+str_k+":")
            Q_integrated = np.sum(qflx_k)
            print("    sum_k Q_k = %e" % (Q_integrated))
            for kfilter_val in kfilter_vals:
                Q_integrated_filter = np.sum(qflx_k[k>kfilter_val])
                print("    sum_k Q_k (k>%.4f) = %e (%.2f percent)" % (kfilter_val, Q_integrated_filter, Q_integrated_filter/Q_integrated*100))

        ls = _style_from_list(ls_list, i, default=None)
        try:
            marker = marker_list[i]
        except:
            marker = '.'
        color = _style_from_list(color_list, i, default=None)

        if plot_kx:
            plt.xscale('symlog', linthresh=k[1]-k[0])
            k = np.abs(k)

        if plot_k_qk:
            qflx_k = qflx_k*k

        ax.loglog(k, qflx_k, label=run.list_labels[i], ls=ls, marker=marker, color=color)

        if i == len(run.list_runs)-1:
            # Plot theoretical -7/3 scaling (Barnes et al. 2011)
            idx_Q_max = np.argmax(qflx_k[1:]) + 12
            k_plot = np.linspace(1,10,10)*k[1+idx_Q_max]
            if plot_kx:
                qflx_k_theory = qflx_k[idx_Q_max] * (k_plot/k[idx_Q_max])**(-7/3)
            else:
                qflx_k_theory = qflx_k[idx_Q_max] * (k_plot/k[idx_Q_max])**(-4/3)
                ax.plot(k_plot/4, qflx_k_theory, ls='--', c='g', label=r"$\sim k^{-4/3}$", lw=4)

    rho_label = get_rho_label(run.list_runs[0].ncdata) if run.list_runs else r"\rho"

    if plot_kx:
        ylabel_base = r"$Q_{k_x}$"
        if plot_k_qk:
            ylabel_base += r"$k_x$"
        xlabel = r"$k_x %s$" % rho_label
        if scale_k:
            xlabel = xlabel + r"$|\nabla x|$"
        #xlabel = r"$|k_x| \rho_i$"
    else:
        ylabel_base = r"$Q_{k_y}$"
        if plot_k_qk:
            ylabel_base += r"$k_y$"
        xlabel = r"$k_y %s$" % rho_label
        if scale_k:
            xlabel = xlabel + r"$|\nabla y|$"

    #if delta_t_avg is not None:
    #    ylabel_base = r"$\langle$" + ylabel_base + r"$\rangle_{\Delta t = %i}$" % (delta_t_avg)

    ylabel = ylabel_base# + r"$(t=%.1f)$" % (time[time_idx])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
#        plt.gca().xaxis.grid(True, which='minor')
#        plt.gca().yaxis.grid(True, which='minor')
    ax.grid()
    #ax.legend()

    return fig, ax


def get_alpha_spectrum(k, f_k):
    k_mid = 0.5*(k[1:]+k[:-1])
    f_k_mid = 0.5*(f_k[1:]+f_k[:-1])
    alpha = (f_k[1:]-f_k[:-1])/(k[1]-k[0])*k_mid/f_k_mid
    #alpha = ( np.log(f_k[1:])-np.log(f_k[:-1]) )/(k[1]-k[0])
    return k_mid, alpha
