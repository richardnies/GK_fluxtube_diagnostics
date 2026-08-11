"""Plots of quantities as a function of the field-line-following coordinate zed (electrostatic potential, flux-tube geometry, generic quantities)."""

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
from stella_diagnostics.plotting.mpl_helpers import get_or_create_ax


def read_phi_vs_zed(run, time_avg=None, time_idx=-1, normalise_phi=True, kx_idx=0, ky_idx=0, eval_real=True, squared=False, remove_zonal=False):


    # phi_vs_t(t, tube, zed, theta0, ky, ri)
    if time_avg is None:
        phi_vs_zed_theta0_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx,0]
    else:
        phi_vs_t_zed_theta0_ky_ri = run.ncdata.variables['phi_vs_t'][:,0]

        time   = run.ncdata.variables['t'][:]
        time_max = time[time_idx]
        phi_vs_zed_theta0_ky_ri = np.mean( phi_vs_t_zed_theta0_ky_ri[time > time_max-time_avg], axis=0)

    if eval_real:
        phi_vs_zed_theta0_ky = phi_vs_zed_theta0_ky_ri[:,:,:,0]
    else:
        phi_vs_zed_theta0_ky = np.abs(phi_vs_zed_theta0_ky_ri[:,:,:,0] + 1j*phi_vs_zed_theta0_ky_ri[:,:,:,1])

    if squared:
        phi_vs_zed_theta0_ky = phi_vs_zed_theta0_ky**2

    if remove_zonal:
        phi_vs_zed_theta0_ky[:,:,0] = 0

    if ky_idx is not None:
        phi_vs_zed_theta0 = phi_vs_zed_theta0_ky[:,:,ky_idx]
    else:
        phi_vs_zed_theta0 = np.sum(phi_vs_zed_theta0_ky, axis=2)
    
    if kx_idx is not None:
        phi_vs_zed = phi_vs_zed_theta0[:,kx_idx]
    else:
        phi_vs_zed = np.sum(phi_vs_zed_theta0, axis=1)

    if normalise_phi:
        max_phi = np.max(phi_vs_zed)
        min_phi = np.min(phi_vs_zed)
        if np.abs(max_phi) > np.abs(min_phi):
            phi_vs_zed = phi_vs_zed / max_phi
        else:
            phi_vs_zed = phi_vs_zed / min_phi
        
    zed      = run.ncdata.variables['zed'][:]

    return phi_vs_zed, zed


def plot_phi_vs_zed(run, ax=None, label=None, ls=None, color=None, zed_times_nfield_periods=False, time_idx=-1, normalise_phi=True):

    fig, ax = get_or_create_ax(ax=ax, nrows=1, ncols=1, figsize=(12,9))

    phi_vs_t, zed = run.read_phi_vs_zed(time_idx=time_idx, normalise_phi=normalise_phi)

    time_eval   = run.ncdata.variables['t'][time_idx]

    set_xlim = True
    if zed_times_nfield_periods:
        geom_quantities = np.loadtxt(run.geo_file, skiprows=2).T
        zed = geom_quantities[1]
        set_xlim = False

    plot_y_over_zed(ax, zed, phi_vs_t, ylabel=r"$\phi$", label=label, ls=ls, color=color, set_xlim=set_xlim)

    return ax, time_eval

    # NOTE (pre-existing bug, not fixed -- see decision to preserve behavior
    # byte-for-byte during the restructure): the comment below originally
    # read "#######  Plot electrostatic potential over the flux tube def
    # plot_phi2_vs_t_zed(...):" in the source file this was extracted from.
    # The leading "#" swallowed the `def` line, so what would have been a
    # separate plot_phi2_vs_t_zed(...) method never existed as a callable
    # function -- the code below is unreachable dead code appended after
    # the `return` above. Left as-is; a real fix would either restore
    # plot_phi2_vs_t_zed as its own function or delete this block.
    #######  Plot electrostatic potential over the flux tube def plot_phi2_vs_t_zed(run, tube=0, ax=None, label=None, zed_times_nfield_periods=False, remove_zonal=False):

    fig, ax = get_or_create_ax(ax=ax, nrows=1, ncols=1, figsize=(12,9))

    phi2_vs_t_zed, time, zed = run.read_phi2_vs_t_zed(tube, remove_zonal=remove_zonal)

    set_xlim = True
    if zed_times_nfield_periods:
        geom_quantities = np.loadtxt(run.geo_file, skiprows=2).T
        zed = geom_quantities[1]
        set_xlim = False

    #X, Y = np.meshgrid(zed, time)
    X, Y = np.meshgrid(time, zed)
    Z = phi2_vs_t_zed.T

    eps_rel = 3e-2
    #im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='inferno', vmax=10)
    im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=max(Z.min(), eps_rel*Z.max()), vmax=Z.max()), shading='auto', cmap='inferno')

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\zeta$")
    ax.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    return fig, ax, im


def plot_flux_tube_geometry(run, fig=None, axs=None, label=None, plot_phi=True, zed_times_nfield_periods=False, load_from_nc=True, normalise_bmag=False, color=None, ls="-", xlim=None, norm_gradpar=False):

    if axs is None:
        fig, axs = plt.subplots(nrows=3,ncols=4, figsize=(20,12))
        plt.subplots_adjust(hspace=0,left=0.08,right=0.95,top=0.95,bottom=0.05,wspace=0.5)

    _, _, zed = run.get_kx_ky_zed()

    set_xlim = True
    if zed_times_nfield_periods:
        geom_quantities = np.loadtxt(run.geo_file, skiprows=2).T
        zed = geom_quantities[1]
        set_xlim = False

    if load_from_nc:
        if run.code == "stella":
            bmag     = run.ncdata.variables['bmag'][:]
            i = 0
            #np.savetxt("data_bmag.dat", bmag)
            #np.savetxt("data_zed.dat", zed)

            gradpar  = run.ncdata.variables['gradpar'][:]
            kperp2   = np.zeros_like(gradpar)#run.ncdata.variables['kperp2'][:][:,0,0,0] 
            #kperp2   = run.ncdata.variables['kperp2'][:][:,0,1,0] 
            jacob    = run.ncdata.variables['jacob'][:]
            gbdrift  = run.ncdata.variables['gbdrift'][:]  # drift * grad(y)
            gbdrift0 = run.ncdata.variables['gbdrift0'][:] # drift * grad(x) * shat
            cvdrift  = run.ncdata.variables['cvdrift'][:]  # drift * grad(y)
            cvdrift0 = run.ncdata.variables['cvdrift0'][:] # drift * grad(x) * shat
            gds2     = run.ncdata.variables['gds2'][:]
            gds21    = run.ncdata.variables['gds21'][:]
            gds22    = run.ncdata.variables['gds22'][:]
            grho     = run.ncdata.variables['grho'][:]

        elif run.code == "GX":
            i = 0
            bmag     = run.ncdata['Geometry']['bmag'][:]
            gradpar  = np.array(run.ncdata['Geometry']['gradpar']    )* np.ones_like(bmag)
            gbdrift  = 2*np.array(run.ncdata['Geometry']['gbdrift'][:] ) # drift * grad(y)
            gbdrift0 = 2*np.array(run.ncdata['Geometry']['gbdrift0'][:]) # drift * grad(x) * shat
            cvdrift  = 2*np.array(run.ncdata['Geometry']['cvdrift'][:] ) # drift * grad(y)
            cvdrift0 = 2*np.array(run.ncdata['Geometry']['cvdrift0'][:]) # drift * grad(x) * shat
            gds2     = run.ncdata['Geometry']['gds2'][:]
            gds21    = run.ncdata['Geometry']['gds21'][:]
            gds22    = run.ncdata['Geometry']['gds22'][:]
            grho     = run.ncdata['Geometry']['grho'][:]
            kperp2   = np.zeros_like(gradpar)#run.ncdata.variables['kperp2'][:][:,0,0,0] 
            jacob    = np.zeros_like(gradpar)#run.ncdata.variables['kperp2'][:][:,0,0,0] 

    else:
        geom_quantities = np.loadtxt(run.geo_file, skiprows=2).T

        # alpha zeta bmag gradpar grad_alpha2 gd_alph_psi grad_psi2 gds23 gds24 gbdriftalph gbdrift0psi cvdriftalph cvdrift0psi theta_vmec B_sub_theta B_sub_zeta
        # 0     1    2    3       4           5           6         7     8     9           10          11          12          13         14          15
#            bmag     = geom_quantities[2]
#            gradpar  = geom_quantities[3]
#            kperp2   = run.ncdata.variables['kperp2'][:][:,0,0,0] 
#            jacob    = run.ncdata.variables['jacob'][:]
#            gbdrift  = geom_quantities[9]
#            gbdrift0 = geom_quantities[10]
#            cvdrift  = geom_quantities[11]
#            cvdrift0 = geom_quantities[12]
#            gds2     = geom_quantities[4]
#            gds21    = geom_quantities[5]
#            gds22    = geom_quantities[6]
#            grho     = run.ncdata.variables['grho'][:]


        # alpha zed zeta bmag bdot_grad_z gds2 gds21 gds22 gds23 gds24 gbdrift cvdrift gbdrift0 bmag_psi0 btor
        # 0     1   2    3    4           5    6     7     8     9     10      11      12       13
        bmag     = geom_quantities[3]
        gradpar  = geom_quantities[4]
        kperp2   = run.ncdata.variables['kperp2'][:][:,0,0,0] 
        jacob    = run.ncdata.variables['jacob'][:]
        gbdrift  = geom_quantities[10]
        gbdrift0 = geom_quantities[12]
        cvdrift  = geom_quantities[11]
        cvdrift0 = geom_quantities[12]
        gds2     = geom_quantities[5]
        gds21    = geom_quantities[6]
        gds22    = geom_quantities[7]
        grho     = run.ncdata.variables['grho'][:]

#            # alpha zeta bmag gradpar bdot_grad_z grad_alpha2 gd_alph_psi grad_psi2 gds23 gds24 gbdriftalph gbdrift0psi cvdriftalph cvdrift0psi theta_vmec B_sub_theta B_sub_zeta
#            # 0     1    2    3       4           5           6           7         8     9     10          11          12          13          14         15          16
#            bmag     = geom_quantities[2]
#            gradpar  = geom_quantities[3]
#            kperp2   = run.ncdata.variables['kperp2'][:][:,0,0,0] 
#            jacob    = run.ncdata.variables['jacob'][:]
#            gbdrift  = geom_quantities[10]
#            gbdrift0 = geom_quantities[11]
#            cvdrift  = geom_quantities[12]
#            cvdrift0 = geom_quantities[13]
#            gds2     = geom_quantities[5]
#            gds21    = geom_quantities[6]
#            gds22    = geom_quantities[7]
#            grho     = run.ncdata.variables['grho'][:]
#

    if run.code == "stella":
        shat   = run.ncdata.variables['shat'].getValue()
    else:
        shat   = run.ncdata['Geometry']['shat'].getValue()

    if not plot_phi:
        gradpar_or_phi       = gradpar
        label_gradpar_or_phi = r"$\nabla_\parallel \zeta$"
    else:
        gradpar_or_phi, _   = run.read_phi_vs_zed()
        label_gradpar_or_phi = r"$\phi$"

    if norm_gradpar:
        norm = gradpar
    else:
        norm = 1

    # Normalise Bmag to go from 0.5 to 1.5
    if normalise_bmag:
        bmag = 0.5 * ( 2 + (bmag - (min(bmag)+max(bmag))/2 ) / ((max(bmag)-min(bmag))/2) )

    # Plot
    cvdrift = cvdrift/2
    plot_y_over_zed(axs[0,0], zed, cvdrift, ylabel=r"$B^{-2} \mathbf{B}\times\mathbf{\kappa}\cdot\nabla y$", no_xticks=True, set_xlim=set_xlim, label=label, color=color, ls=ls, xlim=xlim)
    cvdrift0 = cvdrift0 / (2*shat)
    plot_y_over_zed(axs[1,0], zed, cvdrift0, ylabel=r"$B^{-2} \mathbf{B}\times\mathbf{\kappa}\cdot\nabla x$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
    plot_y_over_zed(axs[2,0], zed, gradpar_or_phi, ylabel=label_gradpar_or_phi, no_xticks=False, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)

    plot_y_over_zed(axs[0,1], zed, bmag, ylabel=r"$B$", no_xticks=True, label=label, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
    gbdrift = gbdrift/2
    plot_y_over_zed(axs[1,1], zed, gbdrift/norm, ylabel=r"$v_{My}$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
    #plot_y_over_zed(axs[1,1], zed, gbdrift, ylabel=r"$B^{-3} \mathbf{B}\times\nabla B\cdot\nabla y$", no_xticks=True, set_xlim=set_xlim, color=color, xlim=xlim)
    gbdrift0 = gbdrift0 / (2*shat)
    plot_y_over_zed(axs[2,1], zed, gbdrift0/norm, ylabel=r"$v_{Mx}$", no_xticks=False, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
    #plot_y_over_zed(axs[2,1], zed, gbdrift0, ylabel=r"$B^{-3} \mathbf{B}\times\nabla B\cdot\nabla x$", no_xticks=False, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)


    plot_y_over_zed(axs[0,2], zed, gds2, ylabel=r"$|\nabla y|^2$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
    axs[0,2].set_ylim(ymin=0)
    gds21 = gds21/shat
    plot_y_over_zed(axs[1,2], zed, gds21, ylabel=r"$\nabla y \cdot \nabla x$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
    gds22 = gds22/shat**2
    plot_y_over_zed(axs[2,2], zed, gds22, ylabel=r"$|\nabla x|^2$", no_xticks=False, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
    axs[2,2].set_ylim(ymin=0)

    plot_y_over_zed(axs[0,3], zed, kperp2, ylabel=r"$(\rho_i k_\perp)^2$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
    plot_y_over_zed(axs[1,3], zed, jacob, ylabel=r"$\sqrt{g}$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
    plot_y_over_zed(axs[2,3], zed, grho, ylabel=r"$|\nabla \rho|$", no_xticks=False, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)

    if label is not None:
        axs[0,0].legend()

    # Evaluate flux-surface avg of FLR related quantities
    dl_over_B_avg = run.dl_over_B_avg()
    nablax2_avg = np.sum(dl_over_B_avg * gds22[:])
    nablay2_avg = np.sum(dl_over_B_avg * gds2[:])
    nablaxy_avg = np.sum(dl_over_B_avg * gds21[:])
    kperp2_avg  = np.sum(dl_over_B_avg * kperp2)

    print("\n"+run.filename_base+":")
    print("shat = %e" % (shat))
    print("Bmin, Bmax = %.2e, %.2e" % (bmag.min(), bmag.max()))
    print("Max(vMx) = %.2e" % (gbdrift0.max()))
    print("Avg of |nabla x|^2 = %e" % (nablax2_avg))
    print("Avg of |nabla y|^2 = %e" % (nablay2_avg))
    print("Avg of nabla x * nabla y = %e" % (nablaxy_avg))
    print("Avg of |kperp|^2  = %e" % (kperp2_avg))
    print("gradpar(theta(0)) = %.2e" % (gradpar[0]))
    print("(kperp*rho)^2 in [%e, %e]" % (kperp2.min(), kperp2.max()) )
    print(" min, max of |grad-y| = %e, %e" % (np.sqrt(gds2).min(), np.sqrt(gds2).max()))
    print(" min, max of |grad-x| = %e, %e" % (np.sqrt(gds22).min()/shat, np.sqrt(gds22).max()/shat))

    return fig, axs


def plot_quantities_over_zed(run, fig=None, ax=None, mult_zed=1, zed_times_nfield_periods=False, time_idx=-1, ls=None, color=None, norm_all=False, **kwargs):


    plot_phi       = kwargs.get('plot_phi'       , False)
    norm_phi       = kwargs.get('norm_phi'       , True)
    log_phi        = kwargs.get('log_phi'        , False)
    time_avg       = kwargs.get('time_avg'       , None)
    kx_idx_phi     = kwargs.get('kx_idx_phi'     , 0)
    ky_idx_phi     = kwargs.get('ky_idx_phi'     , 0)
    eval_reim_phi  = kwargs.get('eval_reim_phi'  , True)
    squared_phi    = kwargs.get('squared_phi'    , False)
    remove_zonal_phi = kwargs.get('remove_zonal_phi', False)
    label_phi      = kwargs.get('label_phi'      , "")
    return_phi     = kwargs.get('return_phi'     , False)

    plot_nablax2   = kwargs.get('plot_nablax2'   , False)
    plot_nablaxy   = kwargs.get('plot_nablaxy'   , False)
    plot_nablay2   = kwargs.get('plot_nablay2'   , False)
    plot_B         = kwargs.get('plot_B'         , False)
    norm_B         = kwargs.get('norm_B'         , False)
    plot_Gamma0    = kwargs.get('plot_Gamma0'    , False)
    plot_omega_s_k = kwargs.get('plot_omega_s_k' , False)
    norm_omega_s_k = kwargs.get('norm_omega_s_k' , True)
    plot_gi        = kwargs.get('plot_gi'        , False)
    plot_ge        = kwargs.get('plot_ge'        , False)
    norm_factor_omega_s_k = kwargs.get('norm_factor_omega_s_k' , None)
    plot_qflx      = kwargs.get('plot_qflx',       False)

    _, _, gds2, gds21, gds22, bmag = run.get_FLR(ky_idx=0, kx_idx=0)

    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    zed = run.ncdata.variables['zed'][:] * mult_zed
    time_val = run.ncdata.variables['t'][time_idx]
    set_xlim = True
    if zed_times_nfield_periods:
        geom_quantities = np.loadtxt(run.geo_file, skiprows=2).T
        zed = geom_quantities[1] * mult_zed
        set_xlim = False

    if plot_Gamma0:
        Gamma0   = run.get_Gamma0()
        plot_y_over_zed(ax, zed, Gamma0, label=r"$\Gamma_0(b)$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)

    if plot_omega_s_k:
        omega_s_k, _, _ = run.get_omega_s_k()
        if norm_omega_s_k:
            if norm_factor_omega_s_k is None:
                norm_factor_omega_s_k = np.abs(omega_s_k).max()
            print("omega star over omega curvature normalised by %e" % (norm_factor_omega_s_k))
            omega_s_k = omega_s_k / norm_factor_omega_s_k

        plot_y_over_zed(ax, zed, omega_s_k, label=r"$\omega_\star^T / \omega_\kappa$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)


    if plot_B:
        if norm_B:
            bmag = bmag/bmag.max()
        plot_y_over_zed(ax, zed, bmag, label=r"$B$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)

    if plot_nablax2:
        plot_y_over_zed(ax, zed, gds22, label=r"$|\nabla x|^2$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)
    if plot_nablaxy:
        plot_y_over_zed(ax, zed, gds21, label=r"$\nabla x\cdot\nabla y$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)
    if plot_nablay2:
        plot_y_over_zed(ax, zed, gds2,  label=r"$|\nabla y|^2$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)

    if plot_phi:
        phi_t, _ = run.read_phi_vs_zed(normalise_phi=norm_phi, time_avg=time_avg, time_idx=time_idx, kx_idx=kx_idx_phi, ky_idx=ky_idx_phi, eval_real=False, squared=squared_phi, remove_zonal=remove_zonal_phi)

        if log_phi:
            phi_plot = np.log(np.abs(phi_t))
            eval_reim_phi = False
        else:
            phi_plot = phi_t

        if eval_reim_phi:
            if ky_idx_phi != 0:
                label    = r"$\varphi_r$"
                plot_y_over_zed(ax, zed, np.real(phi_plot), label=label+label_phi, set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, lw=2)
                label    = r"$\varphi_i$"
                plot_y_over_zed(ax, zed, np.imag(phi_plot), label=label+label_phi, set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, lw=1)
            else:
                plot_y_over_zed(ax, zed, np.real(phi_plot), label=label_phi, set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, lw=2)

        else:
            if log_phi:
                label    = r"$\mathrm{log}|\varphi|$"
            else:
                label    = r"$|\varphi|$"
            if squared_phi:
                label = label + r"$^2$"
            plot_y_over_zed(ax, zed, phi_plot, label=label+label_phi, set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, lw=1)


    if plot_qflx:
        qflx_t_zed_kx_ky, _, _, _, _ = run.read_flux_spectra(species_idx=0, tube=0)
        qflx = np.sum( qflx_t_zed_kx_ky[time_idx], axis=(1,2))
        if norm_all:
            qflx = qflx/qflx.max()
        plot_y_over_zed(ax, zed, qflx, label=r"$Q$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, lw=2)


    #vpa_index = 0
    vpa_index = None
    if plot_gi:
        gz_i, _ = run.read_g_vs_zed(species_idx=0, vpa_index=vpa_index, time_idx=time_idx, normalise=False)
        norm_g = np.abs(gz_i).max()
        plot_y_over_zed(ax, zed, gz_i/norm_g, label=r"$g_i$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, alpha=0.5)

    if plot_ge:
        gz_e, _ = run.read_g_vs_zed(species_idx=1, vpa_index=vpa_index, time_idx=time_idx, normalise=False)
        #if not plot_gi:
        norm_g = np.abs(gz_e).max()
        plot_y_over_zed(ax, zed, gz_e/norm_g, label=r"$g_e$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, alpha=0.5)

    ax.set_title(r"$t=%.2f$" % (time_val))

    if return_phi:
        return fig, ax, phi_plot, zed
    else:
        return fig, ax


def plot_quantity_zed_t(run, quantity, fig=None, ax=None, vmin=None, vmax=None, species_idx=0, logarithmic=False, remove_zonal=False, only_zonal=False, sideband=False, time_idx_skip=1, normalise_each_t=False, cmap='inferno', kx_order=0, ky_order=0, nx=None, ny=None, avg_norm=None, time_min=0, time_max=99999, mult_zed=None, kxmin_filter=np.inf, plot_zed_avg=True):


    zed    = run.ncdata.variables['zed'][:]
#        kx     = run.ncdata.variables['kx'][:]
#        ky     = run.ncdata.variables['ky'][:]

    time_all   = run.ncdata.variables['t'][:]
    time_idx_min = nearest_index(time_all-time_min)
    time_idx_max = nearest_index(time_all-time_max)
    time_plot    = time_all[time_idx_min:time_idx_max:time_idx_skip]
    time_idxs = range(time_idx_min, time_idx_max, time_idx_skip)
    assert(len(time_plot)==len(time_idxs))
    assert(len(time_plot)>0)

    zed_weight = run.get_zed_weight(mult_zed=mult_zed, zed=zed)

    f_t_zed = np.zeros((len(time_plot), len(zed) ))
    for i_idx, time_idx in enumerate(time_idxs):
        print("Evaluating time_idx %.6i/%i..." % (i_idx+1, len(time_idxs)), end="\r")

        x_der_taken = False
        y_der_taken = False
        if quantity == "phi-phi":
            phi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, remove_zonal=True, only_zonal=False, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            f_zed_x_y = phi_zed_x_y**2

        elif quantity == "phi-pressure_perp":
            phi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            Pprp_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            f_zed_x_y = phi_zed_x_y * Pprp_zed_x_y

        elif quantity == "dyphi-T":
            dyphi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            T_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("temperature", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            f_zed_x_y = dyphi_zed_x_y * T_zed_x_y

        elif quantity == "dyphi-upar":
            dyphi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            upar_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("upar", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            f_zed_x_y = dyphi_zed_x_y * upar_zed_x_y

        elif quantity == "dyphi-P":
            dyphi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            P_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("pressure", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            f_zed_x_y = dyphi_zed_x_y * P_zed_x_y

        elif quantity == "dyphi-chi":
            dyphi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            chi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("chi", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            f_zed_x_y = dyphi_zed_x_y * chi_zed_x_y

        elif quantity == "dyphi-dyPprp":
            dyphi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            dyPprp_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            f_zed_x_y = dyphi_zed_x_y * dyPprp_zed_x_y

        elif quantity == "dxphi-dyPprp":
            dxphi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, kx_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            dyPprp_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            f_zed_x_y = dxphi_zed_x_y * dyPprp_zed_x_y

        elif quantity == "dyphi-dyphi":
            dyphi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            f_zed_x_y = dyphi_zed_x_y**2

        elif quantity == "kx-avg":
            phi_zed_kx_ky, zed, kx, ky, time_eval = run.get_quantity_zed_kx_ky("phi", time_idx=time_idx, species_idx=species_idx, kxmin_filter=kxmin_filter)
            kx_avg = np.sum( kx[None,:,None] * np.abs(phi_zed_kx_ky[:,:,1:])**2, axis=(1,2)) / np.sum( np.abs(phi_zed_kx_ky[:,:,1:])**2, axis=(1,2))

            f_zed_x_y = kx_avg[:,None,None]

        elif quantity == "dxphi-dyphi":
            dxphi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, kx_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            dyphi_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            f_zed_x_y = dxphi_zed_x_y * dyphi_zed_x_y

        else:
            f_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity=quantity, time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
            x_der_taken = True
            y_der_taken = True

        # Take derivatives by finite differences if needed
        if not x_der_taken:
            for i in range(kx_order):
                f_zed_x_y = np.gradient(f_zed_x_y, axis=1)/(x[1]-x[0])
        if not y_der_taken:
            for i in range(ky_order):
                f_zed_x_y = np.gradient(f_zed_x_y, axis=2)/(y[1]-y[0])

        # Average over x-y
        if avg_norm == "abs":
            f_zed = np.sum( np.abs(f_zed_x_y), axis=(1,2) )
        elif avg_norm == 2:
            f_zed = np.sqrt( np.sum( f_zed_x_y**2, axis=(1,2) ) )
        elif avg_norm == "center":
            f_zed = f_zed_x_y[:,0,0]
        elif avg_norm == "zonal_center":
            f_zed = np.sum(f_zed_x_y[:,0], axis=1)
        else:
            f_zed = np.sum( f_zed_x_y, axis=(1,2) )

        # Save to array
        f_t_zed[i_idx] = f_zed

    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,10))
        plt.subplots_adjust(left=0.15,right=0.95)

    f_t_zed = f_t_zed*zed_weight[None,:]

    X, Y = np.meshgrid(time_plot, zed)
    Z    = f_t_zed.T

    if normalise_each_t:
        for time_idx in range(len(time_plot)):
            Z[:,time_idx] = Z[:,time_idx]/max(np.abs(Z[:,time_idx]))

    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    if logarithmic:
        Z = np.abs(Z)

    if vmax is None:
        vmax = Z.max()
    if vmax == "last":
        vmax = np.abs(Z[:,-1]).max()
    if vmin == "symm":
        vmin = -vmax
    elif vmin is None:
        if logarithmic:
            vmin = 1e-2*vmax
        else:
            vmin = Z.min()

    if logarithmic:
        im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
    else:
        im = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)

    if plot_zed_avg:
        zed_avg_t = np.sum(np.abs(f_t_zed)*zed[None,:], axis=1)/np.sum(np.abs(f_t_zed), axis=1)*10
        ax.plot(time_plot, zed_avg_t, ls='--', lw=2, c='k')

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\zeta$")

    return fig, ax, im


def plot_quantity_x_zed(run, quantity="phi", fig=None, ax=None, time_idx=-1, vmin=None, vmax=None, logarithmic=False, remove_zonal=False, only_zonal=False, avg_norm=None, nx=None, ny=None, species_idx=0, cmap='inferno', kx_order=0, ky_order=0, kxmin_filter=1000, kxmax_filter=0, polar_plot=False, idx_x_shift=None, mult_zed=None, mult_fac=1, xlim_box=None):

    # Figure
    if ax is None:
        if polar_plot:
    #        x += x[-1]/2
            fig, ax = plt.subplots(figsize=(12,10), subplot_kw=dict(projection='polar'))
            ax.set_rorigin(-2*x[-1])
        else:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,10))
    #    plt.subplots_adjust(left=0.15,right=0.95)

    # Load data
    if isinstance(quantity, str):
        f_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity, time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, nx=nx, ny=ny, kx_order=kx_order, ky_order=ky_order, kxmin_filter=kxmin_filter, kxmax_filter=kxmax_filter)

        # zed weight and multiplication factor
        zed_weight = run.get_zed_weight(mult_zed, zed)
        f_zed_x_y = f_zed_x_y*zed_weight[:,None,None]*mult_fac

        # Shift in x if desired
        if idx_x_shift and idx_x_shift > 0 and idx_x_shift < len(x)-1:
            idx_sort  = np.concatenate( (range(idx_x_shift,len(x)), range(idx_x_shift)) )
            f_zed_x_y = f_zed_x_y[:,idx_sort,:]

        if only_zonal:
            avg_norm = "zonal"

        # Average over y
        if avg_norm == "abs":
            f_zed_x = np.sum( np.abs(f_zed_x_y), axis=2 )
        elif avg_norm == 2:
            f_zed_x = np.sqrt( np.sum( f_zed_x_y**2, axis=2 ) )
        elif avg_norm == "center":
            f_zed_x = f_zed_x_y[:,:,0]
        else:
            f_zed_x = np.sum( f_zed_x_y, axis=2 )

        # Plot only part of box if desired
        if xlim_box is not None:
            idx_min = nearest_index(x-xlim_box[0])
            idx_max = nearest_index(x-xlim_box[1])

            x = x[idx_min:idx_max]
            f_zed_x = f_zed_x[:,idx_min:idx_max]

        fig.suptitle(r"$t v_T/a=%.2f$" % (time_eval))


    else:
        f_zed_x = quantity
        kx, _, zed = run.get_kx_ky_zed()
        xmax = np.pi/(kx[1]-kx[0])
        x = np.linspace(-xmax, xmax, len(f_zed_x[0]), endpoint=False)

    # Plot

    X, Y = np.meshgrid(x, zed)
    Z = f_zed_x

    if vmax is None:
        vmax = np.abs(Z).max()
    if vmax == "last":
        vmax = np.abs(Z[:,-1]).max()
    if vmin == "symm":
        vmin = -vmax
    elif vmin is None:
        if logarithmic:
            vmin = 1e-2*vmax
        else:
            vmin = Z.min()

    if logarithmic:
        im = ax.pcolormesh(Y, X, np.abs(Z), norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)#, rasterized=True)
        #im = ax.pcolormesh(X, Y, np.abs(Z), norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
    else:
        #im = ax.contourf(Y, X, Z)#, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
        im = ax.pcolormesh(Y, X, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)#, rasterized=True)

    im.set_edgecolor('face')

    if not polar_plot:
        ax.set_ylim(ymin=x[0],ymax=x[-1])
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$x/\rho$")

    ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    # Evaluate zed_avg(x)
    zed_avg_x = np.sum(f_zed_x*zed[:,None], axis=0)/np.sum(np.abs(f_zed_x), axis=0)
    #dl_over_B_avg = run.dl_over_B_avg()
    #zed_avg_x = np.sum(f_zed_x*dl_over_B_avg[:,None], axis=0)

    return fig, ax, im, x, zed, f_zed_x, zed_avg_x, vmin, vmax


def plot_y_over_zed(ax, zed, y, ylabel=None, label=None, set_xlim=False, ls=None, color=None, no_xticks=False, lw=1, norm=False, xlim=None, alpha=1):
    if norm:
        y = y/y.max()

    if xlim is not None:
        y   = y[  zed >= xlim[ 0]]
        zed = zed[zed >= xlim[ 0]]
        y   = y[  zed <= xlim[-1]]
        zed = zed[zed <= xlim[-1]]

    ax.plot(zed, y, label=label, ls=ls, c=color, lw=lw, alpha=alpha)
    ax.set_ylabel(ylabel)
    #if set_xlim:
    #    #ax.set_xlim([-np.pi, np.pi])
    #    #ax.set_xticks([])
    #    ax.set_xticks([-np.pi,0,np.pi])
    #    ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

    if no_xticks:
        #ax.set_xticks([-np.pi,0,np.pi])
        ax.set_xticklabels([])

    else:
        ax.set_xlabel(r"$\chi$")
