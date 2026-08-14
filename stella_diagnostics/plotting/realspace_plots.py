"""Plots of quantities in real space (x, y): 3D torus, poloidal ring, flux-surface box, and flat x-y projections."""

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
from stella_diagnostics.plotting.mpl_helpers import get_or_create_ax, resolve_vmin_vmax
from stella_diagnostics.quantities.labels import get_quantity_label


def plot_quantity_3d_torus(run, quantity="phi", fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, vmin=None, vmax=None, cmap=None,torus_rmax=0.6, torus_rmin=0.25, Delta_zeta=np.pi/3, nzeta=50, xlim=np.inf, lighting=True, ikymin=0, ikymax=None):


    f_theta_kx_ky, theta, kx, ky, time_eval = run.get_quantity_zed_kx_ky(quantity, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)
    qinp   = run.safety_factor
    shat   = run.ncdata.variables['shat'].getValue()
    jtwist = run.ncdata.variables['jtwist'].getValue()
    drhodpsi   = run.ncdata.variables['drhodpsi'].getValue()
    dqinp_dx = 1/(jtwist)#/jtwist #shat*qinp*drhodpsi/(2*np.pi)
    #print(dqinp_dx)
    theta0 = run.ncdata.variables['theta0'][:]

    #idx_sort_theta = np.argsort(theta)
    #theta = theta[idx_sort_theta]
    #f_theta_kx_ky = f_theta_kx_ky[idx_sort_theta]

    x = np.linspace(-1, 1, len(kx), endpoint=False)*np.pi/(kx[1]-kx[0])
    if xlim is not None:
        x = x[np.abs(x)<xlim]
    nx = len(x)
    zeta = np.linspace(-np.pi+Delta_zeta, np.pi, nzeta)

    if ikymax is None:
        ky = ky[ikymin:]
        f_theta_kx_ky = f_theta_kx_ky[:,:,ikymin:]
    else:
        ky = ky[ikymin:ikymax]
        f_theta_kx_ky = f_theta_kx_ky[:,:,ikymin:ikymax]


    #print(f_theta_kx_ky[0, :10,3])
    #print(f_theta_kx_ky[-1,:10,3])

    # Evaluate quantity at x extrema
    theta_2D_lastx, zeta_2D_lastx = np.meshgrid(theta, zeta)
    f_theta_zeta_firstx = np.zeros((len(theta), len(zeta)))
    f_theta_zeta_lastx  = np.zeros((len(theta), len(zeta)))
    for i_theta in range(len(theta)):
        for i_zeta in range(len(zeta)):
            #eikonalplus = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[0] + ky[None,:]/ky[1]*(zeta[i_zeta]-qinp*theta[i_theta])
            ##eikonalmin  = -eikonalplus
            #eikonalmin  = (kx[:,None] + shat*ky[None,:]*theta[i_theta])*x[0] - ky[None,:]/ky[1]*(zeta[i_zeta]-qinp*theta[i_theta])
            #f_theta_zeta_firstx[i_theta, i_zeta] = np.sum( np.real( f_theta_kx_ky[i_theta]*np.exp(1j*eikonalplus) + np.conj(f_theta_kx_ky[i_theta])*np.exp(1j*eikonalmin) ))

            qinp_x = qinp + dqinp_dx*x[-1] / x[-1]
            eikonal = ky[None,:]/(ky[1]-ky[0])*( zeta[i_zeta]-qinp_x*(theta[i_theta]-theta0) )
            eikonal[:,0] = kx*x[-1]

            #eikonal = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[-1] + ky[None,:]/(ky[1]-ky[0])*(zeta[i_zeta]-qinp*theta[i_theta])/3
            #eikonal = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[-1] + ky[None,:]/(ky[1]-ky[0])*(zeta[i_zeta]-qinp*theta[i_theta])
            f_theta_zeta_lastx[ i_theta, i_zeta] = np.sum( np.real(f_theta_kx_ky[i_theta])*np.cos(eikonal) - np.imag(f_theta_kx_ky[i_theta])*np.sin(eikonal) )

            qinp_x = qinp + dqinp_dx*x[0] / x[-1]
            eikonal = ky[None,:]/(ky[1]-ky[0])*( zeta[i_zeta]-qinp_x*(theta[i_theta]-theta0) )
            eikonal[:,0] = kx*x[0]
            #eikonal = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[0]  + ky[None,:]/(ky[1]-ky[0])*(zeta[i_zeta]-qinp*theta[i_theta])
            f_theta_zeta_firstx[i_theta, i_zeta] = np.sum( np.real(f_theta_kx_ky[i_theta])*np.cos(eikonal) - np.imag(f_theta_kx_ky[i_theta])*np.sin(eikonal) )

    # Evaluate quantity at zeta cuts
    theta_2D_zeta, x_2D_zeta = np.meshgrid(theta, x)
    f_theta_x_b = np.zeros((len(theta), nx))
    f_theta_x_e = np.zeros((len(theta), nx))
    for i_theta in range(len(theta)):
        for i_x in range(nx):
            #eikonal_bplus = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[i_x] + ky[None,:]/ky[1]*(zeta[0]- qinp*theta[i_theta])
            ##eikonal_bmin  = -eikonal_bplus
            #eikonal_bmin  = (kx[:,None] + shat*ky[None,:]*theta[i_theta])*x[i_x] - ky[None,:]/ky[1]*(zeta[0]- qinp*theta[i_theta])
            #eikonal_eplus = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[i_x] + ky[None,:]/ky[1]*(zeta[-1]-qinp*theta[i_theta])
            ##eikonal_emin  = -eikonal_eplus
            #eikonal_emin  = (kx[:,None] + shat*ky[None,:]*theta[i_theta])*x[i_x] - ky[None,:]/ky[1]*(zeta[-1]-qinp*theta[i_theta])
            #f_theta_x_b[i_theta, i_x] = np.sum( np.real(f_theta_kx_ky[i_theta]*np.exp(1j*eikonal_bplus) + np.conj(f_theta_kx_ky[i_theta])*np.exp(1j*eikonal_bmin) ) )
            #f_theta_x_e[i_theta, i_x] = np.sum( np.real(f_theta_kx_ky[i_theta]*np.exp(1j*eikonal_eplus) + np.conj(f_theta_kx_ky[i_theta])*np.exp(1j*eikonal_emin) ) )
            qinp_x = qinp + dqinp_dx*x[i_x] / x[-1]
            eikonal_b = ky[None,:]/(ky[1]-ky[0])*( zeta[ 0]-qinp_x*(theta[i_theta]-theta0) )
            eikonal_b[:,0] = kx*x[i_x]
            eikonal_e = ky[None,:]/(ky[1]-ky[0])*( zeta[-1]-qinp_x*(theta[i_theta]-theta0) )
            eikonal_e[:,0] = kx*x[i_x]

            #eikonal_b = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[i_x] + ky[None,:]/(ky[1]-ky[0])*(zeta[0]- qinp*theta[i_theta])
            #eikonal_e = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[i_x] + ky[None,:]/(ky[1]-ky[0])*(zeta[-1]-qinp*theta[i_theta])
            f_theta_x_b[i_theta, i_x] = np.sum( np.real(f_theta_kx_ky[i_theta])*np.cos(eikonal_b) - np.imag(f_theta_kx_ky[i_theta])*np.sin(eikonal_b) )
            f_theta_x_e[i_theta, i_x] = np.sum( np.real(f_theta_kx_ky[i_theta])*np.cos(eikonal_e) - np.imag(f_theta_kx_ky[i_theta])*np.sin(eikonal_e) )

    if vmin is None or vmax is None:
        vmax =  max( np.abs(f_theta_x_b).max(), np.abs(f_theta_x_e).max(),np.abs(f_theta_zeta_firstx).max(), np.abs(f_theta_zeta_lastx).max())
        vmin = -vmax

    #### PLOTS
    from mayavi import mlab
    resolution = 4
    mlab.options.offscreen = True
    fig_mlab = mlab.figure(size=(1024, 1024))

    # Plot at first & last x
    X_rmax = np.cos(zeta_2D_lastx)*(1 + torus_rmax*np.cos(theta_2D_lastx))
    Y_rmax = np.sin(zeta_2D_lastx)*(1 + torus_rmax*np.cos(theta_2D_lastx))
    Z_rmax =                            torus_rmax*np.sin(theta_2D_lastx)
    X_rmin = np.cos(zeta_2D_lastx)*(1 + torus_rmin*np.cos(theta_2D_lastx))
    Y_rmin = np.sin(zeta_2D_lastx)*(1 + torus_rmin*np.cos(theta_2D_lastx))
    Z_rmin =                            torus_rmin*np.sin(theta_2D_lastx)

    Out = mlab.mesh(X_rmin,Y_rmin,Z_rmin,colormap='coolwarm',scalars=f_theta_zeta_firstx.T,figure=fig_mlab, vmin=vmin, vmax=vmax, resolution=resolution)
    Out.actor.property.lighting = lighting
    Out = mlab.mesh(X_rmax,Y_rmax,Z_rmax,colormap='coolwarm',scalars=f_theta_zeta_lastx.T ,figure=fig_mlab, vmin=vmin, vmax=vmax, resolution=resolution)
    Out.actor.property.lighting = lighting

    # Plot at zeta cuts
    X_b = np.cos(zeta[0]) * (1 + np.cos(theta_2D_zeta) * (torus_rmin + (torus_rmax-torus_rmin)*(x_2D_zeta-x[0])/(x[-1]-x[0]) ))
    Y_b = np.sin(zeta[0]) * (1 + np.cos(theta_2D_zeta) * (torus_rmin + (torus_rmax-torus_rmin)*(x_2D_zeta-x[0])/(x[-1]-x[0]) ))
    X_e = np.cos(zeta[-1])* (1 + np.cos(theta_2D_zeta) * (torus_rmin + (torus_rmax-torus_rmin)*(x_2D_zeta-x[0])/(x[-1]-x[0]) ))
    Y_e = np.sin(zeta[-1])* (1 + np.cos(theta_2D_zeta) * (torus_rmin + (torus_rmax-torus_rmin)*(x_2D_zeta-x[0])/(x[-1]-x[0]) ))
    Z =                          np.sin(theta_2D_zeta) * (torus_rmin + (torus_rmax-torus_rmin)*(x_2D_zeta-x[0])/(x[-1]-x[0]) )

    Out = mlab.mesh(X_b,Y_b,Z,colormap='coolwarm',scalars=f_theta_x_b.T,figure=fig_mlab, vmin=vmin, vmax=vmax, resolution=resolution)
    Out.actor.property.lighting = lighting

    Out = mlab.mesh(X_e,Y_e,Z,colormap='coolwarm',scalars=f_theta_x_e.T,figure=fig_mlab, vmin=vmin, vmax=vmax, resolution=resolution)
    Out.actor.property.lighting = lighting

    #mlab.view(azimuth=0, elevation=45, figure=fig_mlab)
    #mlab.view(azimuth=Delta_zeta*180/np.pi*0.2, elevation=0, figure=fig_mlab)
    mlab.view(azimuth=Delta_zeta*180/np.pi*0.25 +180, elevation=70, figure=fig_mlab)
    #mlab.view(azimuth=Delta_zeta, elevation=70, figure=fig_mlab)
    imgmap = mlab.screenshot(figure=fig_mlab, mode='rgba')

    fig, ax = get_or_create_ax(fig, ax, figsize=(6,8))
    ax.imshow(imgmap)
    mlab.close()

    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])

    title = get_quantity_label(quantity)

    if remove_zonal:
        title = title+r"$_\mathrm{NZ}$"
    if only_zonal:
        title = title+r"$_\mathrm{Z}$"
    title = title+r"$(t %s/a=%.2f)$" % (get_vt_label(run.ncdata), time_eval if np.ndim(time_eval) == 0 else time_eval[-1])

    if time_avg is not None:
        title = title + r"$_{\Delta t = %.1f}$" % (time_avg)

    fig.suptitle(title)

    return fig, ax, vmin, vmax


def plot_quantity_poloidal_ring(run, quantity="phi", fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, vmin=None, vmax=None, cmap=None, xmin=None, xmax=None, ymin=None, ymax=None, rorigin_fac=2, zed_idx_skip=1, kyfilter_fac=None, ky_lowpass_cutoff=np.inf):

    if kyfilter_fac is not None:
        ky = run.ncdata['ky'][:]
        ky_lowpass_cutoff = ky[1]*kyfilter_fac*0.9999
        print(ky_lowpass_cutoff)

    quantity_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity=quantity, species_idx=species_idx, time_val=time_val, time_idx=time_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, ny=ny, nx=nx, ky_lowpass_cutoff=ky_lowpass_cutoff)

    fig, ax = get_or_create_ax(fig, ax, figsize=(12,10), subplot_kw=dict(projection='polar'))

    if kyfilter_fac is not None:
        ymin = y[0] /(kyfilter_fac-1)
        ymax = y[-1]/(kyfilter_fac-1)
        print(ymin)
        print(ymax)

    if xmin is not None:
        quantity_zed_x_y = quantity_zed_x_y[:,x>xmin]
        x = x[x>xmin]
    if xmax is not None:
        quantity_zed_x_y = quantity_zed_x_y[:,x<xmax]
        x = x[x<xmax]
    if ymin is not None:
        quantity_zed_x_y = quantity_zed_x_y[:,:,y>ymin]
        y = y[y>ymin]
    if ymax is not None:
        quantity_zed_x_y = quantity_zed_x_y[:,:,y<ymax]
        y = y[y<ymax]

    ax.set_rorigin(-rorigin_fac*x[-1])

    #zed = zed[::zed_idx_skip]
    #quantity_zed_x_y = quantity_zed_x_y[::zed_idx_skip,:,:]

    zed = zed[::zed_idx_skip]
    quantity_zed_x_y = quantity_zed_x_y[::zed_idx_skip,:,:]

    nzed = len(zed)
    dzed = zed[1]-zed[0]
    ny   = len(y)
    dy   = y[1]-y[0]
    Ly   = y[-1]-y[0] + dy
    angle     = np.zeros(int(nzed*ny))
    f_x_angle = np.zeros(shape=(len(x),int(nzed*ny)))
    for i_zed in range(nzed):
        angle[int(i_zed*ny):int((i_zed+1)*ny)] = zed[i_zed] + dzed*y/Ly
        #f_x_angle[:,int(i_zed*ny):int((i_zed+1)*ny)] = quantity_zed_x_y[0,:,:]
        f_x_angle[:,int(i_zed*ny):int((i_zed+1)*ny)] = quantity_zed_x_y[i_zed,:,:]

    assert(np.all(np.diff(angle)>=0))

    # Bring back between -pi and pi
    angle = (angle-angle[0])/(angle[-1]-angle[0])*2*np.pi - np.pi

    X, Y = np.meshgrid(x, angle)
    Z = f_x_angle.T

    Zabsmax = np.abs(Z).max()
    if vmin is None:
        vmin = -Zabsmax
    if vmax is None:
        vmax = Zabsmax

    im = ax.pcolormesh(Y, X, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)

    title = get_quantity_label(quantity)

    if remove_zonal:
        title = title+r"$_\mathrm{NZ}$"
    if only_zonal:
        title = title+r"$_\mathrm{Z}$"
    title = title+r"$(t=%.2f)$" % (time_eval if np.ndim(time_eval) == 0 else time_eval[-1])

    if time_avg is not None:
        title = title + r"$_{\Delta t = %.1f}$" % (time_avg)

    fig.suptitle(title)

    return fig, ax, im, vmin, vmax


def plot_quantity_box_zed_x_y(run, quantity="phi", fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, symm=False, vmin=None, vmax=None, kx_lowpass_cutoff=np.inf, ky_lowpass_cutoff=np.inf, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1, cmap=None, xmin=None, xmax=None, ymin=None, ymax=None, zed_neg=True):

    quantity_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity=quantity, species_idx=species_idx, time_val=time_val, time_idx=time_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, ny=ny, nx=nx, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)

    if ax is None:
        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(projection='3d')

    if xmin is not None:
        quantity_zed_x_y = quantity_zed_x_y[:,x>xmin]
        x = x[x>xmin]
    if xmax is not None:
        quantity_zed_x_y = quantity_zed_x_y[:,x<xmax]
        x = x[x<xmax]
    if ymin is not None:
        quantity_zed_x_y = quantity_zed_x_y[:,:,y>ymin]
        y = y[y>ymin]
    if ymax is not None:
        quantity_zed_x_y = quantity_zed_x_y[:,:,y<ymax]
        y = y[y<ymax]

    if zed_neg:
        quantity_zed_x_y = quantity_zed_x_y[zed<0]
        zed = zed[zed<0]


    #xmax = 20
    #x = x[np.abs(x)<xmax]
    #quantity_zed_x_y = quantity_zed_x_y 

    X1, X2, X3 = np.meshgrid(zed, x, y, indexing='ij')
    data    = quantity_zed_x_y
    #data    = np.transpose(quantity_zed_x_y, axes=(1,0,2))

    if symm:
        vmax = (np.abs(data)).max()
        vmin = -vmax

    kw = {
        'vmin': vmin,
        'vmax': vmax,
        'cmap': cmap,
        'levels': 50
    }

    # Plot contour surfaces

    # zed-x (top)
    _ = ax.contourf(
        X1[:, :, -1], X2[:, :, -1], data[:, :, -1],
        zdir='z', offset=y.max(), **kw
    )

    # zed-y (side)
    _ = ax.contourf(
        X1[:, 0, :], data[:, 0, :], X3[:, 0, :],
        zdir='y', offset=x.min(), **kw
    )

    # x-y (front)
    im = ax.contourf(
        data[-1, :, :], X2[-1, :, :], X3[-1, :, :],
        zdir='x', offset=zed.max(), **kw
    )
    ax.set(xlim=[X1.min(), X1.max()], ylim=[X2.min(), X2.max()], zlim=[X3.min(), X3.max()])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_box_aspect((2,1,1))
    ax.view_init(azim=-30, elev=15)
    # --

    title = get_quantity_label(quantity)

    if remove_zonal:
        title = title+r"$_\mathrm{NZ}$"
    if only_zonal:
        title = title+r"$_\mathrm{Z}$"
    title = title+r"$(t=%.2f)$" % (time_eval if np.ndim(time_eval) == 0 else time_eval[-1])

    if time_avg is not None:
        title = title + r"$_{\Delta t = %.1f}$" % (time_avg)

    #ax.set_title(title)
    rho_label = get_rho_label(run.ncdata)
    ax.set_xlabel("\n\n"+r"$\zeta$")
    ax.set_ylabel("\n"+r"$x/%s$" % rho_label)
    ax.set_zlabel(r"$y/%s$" % rho_label)
    fig.suptitle(title)

    return fig, ax, im, vmin, vmax


def plot_quantity_x_y(run, quantity="phi", fig=None, ax=None, zed_val=None, zed_idx=None, mult_zed=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, show_iota_x=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, symm=False, vmin=None, vmax=None, kx_lowpass_cutoff=np.inf, ky_lowpass_cutoff=np.inf, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1, cmap=None, xmin=None, xmax=None, ymin=None, ymax=None, interpolation=False, projection_3d=False, plot_contours=False, suptitle=True, xy_layout=True):

    quantity_x_y, x, y, time_eval = run.get_quantity_x_y(quantity=quantity, zed_val=zed_val, zed_idx=zed_idx, mult_zed=mult_zed, species_idx=species_idx, time_val=time_val, time_idx=time_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, ny=ny, nx=nx, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)

    if ax is None:
        if projection_3d:
            fig, ax = plt.figure(figsize=(12,9)).add_subplot(projection='3d')
        else:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    if xmin is not None:
        quantity_x_y = quantity_x_y[x>xmin]
        x = x[x>xmin]
    if xmax is not None:
        quantity_x_y = quantity_x_y[x<xmax]
        x = x[x<xmax]
    if ymin is not None:
        quantity_x_y = quantity_x_y[:,y>ymin]
        y = y[y>ymin]
    if ymax is not None:
        quantity_x_y = quantity_x_y[:,y<ymax]
        y = y[y<ymax]

    X, Y = np.meshgrid(x, y)
    Z    = quantity_x_y.T

    # NOTE: no real caller passes vmin="symm" here (unlike plot_quantity_x_t/
    # plot_quantity_x_zed/plot_contour_gvmu_vpa, where that string sentinel
    # is the actual convention) -- every caller of this function uses the
    # symm=True kwarg instead, so that's kept as the sole mechanism here.
    if symm:
        vmax = (np.abs(Z)).max()
        vmin = -vmax

    if projection_3d:
        # From https://matplotlib.org/stable/gallery/mplot3d/contour3d_3.html#sphx-glr-gallery-mplot3d-contour3d-3-py
        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph.
        ax.contour(X, Y, Z, zdir='z', offset=vmin,  cmap=cmap)
        ax.contour(X, Y, Z, zdir='x', offset=x[0],  cmap=cmap)
        ax.contour(X, Y, Z, zdir='y', offset=y[-1], cmap=cmap)
        # Plot the 3D surface
        im = ax.plot_surface(X, Y, Z, color="None", edgecolor='k', lw=0.5, rstride=8, cstride=8, alpha=0.1)
        ax.set_xlim([x[0],x[-1]])
        ax.set_ylim([y[0],y[-1]])
        ax.set_zlim([vmin, vmax])

    else:
        if interpolation:
            shading='gouraud'
        else:
            shading='auto'

        if vmin is not None and vmax is not None:
            if cmap is None:
                cmap = 'coolwarm'
            if plot_contours:
                #dx, dy = np.gradient(Z)
                #im = ax.quiver(X, Y, -dx, -dy, scale=50)
                Z = Z/np.abs(Z).max()
                levels = [-1, -2/3, -1/3, 0, 1/3, 2/3, 1]
                im = ax.contour(X, Y, Z, levels=levels, colors=cmap, vmin=vmin, vmax=vmax, linewidths=2)
            else:
                #im = ax.pcolormesh(X, Y, Z, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)

                dx = x[1]-x[0]
                dy = y[1]-y[0]

                if xy_layout:
                    im = ax.imshow(Z, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap, extent=[x.min()-dx/2, x.max()-dx/2, y.min()-dy/2, y.max()-dy/2], aspect='auto', origin='lower')
                else:
                    im = ax.imshow(Z.T, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap, extent=[y.min()-dy/2, y.max()-dy/2, x.min()-dx/2, x.max()-dx/2], aspect='auto', origin='lower')
                #im = ax.pcolormesh(X, Y, Z.T, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)

        else:
            if cmap is None:
                cmap = 'inferno'
            if plot_contours:
                #dZdx, dZdy = np.gradient(Z, x[1]-x[0], y[1]-y[0])
                #magnitude = np.sqrt(dZdx**2 + dZdy**2)
                #dZdx /= magnitude
                #dZdy /= magnitude
                #im = ax.quiver(X, Y, -dZdy, dZdx, scale=30, pivot='middle')#, scale=1e-10*Z.max()/(x[1]-x[0]))
                
                Z = Z/np.abs(Z).max()
                levels = [-1, -2/3, -1/3, 0, 1/3, 2/3, 1]
                im = ax.contour(X, Y, Z, levels=levels, colors=cmap, linewidths=2)
                #im = ax.streamplot(X[0,:], Y[:,0], dx, dy)#, scale=1e-10*Z.max()/(x[1]-x[0]))
            else:

                im = ax.pcolormesh(X, Y, Z, shading=shading, cmap=cmap, rasterized=True)

            ax.set_aspect('equal')


    title = get_quantity_label(quantity)

    if remove_zonal:
        title = title+r"$_\mathrm{NZ}$"
    if only_zonal:
        title = title+r"$_\mathrm{Z}$"
    # time_eval is array-valued whenever a time_avg window was resolved
    # (not just a single reference point) -- show its last (trailing-
    # window-end) value in the title rather than crashing on
    # "%.2f" % array.
    time_eval_title = time_eval if np.ndim(time_eval) == 0 else time_eval[-1]
    title = title+r"$(t=%.2f$ $a/%s)$" % (time_eval_title, get_vt_label(run.ncdata))

    if time_avg is not None:
        title = title + r"$_{\Delta t = %.1f}$" % (time_avg)

    #ax.set_title(title)
    rho_label = get_rho_label(run.ncdata)
    if projection_3d:
        ax.set_xlabel("\n"+r"$x/%s$" % rho_label)
        ax.set_ylabel("\n"+r"$y/%s$" % rho_label)
        ax.set_zlabel("\n"+title)
    else:
        if xy_layout:
            ax.set_xlabel(r"$x/%s$" % rho_label)
            ax.set_ylabel(r"$y/%s$" % rho_label)
        else:
            ax.set_xlabel(r"$y/%s$" % rho_label)
            ax.set_ylabel(r"$x/%s$" % rho_label)
        if suptitle:
            fig.suptitle(title)

    return fig, ax, im, vmin, vmax


def plot_Q_x_y(run, fig=None, ax=None, zed_idx=None, time_idx=-1, species_idx=0, time_val=None):

    Q_x_y, _, _, x, y, time_eval = run.get_Q_x_y(zed_idx=zed_idx, time_val=time_val, time_idx=time_idx, species_idx=species_idx)

    fig, ax = get_or_create_ax(fig, ax, nrows=1, ncols=1, figsize=(12,9))

    X, Y = np.meshgrid(x, y)
    Z = Q_x_y.T

    im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='inferno')

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$Q(t=%.2f)$" % (time_eval if np.ndim(time_eval) == 0 else time_eval[-1]))

    ax.set_aspect('equal')

    return fig, ax, im


def plot_quantity_x(run, quantity="phi", species_idx=0, fig=None, ax=None, zed_idx=None, time_idx=-1, label=None, ls=None, color=None, marker=None, normalise=False, time_avg=None, nx=None, mult_zed=None, kx_order=0, kx_lowpass_cutoff=1e5, mult=1, plot_factor=1):

    f_Z,       x, _, time_eval = run.get_quantity_x_y(quantity=quantity, species_idx=species_idx, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=False, only_zonal=True, kx_order=kx_order, time_avg=time_avg, nx=nx, mult_zed=mult_zed, kx_lowpass_cutoff=kx_lowpass_cutoff)

    # Make 1D array
    f_Z       = mult*f_Z[:,0]

    if normalise:
        norm_val = 1/np.abs(f_Z).max()
        f_Z_plot = f_Z*norm_val*plot_factor
    else:
        norm_val = 1
        f_Z_plot = f_Z*plot_factor

    fig, ax = get_or_create_ax(fig, ax, nrows=1, ncols=1, figsize=(8,5))

    # time_eval is array-valued whenever a time_avg window was resolved --
    # show its last (trailing-window-end) value rather than crashing on
    # "%.2f" % array.
    time_eval_title = time_eval if np.ndim(time_eval) == 0 else time_eval[-1]
    title = r"$t %s/a = %.2f$" % (get_vt_label(run.ncdata), time_eval_title)
    if time_avg is not None:
        title = title + r"$_{\Delta t = %.1f}$" % (time_avg)
    fig.suptitle(title)

    ax.plot(x, f_Z_plot,  ls=ls, c=color, marker=marker, label=label)
    ax.grid(True, alpha=0.5)
    ax.set_xlim(xmin=x[0],xmax=x[-1])

    if label is not None:
        ax.legend()

    ax.set_xlabel(r"$x/%s$" % get_rho_label(run.ncdata))
    return fig, ax, norm_val, x, f_Z


def plot_quantity_x_t(run, quantity, fig=None, ax=None, vmin=None, vmax=None, species_idx=0, logarithmic=False, remove_zonal=False, only_zonal=False, time_idx_skip=1, normalise_each_t=False, y_val=None, cmap='inferno', kx_order=0, zed_val=None, zed_idx=None, mult_zed=None, time_min=0, time_max=1e10, nx=None, kx_lowpass_cutoff=1e4, kx_highpass_cutoff=-1, par_der_order=0, scale_eps=1, return_avg=False, mult=1):

    time_all    =  run.get_time_array(GX_big=True)
    kx, ky, zed = run.get_kx_ky_zed()
    time_idx_min = nearest_index(time_all-time_min)
    time_idx_max = nearest_index(time_all-time_max)
    time      = time_all[time_idx_min:time_idx_max:time_idx_skip]
    time_idxs = range(time_idx_min, time_idx_max, time_idx_skip)
    assert(len(time)==len(time_idxs))

    if nx is None:
        nx = len(kx)

    f_t_x_y = np.zeros( (len(time_idxs), nx, 2*len(ky)-1) )
    for i_idx, time_idx in enumerate(time_idxs):
        print("Evaluating time_idx %.6i/%i..." % (i_idx+1, len(time_idxs)), end="\r")
        f_x_y, x, y, _ = run.get_quantity_x_y(quantity=quantity, remove_zonal=remove_zonal, only_zonal=only_zonal, time_idx=time_idx, kx_order=kx_order, zed_val=zed_val, zed_idx=zed_idx, mult_zed=mult_zed, nx=nx, kx_lowpass_cutoff=kx_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, par_der_order=par_der_order)
        f_t_x_y[i_idx] = f_x_y*mult

    if only_zonal:
        f_t_x = f_t_x_y[:,:,0]
    elif y_val is None:
        # Integrate over y
        print("Note! Integrating over y")
        dy = y[1]-y[0]
        f_t_x = np.sum(f_t_x_y, axis=2)*dy
    else:
        yval_idx = np.argmin( np.abs(y-y_val) )
        f_t_x = f_t_x_y[:,:,yval_idx]

    time  = time*scale_eps
    f_t_x = f_t_x/scale_eps

    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,10))
        plt.subplots_adjust(left=0.15,right=0.95)

    X, Y = np.meshgrid(time, x)
    Z    = f_t_x.T

    if normalise_each_t:
        for time_idx in range(len(time)):
            Z[:,time_idx] = Z[:,time_idx]/max(np.abs(Z[:,time_idx]))

    vmin, vmax = resolve_vmin_vmax(Z, vmin, vmax, logarithmic, default_vmax=Z.max(), fill_vmin_default=False)

    if logarithmic:
        im = ax.pcolormesh(X, Y, np.abs(Z), norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
    else:
        im = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap, rasterized=True)
        #dx = x[1]-x[0]; xmin=x[0]-dx/2; xmax=x[-1]+dx/2
        #dt = time[1]-time[0]; tmin=time[0]-dt/2; tmax=time[-1]+dt/2
        #im = ax.imshow(Z, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest', aspect='auto', origin='lower', extent=[tmin, tmax, xmin, xmax])
        ax.set_xlim([time[0], time[-1]])
        ax.set_ylim([x[0],    x[-1]])

    vt_label = get_vt_label(run.ncdata)
    if scale_eps == 1:
        ax.set_xlabel(r"$t %s/a$" % vt_label)
    else:
        ax.set_xlabel(r"$t %s/R$" % vt_label)
    ax.set_ylabel(r"$x/%s$" % get_rho_label(run.ncdata))

    if return_avg:
        f_t_mean = np.mean(f_t_x, axis=1)
        Lx = x[-1]-x[0]
        f_t_mean_norm = f_t_mean/np.abs(f_t_mean).max() * Lx/4
        return fig, ax, im, time, f_t_mean_norm

    else:
        return fig, ax, im, X, Y, Z
