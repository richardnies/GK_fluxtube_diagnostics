import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})

import stellaDiagnostics as sD

filename_base = "CBC"

#restart_file = "template_restart/restart/CBC.nc.0"
#nc_restart   = nc4.Dataset(restart_file,'r')
#t_restart = nc_restart['t0'].getValue()

#quantity = "pressure_perp"
#quantity = "temperature"
#quantity = "Q_es"
#quantity = "dyphi-dxphi"
#quantity = "dyT-dxphi"
#quantity = "dyT-dyphi"
#quantity = "dyphi-dyphi"
#quantity = "dyphi-dyphi"

#quantity = "dyphi-dxphi"
#quantity = "dyphi-dyphi"
#quantity = "dyphi-P"
quantity = "dyphi-T"
#quantity = "dyPprp-dyphi"
#quantity = "dyPprp-dxphi"
#quantity = "par_mom_transport"

quantity = "P_RH_tot"
#quantity = "P_RH_odd"
#quantity = "P_RH_even"

#plot_around_restart = True
#dt =100
#plot_around_restart = False

kx_order = 0

#quantity = "phi"
#quantity = "upar"
#kx_order = 1

from glob import glob

#dirname_base = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0_higher-vel-res"
dirname_base = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0"
#dirname_base = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001"
#dirname_base = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-4"
#dirname_base = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001"

dirnames = sorted(glob(dirname_base+"/run*00/"))

time_min_vals = np.ones(len(dirnames))*0
time_max_vals = np.ones(len(dirnames))*5000
#time_min_vals = np.ones(len(dirnames))*3000
#time_max_vals = np.ones(len(dirnames))*3500

#time_min_vals = np.array([3000, 3000, 4000, 3000, 3000])-200
#time_max_vals = time_min_vals + 500

normalise=False
#normalise=True

only_zonal = True
remove_zonal = False

#only_zonal = False 
#remove_zonal = True

time_idx_skip_vals = np.ones_like(time_min_vals)*1

cmap='coolwarm'
#cmap='plasma'

vmin="symm"
vmax="last"

if quantity == "Q_es":
    remove_zonal = False
    #logarithmic = True
    logarithmic = False
    if normalise:
        vmin = 0
        vmax = 1
else:
    y_val = 0
    logarithmic = False

mult_zed=None
if quantity in ["dyPprp-dxphi", "dyphi-dxphi"]:
    mult_zed = "nablax-nablax"
elif quantity in ["dyPprp-dyphi", "dyphi-dyphi"]:
   #mult_zed = "nablax-nablay"
    #mult_zed = "nablax-nablax"
    #vmin = 100 
    #vmax = 20000
    normalise=True
    vmax = 1
    vmin = 1e-2
    logarithmic=True
    cmap='plasma'
    #vmax = 1

#mult_zed = "vdriftx"

if quantity in ["Q_es", "dyphi-dxphi", "dyPprp-dxphi", "dyPprp-dyphi", "dyphi-dyphi", "dyphi-T", "par_mom_transport", "P_RH_tot", "P_RH_even", "P_RH_odd"]:
    print("Setting y_val to None.")
    only_zonal = False
    y_val = None

fig, axs = plt.subplots(ncols=len(dirnames), figsize=(8*len(dirnames),8))

for i_dirname, dirname in enumerate(dirnames):

    try:
        time_idx_skip = int(time_idx_skip_vals[i_dirname])

        filename = dirname + "/" + filename_base
        StellaObj = sD.stellaDiagnostics(filename)
        
        if len(dirnames) == 1:
            ax = axs
        else:
            ax = axs[i_dirname]

        fig, ax, im, _, _, _ = StellaObj.plot_quantity_x_t(quantity=quantity, fig=fig, ax=ax, remove_zonal=remove_zonal, only_zonal=only_zonal, vmin=vmin, vmax=vmax, cmap=cmap, logarithmic=logarithmic, normalise_each_t=normalise, time_idx_skip=time_idx_skip, y_val = y_val, kx_order=kx_order, mult_zed=mult_zed, time_min=time_min_vals[i_dirname], time_max=time_max_vals[i_dirname])
        title = dirname[len(dirname_base):]
        ax.set_title(title, fontsize=14)

        plt.colorbar(im, ax=ax)

        # Overplot Q and phi2
        _, _, qflx, time = StellaObj.get_fluxes_over_time()
        phi2_t_kx_ky = StellaObj.ncdata['phi2_vs_kxky'][:]
        kx = StellaObj.ncdata['kx'][:]
        xmax = np.pi/(kx[1]-kx[0])
        phi2_NZ = np.sum(phi2_t_kx_ky[:,:,1:], axis=(1,2))
        phi2_Z_LW = np.sum(phi2_t_kx_ky[:,np.abs(kx)< 0.3,0], axis=1)
        phi2_Z_SW = np.sum(phi2_t_kx_ky[:,np.abs(kx)>=0.3,0], axis=1)
        phi2_Z = phi2_Z_LW+phi2_Z_SW

        time_phi = StellaObj.ncdata['t'][:]
        
        norm = np.log10(qflx.max()) / (xmax/2)
        ax.plot(time, np.log10(qflx)/norm-xmax, c='k', label=r"$Q$")
        
        norm = phi2_Z.max() / (xmax/2)
        ax.plot(time_phi, phi2_Z/norm-xmax, c='0.5', label=r"$(\phi^Z)^2$")
        ax.plot(time_phi, phi2_Z_LW/norm-xmax, c='0.5', label=r"$(\phi^Z)^2 (k_x \rho_i<0.3)$", ls='--', lw=2)
        ax.plot(time_phi, phi2_Z_SW/norm-xmax, c='0.5', label=r"$(\phi^Z)^2 (k_x \rho_i \geq 0.3)$", ls=':', lw=2)
        
        norm = phi2_NZ.max() / (xmax/2)
        ax.plot(time_phi, phi2_NZ/norm-xmax, c='forestgreen', label=r"$(\phi^{NZ})^2$")

    except Exception as e:
        print(e)
        print("Could not load " + dirname)
    
#        if plot_around_restart:
#            if dirname[:7] == "restart":
#                ax.set_xlim([t_restart-dt, t_restart+dt])
#
#        ax.axvline(t_restart, ls='--', c='k')
    
if kx_order == 1:
    title = r"$\partial_x$"
elif kx_order == 2:
    title = r"$\partial^2_x$"
else:
    title = ""

if quantity == "phi":
    title += r"$\varphi$"
elif quantity == "temperature":
    title += r"$T$"
elif quantity == "pressure_perp":
    title += r"$P_\perp$"
elif quantity == "Q_es":
    title += r"$\int\mathrm{d}y\; T \partial_y \varphi$"
elif quantity == "dyphi-dxphi":
    title += r"$\int\mathrm{d}y\; \partial_x \varphi \partial_y \varphi$"
elif quantity == "dyT-dxphi":
    title += r"$\int\mathrm{d}y\; \partial_x \varphi \partial_y T$"
elif quantity == "dyT-dyphi":
    title += r"$\int\mathrm{d}y\; \partial_y \varphi \partial_y T$"
elif quantity == "dyphi-dyphi":
    title += r"$\int\mathrm{d}y\; (\partial_y \varphi)^2 $"
elif quantity == "dyphi-P":
    title += r"$\int\mathrm{d}y\; \partial_y \varphi P$"
elif quantity == "dyphi-T":
    title += r"$\int\mathrm{d}y\; \partial_y \varphi T$"

if remove_zonal:
    title = title+r"$_\mathrm{NZ}(y=0$"
elif only_zonal:
    title = title+r"$_\mathrm{Z}$"

if mult_zed == "vdriftx":
    title = title + r"$v_{Dx}$"

fig.suptitle(title)
plt.tight_layout()

#add_colorbar = True
##add_colorbar = False
#if add_colorbar:
#    fig.subplots_adjust(right=0.8)
#    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#    fig.colorbar(im, cax=cbar_ax)

figname = "fig_contours_"+dirname_base+"_"+quantity+"_x_t"

if remove_zonal:
    figname = figname + "_remove_zonal"
if only_zonal:
    figname = figname + "_only_zonal"
if not normalise:
    figname = figname + "_unnormalised"

if kx_order > 0:
    figname = figname + "_kxorder-%i" % (kx_order)

if mult_zed == "vdriftx":
    figname = figname + "_vdriftx"

#if plot_around_restart:
#    figname = figname + "_zoom_restart"
plt.tight_layout()
plt.savefig(figname+".pdf")
