import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import traceback
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 20, 
    "axes.titlepad": 15,
})

import stellaDiagnostics as sD

# Load data
filename_base = "/CBC"

from glob import glob
import seaborn as sns

# Default values
rhoc = 0.5
time_idx_skip = 10

# Base directory

# tprim=8.4
time_val_avg = None
time_avg = 5

### tprim=6.3
#time_val_avg = 750
#time_avg = 500
#
# tprim=5.95, 5.60, 5.25
#time_val_avg = None
#time_avg = 300
#
### tprim=4.9
#time_val_avg = 2000
#time_avg = 500
#
### tprim=4.9
#time_val_avg = 2000
#time_avg = 500

## tprim=4.2
#time_val_avg = 1500
#time_avg = 1000

# tprim=3.85
#time_val_avg = 1200
#time_avg = 600

# Subdirectories
#dirnames = sorted(glob("*/run_*00/"))
dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-*/run_tprim*00/"))
dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-*/run_tprim*00_restart*/"))
#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-*/run_tprim*8.4000/"))
#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-*/run_tprim*6.3000/"))
#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-*/run_tprim*5.9500/"))
#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-*/run_tprim*5.6000/"))

#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0*/run_tprim*5.2500/"))
#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0*/run_tprim*4.9000/"))
#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0*/run_tprim*4.2000/"))
#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0*/run_tprim*3.8500/"))

#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim*4.9000/"))
#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim*8.4000/"))
#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim*4.2000/"))
#dirnames = sorted(glob("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0*/run_tprim*00/"))

avg_norm = 2
#avg_norm = "center"

time_max = 1e10
qflx_rel_idx_min = 1e-7
qflx_rel_idx_max = 1e-3

kx_max = 0.3

exp_avg = 2

#time_avg_frac = 0.2

#ndirs = len(dirnames)
#colors = sns.color_palette("hls", ndirs)

reeval = False
#reeval = True

for i_dir, dirname in enumerate(dirnames):

    try:

        if exists(dirname+'data_Dimits.json') and not reeval:
            continue

        filename = dirname+filename_base
        diagObj = sD.stellaDiagnostics(filename)

        fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(46,16))

        tprim = float(diagObj.ncdata.variables['tprim'][0])
        qinp  = float(diagObj.ncdata.variables['q'].getValue())

        # Load heat flux and extract maximum linear growth rate        
        _, _, qflx, time = diagObj.get_fluxes_over_time(load_from_nc=True)
#        time_avg = time_avg_frac * time[-1]

        qflx=qflx[time<time_max]
        time=time[time<time_max]

        if time_val_avg is None:
            qflx_avg = np.mean(qflx[time > time[-1]-time_avg])
            qflx_std = np.std( qflx[time > time[-1]-time_avg])
        else:
            qflx_avg = np.mean(qflx[(time > time_val_avg-time_avg/2) & (time < time_val_avg+time_avg/2)])
            qflx_std = np.std( qflx[(time > time_val_avg-time_avg/2) & (time < time_val_avg+time_avg/2)])

        idx_qflx_max = max(np.argmax(qflx),5)
        idx_max = np.argmin( np.abs(qflx[:idx_qflx_max] - qflx_rel_idx_max*qflx.max()) )
        idx_min = np.argmin( np.abs(qflx[:idx_qflx_max] - qflx_rel_idx_min*qflx.max()) )

        gamma_max = 0.5 * np.log(qflx_rel_idx_max/qflx_rel_idx_min)/(time[idx_max]-time[idx_min])

        ax = axs[0,0]
        ax.axvline(time[idx_min], c='crimson', alpha=0.5)
        ax.axvline(time[idx_max], c='crimson', alpha=0.5)
        ax.semilogy(time, qflx, c='k', label=r"$Q/Q_\mathrm{gB}$")
        ax.semilogy(time[:idx_max], qflx[idx_min]*np.exp(2*gamma_max*(time[:idx_max]-time[idx_min])), ls='--', c='crimson', label=r"$\sim$ e$^{\gamma_\mathrm{lin}^\mathrm{max}t}$")
        ax.axhline(qflx_avg, c='k', ls='--')
        ax.fill_between(time, (qflx_avg-qflx_std)*np.ones_like(time), (qflx_avg+qflx_std)*np.ones_like(time), color='k', alpha=0.25)

        # Plot more quantities on Q plot
        time = diagObj.get_time_array()
        gammaE_t    = np.zeros_like(time[::time_idx_skip])
        upar_t      = np.zeros_like(time[::time_idx_skip])
        uparcos_t   = np.zeros_like(time[::time_idx_skip])
        dxT_t       = np.zeros_like(time[::time_idx_skip])
        gammaE_RH_t = np.zeros_like(time[::time_idx_skip])
        phi2_vEpos_t = np.zeros_like(time[::time_idx_skip])
        phi2_vEneg_t = np.zeros_like(time[::time_idx_skip])

        i_t = 0
        for time_idx in np.arange(len(time))[::time_idx_skip]:
            dxphi_x_y, x, _, _   = diagObj.get_quantity_x_y(quantity="phi", only_zonal=True, kx_order=1, time_idx=time_idx)
            dx2phi_x_y, x, _, _   = diagObj.get_quantity_x_y(quantity="phi", only_zonal=True, kx_order=2, time_idx=time_idx)
            dx2phiRH_x_y, x, _, _ = diagObj.get_quantity_x_y(quantity="RH_phi", only_zonal=True, kx_order=2, time_idx=time_idx)
            upar_x_y, x, _, _     = diagObj.get_quantity_x_y(quantity="upar", only_zonal=True, kx_order=0, time_idx=time_idx)
            uparcos_x_y, x, _, _  = diagObj.get_quantity_x_y(quantity="upar", only_zonal=True, kx_order=0, time_idx=time_idx, mult_zed="cos")
            dxT_x_y, x, _, _      = diagObj.get_quantity_x_y(quantity="temperature", only_zonal=True, kx_order=1, time_idx=time_idx)
            phi2_x_y, x, y, _   = diagObj.get_quantity_x_y(quantity="phi", ky_order=0, kx_order=0, abs_squared=True, time_idx=time_idx, remove_zonal=True)
            vEzonal_x = -dxphi_x_y[:,0]

            gammaE_t[i_t]       = (np.mean(dx2phi_x_y[  :,0]**exp_avg   ))**(1/exp_avg) 
            gammaE_RH_t[i_t]    = (np.mean(dx2phiRH_x_y[:,0]**exp_avg   ))**(1/exp_avg)
            upar_t[i_t]         = (np.mean(upar_x_y[    :,0]**exp_avg   ))**(1/exp_avg) 
            uparcos_t[i_t]      = (np.mean(uparcos_x_y[ :,0]**exp_avg   ))**(1/exp_avg) 
            dxT_t[i_t]          = (np.mean(dxT_x_y[     :,0]**exp_avg   ))**(1/exp_avg) 
            phi2_vEpos_t[i_t]   = np.sum(phi2_x_y[vEzonal_x>0,:])*(x[1]-x[0])*(y[1]-y[0])
            phi2_vEneg_t[i_t]   = np.sum(phi2_x_y[vEzonal_x<0,:])*(x[1]-x[0])*(y[1]-y[0])
            i_t += 1

        ax.semilogy(time[::time_idx_skip], gammaE_t,  c='orange',  label=r"$\langle \gamma_E^2(x) \rangle^{1/2}$")
        ax.semilogy(time[::time_idx_skip], gammaE_RH_t,  c='orange',  label=r"$\langle \gamma_{E,\mathrm{RH}}^2(x) \rangle^{1/2}$", alpha=0.5, lw=2)
        ax.semilogy(time[::time_idx_skip], upar_t,    c='forestgreen', label=r"$\langle u_\parallel^2(x) \rangle^{1/2}$")
        ax.semilogy(time[::time_idx_skip], uparcos_t, c='forestgreen', label=r"$\langle (u_\parallel \cos\theta)^2(x) \rangle^{1/2}$", ls='--')
#        ax.semilogy(time[::time_idx_skip], dxT_t,     c='c',           label=r"$\langle (\partial_x T)^2(x) \rangle^{1/2}$")

        norm_phi2 = 2*np.max(qflx)/np.nanmax(np.array([phi2_vEpos_t, phi2_vEneg_t])) 

        ax.semilogy(time[::time_idx_skip], phi2_vEpos_t*norm_phi2, c='mediumblue', label=r"$\langle \tilde\varphi^2 (v_E>0) \rangle$")
        ax.semilogy(time[::time_idx_skip], phi2_vEneg_t*norm_phi2, c='crimson',    label=r"$\langle \tilde\varphi^2 (v_E<0) \rangle$")
        ax.set_xlabel(r"$t v_{Ti}/a$")
#        ax.set_ylabel(r"$Q/Q_\mathrm{gB}$")
        ax.grid(True)
        ax.legend()
#        ax.set_xlim([0, time[-1]])
        #ax.set_ylim(ymin=qflx[10])
        ax.set_ylim(ymin=1e-3*qflx[idx_max])

        if time_val_avg is None:
            ax.fill_betweenx(ax.get_ylim(), time[-1]-time_avg, time[-1], color='0.5', alpha=0.15)
        else:
            ax.fill_betweenx(ax.get_ylim(), time_val_avg-time_avg/2, time_val_avg+time_avg/2, color='0.5', alpha=0.15)

        # Load vE, shear, dxT, upar, and extract gammaE, ...
        dxphi_x_y, x, _, _        = diagObj.get_quantity_x_y(quantity="phi",         time_val=time_val_avg, only_zonal=True, kx_order=1, time_avg=time_avg)
        dxphi_RH_x_y, x, _, _     = diagObj.get_quantity_x_y(quantity="RH_phi",      time_val=time_val_avg, only_zonal=True, kx_order=1, time_avg=time_avg)
        dx2phi_x_y, x, _, _       = diagObj.get_quantity_x_y(quantity="phi",         time_val=time_val_avg, only_zonal=True, kx_order=2, time_avg=time_avg)
        dx2phi_RH_x_y, x, _, _    = diagObj.get_quantity_x_y(quantity="RH_phi",      time_val=time_val_avg, only_zonal=True, kx_order=2, time_avg=time_avg)
        dx2phi_LW_x_y, x, _, _    = diagObj.get_quantity_x_y(quantity="phi",         time_val=time_val_avg, only_zonal=True, kx_order=2, time_avg=time_avg, kxmin_filter=kx_max)
        dx2phi_RH_LW_x_y, x, _, _ = diagObj.get_quantity_x_y(quantity="RH_phi",      time_val=time_val_avg, only_zonal=True, kx_order=2, time_avg=time_avg, kxmin_filter=kx_max)
        dxT_x_y, x, _, _          = diagObj.get_quantity_x_y(quantity="temperature", time_val=time_val_avg, only_zonal=True, kx_order=1, time_avg=time_avg)
        upar_x_y, x, _, _         = diagObj.get_quantity_x_y(quantity="upar",        time_val=time_val_avg, only_zonal=True, kx_order=0, time_avg=time_avg)
        dxupar_x_y, x, _, _       = diagObj.get_quantity_x_y(quantity="upar",        time_val=time_val_avg, only_zonal=True, kx_order=1, time_avg=time_avg)
        uparcos_x_y, x, _, _      = diagObj.get_quantity_x_y(quantity="upar",        time_val=time_val_avg, only_zonal=True, kx_order=0, time_avg=time_avg, mult_zed="cos")
        dyphi2_x_y, x, _, _       = diagObj.get_quantity_x_y(quantity="phi",         time_val=time_val_avg, ky_order=1,      kx_order=0, time_avg=time_avg, abs_squared=True)

        # Evaluate time-averaged momentum transport and P_RH 
        if time_val_avg is None:
            time_idxs = np.argwhere( time>time[-1]-time_avg ).flatten()
        else:
            time_idxs = np.argwhere( (time>time_val_avg-time_avg/2) & (time<time_val_avg+time_avg/2) ).flatten()

        dt_vals = np.gradient(time[time_idxs])

        dl_over_B_avg = diagObj.dl_over_B_avg()

        dE_Pi_parallel_x    = np.zeros_like(x)
        dE_Pi_perp_x        = np.zeros_like(x)
        P_RH_even_x         = np.zeros_like(x) 
        P_RH_odd_x          = np.zeros_like(x) 
        P_RH_even_passing_x = np.zeros_like(x) 
        P_RH_odd_passing_x  = np.zeros_like(x) 
        P_RH_even_trapped_x = np.zeros_like(x) 
        P_RH_odd_trapped_x  = np.zeros_like(x) 

        for i, time_idx in enumerate(time_idxs):
            dx_Pi_parallel_zed_x_y, _, x, _, _  = diagObj.get_quantity_zed_x_y(quantity="par_mom_transport", only_zonal=True, kx_order=1, time_idx=time_idx)
            dx_Pi_perp_zed_x_y, _, x, _, _  = diagObj.get_quantity_zed_x_y(quantity="Reynolds", only_zonal=True, kx_order=1, time_idx=time_idx)

            uparZ_zed_x_y, _, x, _, _      = diagObj.get_quantity_zed_x_y(quantity="upar", only_zonal=True, kx_order=0, time_idx=time_idx)

            dE_Pi_parallel_x += -np.sum(dl_over_B_avg[:,None]*dx_Pi_parallel_zed_x_y[:,:,0]*uparZ_zed_x_y[:,:,0], axis=0) * dt_vals[i]/np.sum(dt_vals)
            dE_Pi_perp_x += -np.sum(dl_over_B_avg[:,None]*dx_Pi_perp_zed_x_y[:,:,0], axis=0)*dxphi_x_y[:,0] * dt_vals[i]/np.sum(dt_vals)

            # P_RH
            try:
                RH_flux_phi_even_x_y, x, _, _  = diagObj.get_quantity_x_y(quantity="RH_fluxes_phi_even", only_zonal=True, kx_order=0, time_idx=time_idx)
                RH_flux_phi_odd_x_y,  x, _, _  = diagObj.get_quantity_x_y(quantity="RH_fluxes_phi_odd",  only_zonal=True, kx_order=0, time_idx=time_idx)
                nan_array = np.zeros_like(RH_flux_phi_even_x_y); nan_array[:] = np.nan
                RH_flux_phi_even_passing_x_y = nan_array
                RH_flux_phi_odd_passing_x_y  = nan_array
                RH_flux_phi_even_trapped_x_y = nan_array
                RH_flux_phi_odd_trapped_x_y  = nan_array
            except Exception as e:
                #print(e)
                RH_flux_phi_even_passing_x_y, x, _, _  = diagObj.get_quantity_x_y(quantity="RH_fluxes_phi_even_passing", only_zonal=True, kx_order=0, time_idx=time_idx)
                RH_flux_phi_odd_passing_x_y,  x, _, _  = diagObj.get_quantity_x_y(quantity="RH_fluxes_phi_odd_passing",  only_zonal=True, kx_order=0, time_idx=time_idx)
                RH_flux_phi_even_trapped_x_y, x, _, _  = diagObj.get_quantity_x_y(quantity="RH_fluxes_phi_even_trapped", only_zonal=True, kx_order=0, time_idx=time_idx)
                RH_flux_phi_odd_trapped_x_y,  x, _, _  = diagObj.get_quantity_x_y(quantity="RH_fluxes_phi_odd_trapped",  only_zonal=True, kx_order=0, time_idx=time_idx)
                RH_flux_phi_even_x_y = RH_flux_phi_even_passing_x_y + RH_flux_phi_even_trapped_x_y
                RH_flux_phi_odd_x_y  = RH_flux_phi_odd_passing_x_y  + RH_flux_phi_odd_trapped_x_y

            RH_flux_phi_even_x = RH_flux_phi_even_x_y[:,0]
            RH_flux_phi_odd_x  = RH_flux_phi_odd_x_y[ :,0]
            RH_flux_phi_even_passing_x = RH_flux_phi_even_passing_x_y[:,0]
            RH_flux_phi_odd_passing_x  = RH_flux_phi_odd_passing_x_y[ :,0]
            RH_flux_phi_even_trapped_x = RH_flux_phi_even_trapped_x_y[:,0]
            RH_flux_phi_odd_trapped_x  = RH_flux_phi_odd_trapped_x_y[ :,0]
            RH_flux_phi_even_x = RH_flux_phi_even_x_y[:,0]
            RH_flux_phi_odd_x  = RH_flux_phi_odd_x_y[ :,0]
            RH_flux_phi_even_passing_x = RH_flux_phi_even_passing_x_y[:,0]
            RH_flux_phi_odd_passing_x  = RH_flux_phi_odd_passing_x_y[ :,0]
            RH_flux_phi_even_trapped_x = RH_flux_phi_even_trapped_x_y[:,0]
            RH_flux_phi_odd_trapped_x  = RH_flux_phi_odd_trapped_x_y[ :,0]

            dxphi_RH_x_y_inst, x, _, _  = diagObj.get_quantity_x_y(quantity="RH_phi", only_zonal=True, kx_order=1, time_idx=time_idx)
            vE_RH_x_inst = -dxphi_RH_x_y_inst[:,0]

            P_RH_even_x         += -RH_flux_phi_even_x         * vE_RH_x_inst * dt_vals[i]/np.sum(dt_vals) 
            P_RH_odd_x          += -RH_flux_phi_odd_x          * vE_RH_x_inst * dt_vals[i]/np.sum(dt_vals)
            P_RH_even_passing_x += -RH_flux_phi_even_passing_x * vE_RH_x_inst * dt_vals[i]/np.sum(dt_vals)
            P_RH_odd_passing_x  += -RH_flux_phi_odd_passing_x  * vE_RH_x_inst * dt_vals[i]/np.sum(dt_vals)
            P_RH_even_trapped_x += -RH_flux_phi_even_trapped_x * vE_RH_x_inst * dt_vals[i]/np.sum(dt_vals)
            P_RH_odd_trapped_x  += -RH_flux_phi_odd_trapped_x  * vE_RH_x_inst * dt_vals[i]/np.sum(dt_vals)

        # Get zonal components
        vE_x           = -dxphi_x_y[:,0]
        vE_RH_x        = -dxphi_RH_x_y[:,0]
        gammaE_x       = -dx2phi_x_y[:,0]
        gammaE_RH_x    = -dx2phi_RH_x_y[:,0]
        gammaE_LW_x    = -dx2phi_LW_x_y[:,0]
        gammaE_RH_LW_x = -dx2phi_RH_LW_x_y[:,0]
        dxT_x          = dxT_x_y[:,0]
        upar_x         = upar_x_y[:,0]
        dxupar_x       = dxupar_x_y[:,0]
        uparcos_x      = uparcos_x_y[:,0]
        dyphi2_x = dyphi2_x_y[ :,0]

        gammaE_avg       = np.sqrt(np.mean(gammaE_x**2))
        gammaE_RH_avg    = np.sqrt(np.mean(gammaE_RH_x**2))
        gammaE_std       = np.sqrt(np.mean((np.abs(gammaE_x)   -gammaE_avg   )**2))
        gammaE_RH_std    = np.sqrt(np.mean((np.abs(gammaE_RH_x)-gammaE_RH_avg)**2))
        gammaE_LW_avg    = np.sqrt(np.mean(gammaE_LW_x**2))
        gammaE_RH_LW_avg = np.sqrt(np.mean(gammaE_RH_LW_x**2))
        gammaE_LW_std    = np.sqrt(np.mean((np.abs(gammaE_LW_x)   -gammaE_LW_avg   )**2))
        gammaE_RH_LW_std = np.sqrt(np.mean((np.abs(gammaE_RH_LW_x)-gammaE_RH_LW_avg)**2))
        vE_avg           = np.sqrt(np.mean(vE_x**2))
        vE_RH_avg        = np.sqrt(np.mean(vE_RH_x**2))
        upar_avg         = np.sqrt(np.mean(upar_x**2))
        uparcos_avg      = np.sqrt(np.mean(uparcos_x**2))
        dxT_avg          = np.sqrt(np.mean(dxT_x**2))

        # Evaluation of average dyphi2 and P_RH over whole box and over regions of positive/negative vE
        dx = x[1]-x[0]
        P_RH_even_avg_alt         = np.sum(dx*P_RH_even_x)
        P_RH_odd_avg_alt          = np.sum(dx*P_RH_odd_x)
        P_RH_even_passing_avg_alt = np.sum(dx*P_RH_even_passing_x)
        P_RH_odd_passing_avg_alt  = np.sum(dx*P_RH_odd_passing_x)
        P_RH_even_trapped_avg_alt = np.sum(dx*P_RH_even_trapped_x)
        P_RH_odd_trapped_avg_alt  = np.sum(dx*P_RH_odd_trapped_x)
        dyphi2_avg                = np.sum(dx*dyphi2_x)
        dE_Pi_parallel_avg        = np.sum(dx*dE_Pi_parallel_x)
        dE_Pi_perp_avg            = np.sum(dx*dE_Pi_perp_x)
        P_RH_even_avg_alt_vEpos   = np.sum(dx*P_RH_even_x[vE_x>=0])
        P_RH_odd_avg_alt_vEpos    = np.sum(dx*P_RH_odd_x[vE_x>=0])
        P_RH_even_passing_avg_alt_vEpos = np.sum(dx*P_RH_even_passing_x[vE_x>=0])
        P_RH_odd_passing_avg_alt_vEpos  = np.sum(dx*P_RH_odd_passing_x[vE_x>=0])
        P_RH_even_trapped_avg_alt_vEpos = np.sum(dx*P_RH_even_trapped_x[vE_x>=0])
        P_RH_odd_trapped_avg_alt_vEpos  = np.sum(dx*P_RH_odd_trapped_x[vE_x>=0])
        dyphi2_avg_vEpos          = np.sum(dx*dyphi2_x[vE_x>=0])
        dE_Pi_parallel_avg_vEpos  = np.sum(dx*dE_Pi_parallel_x[vE_x>=0])
        dE_Pi_perp_avg_vEpos      = np.sum(dx*dE_Pi_perp_x[vE_x>=0])
        P_RH_even_avg_alt_vEneg   = np.sum(dx*P_RH_even_x[vE_x<0])
        P_RH_odd_avg_alt_vEneg    = np.sum(dx*P_RH_odd_x[vE_x<0])
        P_RH_even_passing_avg_alt_vEneg = np.sum(dx*P_RH_even_passing_x[vE_x<0])
        P_RH_odd_passing_avg_alt_vEneg  = np.sum(dx*P_RH_odd_passing_x[vE_x<0])
        P_RH_even_trapped_avg_alt_vEneg = np.sum(dx*P_RH_even_trapped_x[vE_x<0])
        P_RH_odd_trapped_avg_alt_vEneg  = np.sum(dx*P_RH_odd_trapped_x[vE_x<0])
        dyphi2_avg_vEneg          = np.sum(dx*dyphi2_x[vE_x<0])
        dE_Pi_parallel_avg_vEneg  = np.sum(dx*dE_Pi_parallel_x[vE_x<0])
        dE_Pi_perp_avg_vEneg      = np.sum(dx*dE_Pi_perp_x[vE_x<0])

        ax = axs[0,1]
        ax.plot(x,  gammaE_x,    c='k')
        ax.plot(x,  gammaE_RH_x, c='k', alpha=0.5)
        ax.plot(x,  dxT_x,       c='forestgreen', label=r"$\partial_x T$")
        ax.axhline( gammaE_avg,    c='mediumblue', label=r"$\langle \gamma_E^2(x) \rangle^{1/2}$")
        ax.axhline(-gammaE_avg,    c='mediumblue')
        ax.axhline( gammaE_RH_avg, c='mediumblue', alpha=0.5, label=r"$\langle \gamma_{E, \mathrm{RH}}^2(x) \rangle^{1/2}$")
        ax.axhline(-gammaE_RH_avg, c='mediumblue', alpha=0.5)
        #ax.fill_between(x, gammaE_avg-gammaE_std*np.ones_like(x), gammaE_avg+gammaE_std*np.ones_like(x), color='mediumblue', alpha=0.25)
        #ax.fill_between(x,-gammaE_avg-gammaE_std*np.ones_like(x),-gammaE_avg+gammaE_std*np.ones_like(x), color='mediumblue', alpha=0.25)
#        ax.axhline( gamma_max,  c='crimson', label=r"$\gamma_\mathrm{lin}^\mathrm{max}$")
#        ax.axhline(-gamma_max,  c='crimson')
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(r"$x/\rho_i$")
        ax.set_xlim([x[0], x[-1]])

        # Estimate eps
        bmag = diagObj.ncdata.variables['bmag'][:]
        bmag_inv = 1/bmag
        eps = (bmag_inv.max()-bmag_inv.min())/(bmag_inv.max()+bmag_inv.min())
        print("epsilon = %e" % (eps))

        ax = axs[1,1]
        ax.plot(x,  vE_x, c='k', label=r"$v_E$")
        ax.plot(x,  vE_RH_x, c='k', alpha=0.5)
        ax.plot(x,  2*upar_x/(1.6*np.sqrt(eps)*qinp), c='orange', label=r"$\langle u_\parallel \rangle_\theta/(1.6 \epsilon^{1/2} q)$")
        ax.plot(x,  2*uparcos_x*2/(2*qinp), c='forestgreen', label=r"$2 \langle u_\parallel \cos\theta \rangle_\theta / (2q)$")

        norm = np.abs(vE_x).max() / np.abs(dyphi2_x).max()
        ax.plot(x, dyphi2_x*norm, c='c', label=r"$v_{Ex}^2$")

        norm = np.abs(vE_x).max() / max(np.abs(P_RH_even_x).max(), np.abs(P_RH_odd_x).max())
        ax.plot(x, P_RH_even_x*norm, c='crimson', lw=2, label=r"$\mathcal{P}_\varphi^+$")
        ax.plot(x, P_RH_odd_x *norm, c='mediumblue', lw=2, label=r"$\mathcal{P}_\varphi^-$")
        ax.axhline(P_RH_even_avg_alt*norm/2, c='crimson', lw=2, alpha=0.5)
        ax.axhline(P_RH_odd_avg_alt *norm/2, c='mediumblue', lw=2, alpha=0.5)

        ax.hlines(P_RH_even_avg_alt_vEpos*norm, xmin=min(x[(vE_x>0) & (x<0)]), xmax=max(x[(vE_x>0) & (x<0)]), colors='crimson',    lw=1, alpha=0.5)
        ax.hlines(P_RH_odd_avg_alt_vEpos *norm, xmin=min(x[(vE_x>0) & (x<0)]), xmax=max(x[(vE_x>0) & (x<0)]), colors='mediumblue', lw=1, alpha=0.5)
        ax.hlines(P_RH_even_avg_alt_vEpos*norm, xmin=min(x[(vE_x>0) & (x>0)]), xmax=max(x[(vE_x>0) & (x>0)]), colors='crimson',    lw=1, alpha=0.5)
        ax.hlines(P_RH_odd_avg_alt_vEpos *norm, xmin=min(x[(vE_x>0) & (x>0)]), xmax=max(x[(vE_x>0) & (x>0)]), colors='mediumblue', lw=1, alpha=0.5)

        ax.hlines(P_RH_even_avg_alt_vEneg*norm, xmin=min(x[(vE_x<0) & (x<0)]), xmax=max(x[(vE_x<0) & (x<0)]), colors='crimson',    lw=1, alpha=0.5)
        ax.hlines(P_RH_odd_avg_alt_vEneg *norm, xmin=min(x[(vE_x<0) & (x<0)]), xmax=max(x[(vE_x<0) & (x<0)]), colors='mediumblue', lw=1, alpha=0.5)
        ax.hlines(P_RH_even_avg_alt_vEneg*norm, xmin=min(x[(vE_x<0) & (x>0)]), xmax=max(x[(vE_x<0) & (x>0)]), colors='crimson',    lw=1, alpha=0.5)
        ax.hlines(P_RH_odd_avg_alt_vEneg *norm, xmin=min(x[(vE_x<0) & (x>0)]), xmax=max(x[(vE_x<0) & (x>0)]), colors='mediumblue', lw=1, alpha=0.5)
        

        norm = np.abs(vE_x).max() / np.abs(dE_Pi_parallel_x).max()
        ax.plot(x, dE_Pi_parallel_x *norm, c='brown', label=r"$- u_\parallel^Z \partial_x\Pi_\parallel $")

        norm = np.abs(vE_x).max() / np.abs(dE_Pi_perp_x).max()
        ax.plot(x, dE_Pi_perp_x *norm, c='pink', label=r"$- v_E^Z \partial_x\Pi_\perp $")

        #norm = np.abs(vE_x).max() / np.abs(Pi_parallel_x).max()
        #ax.plot(x, Pi_parallel_x *norm, c='brown', ls='-.', label=r"$\Pi_\parallel$")

#        x_plot = x[np.abs(x)<x[-1]/2]
#        ax.plot(x_plot,  gammaE_avg*x_plot, c='mediumblue', ls='--', label=r"$x \langle \gamma_E^2(x) \rangle^{1/2}$")
#        ax.plot(x_plot, -gammaE_avg*x_plot, c='mediumblue', ls='--')
#        ax.plot(x_plot,  gammaE_RH_avg*x_plot, c='mediumblue', ls='--', alpha=0.5)
#        ax.plot(x_plot, -gammaE_RH_avg*x_plot, c='mediumblue', ls='--', alpha=0.5)
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(r"$x/\rho_i$")
        ax.set_xlim([x[0], x[-1]])

        # Load and plot P_RH
        P_RH_even_t_kx, P_RH_odd_t_kx, time, kx = diagObj.get_P_RH(time_max=time_max, time_idx_skip=time_idx_skip, fcoll=0)
        P_RH_coll_even_t_kx, P_RH_coll_odd_t_kx, time, kx = diagObj.get_P_RH(time_max=time_max, time_idx_skip=time_idx_skip, fcoll=1, fphi=0)
        P_RH_coll_t_kx = P_RH_coll_even_t_kx + P_RH_coll_odd_t_kx

        P_RH_even_t = np.sum(P_RH_even_t_kx, axis=1)
        P_RH_odd_t  = np.sum(P_RH_odd_t_kx,  axis=1)
        P_RH_coll_t = np.sum(P_RH_coll_t_kx, axis=1)

        P_RH_even_t_LW = np.sum(P_RH_even_t_kx[:,np.abs(kx)<=kx_max], axis=1)
        P_RH_odd_t_LW  = np.sum(P_RH_odd_t_kx[ :,np.abs(kx)<=kx_max], axis=1)
        P_RH_coll_t_LW = np.sum(P_RH_coll_t_kx[:,np.abs(kx)<=kx_max], axis=1)

        dt = np.gradient(time)

        if time_val_avg is None:
            idxs_avg = np.argwhere( time>time[-1]-time_avg ).flatten()
        else:
            idxs_avg = np.argwhere( (time>time_val_avg-time_avg/2) & (time<time_val_avg+time_avg/2) ).flatten()

        P_RH_even_avg    = np.sum((dt*P_RH_even_t)[   idxs_avg])/np.sum(dt[idxs_avg]) 
        P_RH_odd_avg     = np.sum((dt*P_RH_odd_t)[    idxs_avg])/np.sum(dt[idxs_avg])
        P_RH_coll_avg    = np.sum((dt*P_RH_coll_t)[   idxs_avg])/np.sum(dt[idxs_avg]) 
        P_RH_even_avg_LW = np.sum((dt*P_RH_even_t_LW)[idxs_avg])/np.sum(dt[idxs_avg])
        P_RH_odd_avg_LW  = np.sum((dt*P_RH_odd_t_LW)[ idxs_avg])/np.sum(dt[idxs_avg])
        P_RH_coll_avg_LW = np.sum((dt*P_RH_coll_t_LW)[idxs_avg])/np.sum(dt[idxs_avg])
        P_RH_even_std    = np.std((P_RH_even_t)[      idxs_avg])
        P_RH_odd_std     = np.std((P_RH_odd_t)[       idxs_avg])
        P_RH_coll_std    = np.std((P_RH_coll_t)[      idxs_avg])
        P_RH_even_std_LW = np.std((P_RH_even_t_LW)[   idxs_avg])
        P_RH_odd_std_LW  = np.std((P_RH_odd_t_LW)[    idxs_avg])
        P_RH_coll_std_LW = np.std((P_RH_coll_t_LW)[   idxs_avg])

        ax = axs[1,0]
        ax.plot(time, P_RH_even_t,    c='crimson',    label=r"$P_{\mathrm{RH}}^+$")
        ax.plot(time, P_RH_odd_t,     c='mediumblue', label=r"$P_{\mathrm{RH}}^-$")
        ax.plot(time, P_RH_coll_t,    c='orange',    label=r"$P_{\mathrm{RH}}^C$")
        ax.plot(time, P_RH_coll_t+P_RH_even_t+P_RH_odd_t,     c='k', label=r"$P_{\mathrm{RH}}$", alpha=0.5)
#        ax.plot(time, P_RH_even_t_LW, c='crimson',    label=r"$P_{\mathrm{RH},LW}^+$", alpha=0.5, lw=2)
#        ax.plot(time, P_RH_odd_t_LW,  c='mediumblue', label=r"$P_{\mathrm{RH},LW}^-$", alpha=0.5, lw=2)

        ax.axhline(P_RH_even_avg,    c='crimson',    ls='--')
        ax.axhline(P_RH_odd_avg,     c='mediumblue', ls='--')
        ax.axhline(P_RH_coll_avg,    c='orange',     ls='--')
#        ax.axhline(P_RH_even_avg_LW, c='crimson',    ls='--', alpha=0.5, lw=2)
#        ax.axhline(P_RH_odd_avg_LW,  c='mediumblue', ls='--', alpha=0.5, lw=2)

        ax.set_xlabel(r"$t v_{Ti}/a$")
        ax.set_ylabel(r"$P_\mathrm{RH}$")
#        ax.set_xlim([0, time[-1]])
        if np.abs(P_RH_even_avg)>0:
            ax.set_yscale("symlog", linthresh=1e-1*np.abs(P_RH_even_avg))
        ax.grid(True)
        ax.legend()
        ax.set_ylim(ax.get_ylim())
        if time_val_avg is None:
            ax.fill_betweenx(ax.get_ylim(), time[-1]-time_avg, time[-1], color='0.5', alpha=0.15)
        else:
            ax.fill_betweenx(ax.get_ylim(), time_val_avg-time_avg/2, time_val_avg+time_avg/2, color='0.5', alpha=0.15)

        #####################
        # Plot in (x,y)
        #####################
        nx_padded = 256
        ny_padded = 256
        xold = np.copy(x)
        zed_val = 0

        # Plot potential
        ax = axs[0,2]
        ax.set_title(r"$\tilde v_{Ex} (\theta = %.2f)$" % (zed_val))
        _, _, im, vmin, vmax = diagObj.plot_quantity_x_y(quantity="phi", time_idx=-1, ky_order=1, nx=nx_padded, ny=ny_padded, fig=fig, ax=ax, zed_val=zed_val, symm=True, remove_zonal=True)
 

        # Plot parallel velocity
        ax = axs[1,2]
        ax.set_title(r"$\tilde u_\parallel (\theta = %.2f)$" % (zed_val))
        _, _, im, vmin, vmax = diagObj.plot_quantity_x_y(quantity="upar", time_idx=-1, nx=nx_padded, ny=ny_padded, fig=fig, ax=ax, zed_val=zed_val, symm=True, remove_zonal=True)

        # Add to plots
        dxphizonal, x, y, _  = diagObj.get_quantity_x_y(quantity="phi", time_idx=-1, only_zonal=True, kx_order=1, nx=nx_padded, zed_val=zed_val)
        vEzonal = -dxphizonal
        uparzonal, x, y, _  = diagObj.get_quantity_x_y(quantity="upar", time_idx=-1, only_zonal=True, kx_order=0, nx=nx_padded, zed_val=zed_val)
        tempzonal, x, y, _  = diagObj.get_quantity_x_y(quantity="temperature", time_idx=-1, only_zonal=True, kx_order=0, nx=nx_padded, zed_val=zed_val)
        vE_norm   = 0.5*y[-1]/np.abs(vEzonal).max()
        upar_norm = 0.5*y[-1]/np.abs(uparzonal).max()
        temp_norm = 0.5*y[-1]/np.abs(tempzonal).max()

        Pi_parallel_x_y, x, _, _  = diagObj.get_quantity_x_y(quantity="par_mom_transport", only_zonal=True, time_idx=-1, zed_val=zed_val, nx=nx_padded)
        Pi_parallel_x = Pi_parallel_x_y[:,0]

        norm_dE_Pi_parallel = 0.5*y[-1] / np.abs(dE_Pi_parallel_x).max()
        norm_Pi_parallel    = 0.5*y[-1] / np.abs(   Pi_parallel_x).max()

        for ax in [axs[0,2], axs[1,2]]:
            ax.plot(x, vEzonal[:,0]*vE_norm, c='forestgreen', label=r"$v_E$")
            ax.plot(x, uparzonal[:,0]*upar_norm, c='c', label=r"$u_\parallel$")
            ax.plot(x, tempzonal[:,0]*temp_norm, c='crimson', label=r"$T_\mathrm{tot}$")
#            tprim = diagObj.ncdata.variables['tprim'][0]
#            ax.plot(x, (tempzonal[:,0]-tprim*x)*temp_norm, c='crimson', label=r"$T_\mathrm{tot}$")
            ax.set_aspect('equal')
            ax.set_xlim([x[0], x[-1]])
        #axs[0,2].plot(xold, dE_Pi_parallel_x *norm_dE_Pi_parallel, c='brown', label=r"$- \langle u_\parallel^Z \partial_x\Pi_\parallel \rangle_{t, z} $", ls='--', lw=2)
        #axs[0,2].plot(x, Pi_parallel_x*norm_Pi_parallel, c='orange', label=r"$\Pi_\parallel$", lw=2)
        #axs[0,2].legend(fontsize=12)
        axs[0,2].legend()

        #####################
        # Plot in (x, zed)
        #####################

        # Plot zonal parallel flow
        ax = axs[0,3]

        ax.set_title(r"$\partial_x u_\parallel^Z$")
        diagObj.plot_quantity_x_zed(quantity="upar", fig=fig, ax=ax, only_zonal=True, kx_order=1, nx=nx_padded)

        # Plot parallel momentum transport
        ax = axs[0,4]

        ax.set_title(r"$\langle \Pi_\parallel \rangle_y$")
        diagObj.plot_quantity_x_zed(quantity="par_mom_transport", fig=fig, ax=ax, only_zonal=True, nx=nx_padded)

        # Plot perpendicular momentum transport
        ax = axs[0,5]

        ax.set_title(r"$\langle \Pi_\perp \rangle_y$")
        diagObj.plot_quantity_x_zed(quantity="Reynolds", fig=fig, ax=ax, only_zonal=True, nx=nx_padded)

        # Plot perpendicular momentum transport from |grad x|^2
        ax = axs[0,6]

        ax.set_title(r"$\langle \Pi_{\perp, |\nabla x|^2} \rangle_y$")
        diagObj.plot_quantity_x_zed(quantity="Reynolds_nablax2", fig=fig, ax=ax, only_zonal=True, nx=nx_padded)

        # Plot perpendicular momentum transport from grad x * grad y
        ax = axs[1,6]

        ax.set_title(r"$\langle \Pi_{\perp, \nabla x \cdot \nabla y} \rangle_y$")
        diagObj.plot_quantity_x_zed(quantity="Reynolds_nablaxy", fig=fig, ax=ax, only_zonal=True, nx=nx_padded)

        # Plot potential squared
        ax = axs[1,3]

        if avg_norm == 2:
            ax.set_title(r"$\langle \tilde v_{Ex}^2 \rangle_y$")
        else:
            ax.set_title(r"$\tilde v_{Ex} (y=0)$")
        diagObj.plot_quantity_x_zed(quantity="phi", fig=fig, ax=ax, avg_norm=avg_norm, remove_zonal=True, nx=nx_padded, ky_order=1)

        # Plot temperature fluctuations squared
        ax = axs[1,4]

        if avg_norm == 2:
            ax.set_title(r"$\langle \tilde T^2 \rangle_y$")
        else:
            ax.set_title(r"$\tilde T (y=0)$")
        diagObj.plot_quantity_x_zed(quantity="temperature", fig=fig, ax=ax, avg_norm=avg_norm, remove_zonal=True, nx=nx_padded)

        # Plot NZ parallel velocity squared
        ax = axs[1,5]

        if avg_norm == 2:
            ax.set_title(r"$\langle \tilde u_\parallel^2 \rangle_y$")
        else:
            ax.set_title(r"$\tilde u_\parallel (y=0)$")
        diagObj.plot_quantity_x_zed(quantity="upar", fig=fig, ax=ax, avg_norm=avg_norm, remove_zonal=True, nx=nx_padded)

        #####################
        # Finish plot
        #####################
        fig.suptitle(None)
        # Save plot
        plt.tight_layout()
        fig.savefig(dirname+"/fig_gammalin_gammaE_"+dirname.replace("/","_")+".pdf")
        plt.close()

        # Save data
        data_dict = {
            "tprim": tprim,
            "qinp": qinp,
            "eps": eps, 
            "gamma_lin_max":    gamma_max,
            "gammaE_avg":       gammaE_avg,
            "gammaE_std":       gammaE_std,
            "gammaE_LW_avg":    gammaE_LW_avg,
            "gammaE_LW_std":    gammaE_LW_std,
            "vE_avg":           vE_avg,
            "vE_RH_avg":        vE_RH_avg,
            "upar_avg":         upar_avg,
            "uparcos_avg":      uparcos_avg,
            "dxT_avg":          dxT_avg,
            "gammaE_RH_avg":    gammaE_RH_avg,
            "gammaE_RH_std":    gammaE_RH_std,
            "gammaE_RH_LW_avg": gammaE_RH_LW_avg,
            "gammaE_RH_LW_std": gammaE_RH_LW_std,
            "qflx_avg":         qflx_avg,
            "qflx_std":         qflx_std,
            "P_RH_even_avg":    P_RH_even_avg,
            "P_RH_odd_avg":     P_RH_odd_avg,
            "P_RH_coll_avg":    P_RH_coll_avg,
            "P_RH_even_avg_LW": P_RH_even_avg_LW,
            "P_RH_odd_avg_LW":  P_RH_odd_avg_LW,
            "P_RH_coll_avg_LW": P_RH_coll_avg_LW,
            "P_RH_even_std":    P_RH_even_std,
            "P_RH_odd_std":     P_RH_odd_std,
            "P_RH_coll_std":    P_RH_coll_std,
            "P_RH_even_std_LW": P_RH_even_std_LW,
            "P_RH_odd_std_LW":  P_RH_odd_std_LW,
            "P_RH_coll_std_LW": P_RH_coll_std_LW,
            "P_RH_even_avg_alt":       P_RH_even_avg_alt,
            "P_RH_odd_avg_alt":        P_RH_odd_avg_alt,
            "P_RH_even_avg_alt_vEpos": P_RH_even_avg_alt_vEpos,
            "P_RH_odd_avg_alt_vEpos":  P_RH_odd_avg_alt_vEpos,
            "P_RH_even_avg_alt_vEneg": P_RH_even_avg_alt_vEneg,
            "P_RH_odd_avg_alt_vEneg":  P_RH_odd_avg_alt_vEneg,
            "P_RH_even_passing_avg_alt":       P_RH_even_passing_avg_alt,
            "P_RH_odd_passing_avg_alt":        P_RH_odd_passing_avg_alt,
            "P_RH_even_passing_avg_alt_vEpos": P_RH_even_passing_avg_alt_vEpos,
            "P_RH_odd_passing_avg_alt_vEpos":  P_RH_odd_passing_avg_alt_vEpos,
            "P_RH_even_passing_avg_alt_vEneg": P_RH_even_passing_avg_alt_vEneg,
            "P_RH_odd_passing_avg_alt_vEneg":  P_RH_odd_passing_avg_alt_vEneg,
            "P_RH_even_trapped_avg_alt":       P_RH_even_trapped_avg_alt,
            "P_RH_odd_trapped_avg_alt":        P_RH_odd_trapped_avg_alt,
            "P_RH_even_trapped_avg_alt_vEpos": P_RH_even_trapped_avg_alt_vEpos,
            "P_RH_odd_trapped_avg_alt_vEpos":  P_RH_odd_trapped_avg_alt_vEpos,
            "P_RH_even_trapped_avg_alt_vEneg": P_RH_even_trapped_avg_alt_vEneg,
            "P_RH_odd_trapped_avg_alt_vEneg":  P_RH_odd_trapped_avg_alt_vEneg,
            "dyphi2_avg":        dyphi2_avg,
            "dyphi2_avg_vEpos":  dyphi2_avg_vEpos,
            "dyphi2_avg_vEneg":  dyphi2_avg_vEneg,
            "dE_Pi_parallel":        dE_Pi_parallel_avg,
            "dE_Pi_parallel_vEpos":  dE_Pi_parallel_avg_vEpos,
            "dE_Pi_parallel_vEneg":  dE_Pi_parallel_avg_vEneg,
            "dE_Pi_perp":            dE_Pi_perp_avg,
            "dE_Pi_perp_vEpos":      dE_Pi_perp_avg_vEpos,
            "dE_Pi_perp_vEneg":      dE_Pi_perp_avg_vEneg
            }

        print(data_dict)

        with open(dirname+'data_Dimits.json', 'w') as f:
            json.dump(data_dict, f)

    except Exception as e:
        traceback.print_exc()
        print(e)
        print("Could not process " + dirname)

