import numpy as np
import os
import scipy.special as specialfunc
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})

import stellaDiagnostics as sD

Q_div = 10

skip_phi2 = False
#skip_phi2 = True

plot_ratio = False
#plot_ratio = True

# Setup
code  = "stella"
filename = "CBC"
colors = ["k", "orange", "crimson", "forestgreen", "mediumblue", "purple", "c", "pink", "0.5", "yellow"]

## COLLISION SCAN
dir_cless = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
dir_cal   = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/"
dirnames = [dir_cless+"run_tprim-4.9000", dir_cless+"run_tprim-6.3000", dir_cal+"run_tprim-4.9000", dir_cal+"run_tprim-6.3000"]
labels = [r"CBC ($\nu=0$) $R/L_T=4.9$", r"$R/L_T = 6.3$", r"CBC ($\nu=10^{-3}$) $R/L_T=4.9$", r"$R/L_T = 6.3$"]
ylim = [3e-3, 3e2]
figname_add = "_nu-var"

## UPWIND SCAN
dir_0 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
dir_1 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-5/"
dirnames = [dir_0 + "run_tprim-4.9000", dir_0 + "run_upwind-0.005_tprim-4.9000", dir_0 + "run_upwind-0.002_tprim-4.9000", dir_0 + "run_no-upwind_tprim-4.9000", dir_0 + "run_drifts-implicit-T_tprim-4.9000", dir_0 + "run_upwind-0.002_drifts-implicit-T_tprim-4.9000", dir_0 + "run_upwind-0.005_drifts-implicit-T_tprim-4.9000", dir_0 + "run_no-upwind_drifts-implicit-T_mirror-SL-True_tprim-4.900", dir_1 + "run_tprim-4.9000", dir_1 + "run_no-upwind_tprim-4.9000"]
labels = [r"Expl dr., $u=0.02$, msl T", r"Expl dr., $u=0.005$, msl F", r"Expl dr., $u=0.002$, msl F", r"Expl dr., $u=0$, msl F", r"Impl dr., $u=0.02$, msl T", r"Impl dr., $u=0.002$, msl F", r"Impl dr., $u=0.005$, msl F",  r"Impl dr., $u=0$, msl T", r"$\nu=10^{-5}$", r"$\nu = 10^{-5}$, no upwind"]
#labels = [r"upwind = $0.02$, mirror semi-Lagrange True", r"upwind = $0.002$, mirror semi-Lagrange True", r"upwind = $0$, mirror semi-Lagrange False", r"Implicit drifts", r"Implicit drifts upwind $=0.002$", r"Implicit drifts upwind $=0.005$", r"Implicit drifts upwind = $0$, mirror semi-Lagrange True", r"$\nu=10^{-5}$", r"$\nu = 10^{-5}$, no upwind"]
ylim = [1e-4, 1e2]
figname_add = "_upwind-scan"

## COLLISION SCAN 2
dir_0 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0_higher-vel-res/"
dir_1 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
dir_2 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/"
dir_3 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-4/"
dir_4 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/"
dir_5 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-5/"
#dir_6 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-6/"

dirnames = [dir_0+"run_tprim-4.9000", dir_1+"run_tprim-4.9000", dir_1+"run_long_tprim-4.9000", dir_1+"run_long_tprim-4.9000_continue", dir_2+"run_tprim-4.9000", dir_3+"run_tprim-4.9000", dir_4+"run_tprim-4.9000"]#, dir_5+"run_tprim-4.9000"]
labels = [r"CBC ($\nu=0$ HR) $R/L_T=4.9$", r"CBC ($\nu=0$) $R/L_T=4.9$", None, None, r"$(\nu=10^{-3})$", r"$(\nu=3\cdot 10^{-4})$", r"$(\nu=10^{-4})$"]#, r"$(\nu=10^{-5})$"]
colors = ["k", "crimson", "crimson", "crimson", "forestgreen", "mediumblue", "purple", "c", "pink"]
figname_add = "_nu-var"
ylim = [1e-3, 1e2]

#dirnames = [dir_0+"run_tprim-4.9000", dir_1+"run_tprim-4.9000", dir_1+"run_long_tprim-4.9000", dir_5+"run_tprim-4.9000", dir_4+"run_tprim-4.9000", dir_3+"run_tprim-4.9000"]
#labels = [r"CBC ($\nu=0$ HR) $R/L_T=4.9$", r"CBC ($\nu=0$) $R/L_T=4.9$", None, r"$(\nu=10^{-5})$", r"$(\nu=10^{-4})$", r"$(\nu=3\cdot 10^{-4})$"]
#figname_add = "_nu-var2"
#ylim = [1e-3, 1e2]
#colors = ["k", "orange", "orange", "crimson", "forestgreen", "mediumblue", "purple", "c", "pink"]

#dirnames = [dir_0+"run_tprim-4.9000", dir_1+"run_no-upwind_tprim-4.9000", dir_1+"run_tprim-4.9000", dir_1+"run_long_tprim-4.9000", dir_5+"run_tprim-4.9000", dir_4+"run_tprim-4.9000", dir_3+"run_tprim-4.9000"]
#labels = [r"CBC ($\nu=0$ HR) $R/L_T=4.9$", r"CBC ($\nu=0$ no upwind) $R/L_T=4.9$", r"CBC ($\nu=0$) $R/L_T=4.9$", None, r"$(\nu=10^{-5})$", r"$(\nu=10^{-4})$", r"$(\nu=3\cdot 10^{-4})$"]
#dirnames = [dir_0+"run_tprim-4.9000", dir_1+"run_no-upwind_tprim-4.9000", dir_1+"run_tprim-4.9000", dir_1+"run_long_tprim-4.9000", dir_5+"run_tprim-4.9000", dir_4+"run_tprim-4.9000", dir_3+"run_tprim-4.9000"]
#labels = [r"CBC ($\nu=0$ HR) $R/L_T=4.9$", r"CBC ($\nu=0$ no upwind) $R/L_T=4.9$", r"CBC ($\nu=0$) $R/L_T=4.9$", None, r"$(\nu=10^{-5})$", r"$(\nu=10^{-4})$", r"$(\nu=3\cdot 10^{-4})$"]
#figname_add = "_nu-var3"

#ylim = [1e-3, 1e2]

#### q SCAN
#dir_q1 = "2026-06-26_scan_qinp-0.7_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
#dir_q2 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
#dir_q3 = "2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
#dirnames = [dir_q1+"run_tprim-7.5000", dir_q2+"run_tprim-4.9000", dir_q3+"run_tprim-4.9000"]
#labels = [r"CBC ($q=0.7$) $R/L_T=7.5$", r"CBC ($q=1.4$) $R/L_T=4.9$", r"CBC ($q=2.8$) $R/L_T=4.9$"]
#ylim = [3e-3, 3e2]
#figname_add = "_q-var"

#### eps SCAN
#dir_1 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0/"
#dir_2 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
#dir_3 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0/"
#ylim = [3e-2, 3e2]
#dirnames = [dir_1+"run_tprim-4.9000", dir_2+"run_tprim-4.9000", dir_3+"run_tprim-4.9000"]
#labels = [r"CBC ($\epsilon=0.045$) $R/L_T=4.9$", r"CBC ($\epsilon=0.18$) $R/L_T=4.9$", r"CBC ($\epsilon=0.36$) $R/L_T=4.9$"]
#figname_add = "_eps-var"
#dirnames = [dir_1+"run_tprim-6.3000", dir_2+"run_tprim-6.3000", dir_3+"run_tprim-6.3000"]
#labels = [r"CBC ($\epsilon=0.045$) $R/L_T=6.3$", r"CBC ($\epsilon=0.18$) $R/L_T=6.3$", r"CBC ($\epsilon=0.36$) $R/L_T=6.3$"]
#figname_add = "_eps-var2"

# Plot for Qflx(t)
fig_qflx_time, ax_qflx_time = plt.subplots(1,1,figsize=(12,9))
plt.figtext(0.5,0.99,os.path.abspath('.'),ha='center',va='top',fontsize=6, usetex=False, figure=fig_qflx_time)

for i_dir, dirname in enumerate(dirnames):
    
    try:
        ncol=1
        diagObj  = sD.stellaDiagnostics(dirname +"/"+ filename, code=code)
        
        _, vflx, qflx, time = diagObj.get_fluxes_over_time(norm=False)
        if labels[i_dir] is not None:
            label = labels[i_dir]
        else:
            label = None
        ax_qflx_time.plot(time, qflx/Q_div, label=label, c=colors[i_dir], lw=2)

        if not skip_phi2:
            ax_phi2_time = ax_qflx_time
            dl_over_B_avg = diagObj.dl_over_B_avg()
            time_all = diagObj.get_time_array()

#            phi2_Z_vs_t_zed, time, _  = diagObj.read_phi2_vs_t_zed(only_zonal=True)
#            phi2_Z_vs_t = np.sum(phi2_Z_vs_t_zed*dl_over_B_avg[None,:], axis=1)
#            if labels[i_dir] is not None:
#                label = labels[i_dir] +r" (zonal, x-der)"
#            else:
#                label = None
#            ax_phi2_time.plot(time, phi2_Z_vs_t, label=label, c=colors[i_dir], ls='-.')
#            phi2_NZ_vs_t_zed, time, _ = diagObj.read_phi2_vs_t_zed(remove_zonal=True)
#            phi2_NZ_vs_t = np.sum(phi2_NZ_vs_t_zed*dl_over_B_avg[None,:], axis=1)
#            ax_phi2_time.plot(time, phi2_NZ_vs_t, label=r"(nonzonal)", c=colors[i_dir], lw=3, ls='--')

            phiZ     = diagObj.ncdata.variables['phi_vs_t'][:,0,:,:,0,:] # t-zed-kx-r
            phiZ_C   = phiZ[:,:,:,0] + 1j*phiZ[:,:,:,1]

            zed      = diagObj.ncdata.variables['zed'][:]
            kx       = diagObj.ncdata.variables['kx'][:]
            # Evaluate 1-Gamma0 (single species!)
            Gamma0_vals = np.zeros( (len(zed), len(kx)) )
            shat   = diagObj.ncdata.variables['shat'].getValue()
            gds22  = diagObj.ncdata.variables['gds22'][:,0]/shat**2 # |nabla(x)|^2
            bmag   = diagObj.ncdata.variables['bmag'][:,0]
            for i_kx, kx_val in enumerate(kx):
                kperp2 = (kx_val/bmag)**2 * gds22
                Gamma0_vals[:,i_kx] = specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2)
            E_phi    = np.sum( (1-Gamma0_vals[None,:,:])*np.abs(phiZ_C)**2 * dl_over_B_avg[None,:,None], axis=(1,2)) 
            if labels[i_dir] is not None:
                label_phi = labels[i_dir] + r" $(E_{\varphi})$"
            else:
                label_phi = None

            uparZ    = diagObj.ncdata.variables['upar'][:,0,0,:,:,0,:] # t-zed-kx-r
            uparZ_C  = uparZ[:,:,:,0] + 1j*uparZ[:,:,:,1]
            E_upar   = np.sum(0.5*np.abs(uparZ_C)**2 * dl_over_B_avg[None,:,None], axis=(1,2))
            if labels[i_dir] is not None:
                label_upar = labels[i_dir] + r" $(E_{u_\parallel})$"
            else:
                label_upar = None

            if not plot_ratio:
                ax_phi2_time.plot(time_all, E_phi, label=label_phi, c=colors[i_dir], ls='-.')
                ax_phi2_time.plot(time_all, E_upar, label=label_upar, c=colors[i_dir], ls='--', alpha=0.5)
            else:
                if labels[i_dir] is not None:
                    label = labels[i_dir] + r" $( \epsilon^2/q^2 E_{u_\parallel}/E_\varphi)$"
                else:
                    label = None

                epsilon = 0.18
                qinp    = 1.4
                ax_phi2_time.plot(time_all, (epsilon/qinp)**2 * E_upar/E_phi, label=label, c=colors[i_dir], ls='-.')

#            tempZ    = diagObj.ncdata.variables['temperature'][:,0,0,:,:,0,:] # t-zed-kx-r
#            tempZ_C  = tempZ[:,:,:,0] + 1j*tempZ[:,:,:,1]
#            E_temp   = np.sum(0.5*np.abs(tempZ_C)**2 * dl_over_B_avg[None,:,None], axis=(1,2))
#            if labels[i_dir] is not None:
#                label = labels[i_dir] + r" $(E_T)$"
#            else:
#                label = None
#            ax_phi2_time.plot(time_all, E_temp, label=label, c=colors[i_dir], ls=':', alpha=0.5)
    
            ncol = min(len([label for label in labels if label is not None]), 6)

    except Exception as e:
        print(e)
        continue
    
ax_qflx_time.legend(fontsize=6,ncol=ncol)
ax_qflx_time.grid()
if Q_div != 1:
    ax_qflx_time.set_ylabel(r"$Q/ %.1f Q_\mathrm{gB}$" % (Q_div))
else:
    ax_qflx_time.set_ylabel(r"$Q/Q_\mathrm{gB}$")
ax_qflx_time.set_xlabel(r"$t v_T/a$")
ax_qflx_time.set_xlim(xmin=0)
ax_qflx_time.set_yscale('log')
ax_qflx_time.set_ylim(ylim)
fig_qflx_time.savefig("fig_qflx_over_time"+figname_add+".pdf")

#ax_qflx_time.set_xlim(xmax=1000)
#ax_qflx_time.set_ylim(ymin=1e-8)
#fig_qflx_time.savefig("fig_qflx_over_time_zoom.pdf")

ax_qflx_time.set_xlim(xmin=1e2)
ax_qflx_time.set_xscale('log')
fig_qflx_time.savefig("fig_qflx_over_time"+figname_add+"_loglog.pdf")

plt.close()
