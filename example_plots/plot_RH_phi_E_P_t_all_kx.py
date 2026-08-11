import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})

import stellaDiagnostics as sD

# Load data

fphi  = 1
fapar = 0
fbpar = 0
fcoll = 1

#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0_higher-vel-res/run_tprim-4.9000"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_long_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_no-upwind_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_upwind-0.002_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_upwind-0.005_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-5.6000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-8.4000"
#fcoll = 0

#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-4/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-5/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-5/run_no-upwind_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.2000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-5.6000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-5.9500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-3.8500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-6.3000"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-8.4000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.2000_HR"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.2000_restart_linear"

#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-8.4000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-6.3000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-3.8500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-5.9500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-5.6000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-5.6000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-4.2000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-4.9000"

#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.01/run_tprim-4.9000"

kx_max = 1e4 #0.8

time_min = 500
time_min = 2.2241E+03
#time_min = 1000
time_max = 1e6
time_idx_skip = 1

ylim_P_RH = [-1e-2, 1e-2]
linthresh  = 1e-4

#D_hyper = 0.05
D_hyper = None

#passing_trapped = "passing"
#passing_trapped = "trapped"
passing_trapped = "both"

dirname_string = dirname.replace("/","_")

fig_dir_I_phi_RH = "figs_RH_I_phi_" + dirname_string
fig_dir_E_RH     = "figs_RH_E_"     + dirname_string
fig_dir_P_RH     = "figs_RH_P_"     + dirname_string

if passing_trapped == "passing":
    fig_add = "_passing"
elif passing_trapped == "trapped":
    fig_add = "_trapped"
else:
    fig_add = ""

fig_dir_P_RH += fig_add

import os
for fig_dir in [fig_dir_I_phi_RH, fig_dir_E_RH, fig_dir_P_RH]:
    os.system("mkdir -p " + fig_dir)

filename  = "/CBC"
StellaObj = sD.stellaDiagnostics(dirname+filename)

kx_all = StellaObj.ncdata['kx'][:]

i_fig = 0

colors = ['k' for i in range(len(kx_all))]

str_fig = ""
if np.abs(fphi  - 1) > 1e-10:
    str_fig += "_fphi-%.2f" % (fphi)
#if np.abs(fapar - 1) > 1e-10:
#    str_fig += "_fapar-%.2f" % (fapar)
#if np.abs(fbpar - 1) > 1e-10:
#    str_fig += "_fbpar-%.2f" % (fbpar)

#plot_all_kx_together = True
plot_all_kx_together = False

colors = None

##################################
### LOADING DATA FILES ###########
##################################

data_filenames = ["data_ERH_mean_kx", 
                  "data_P_RH_num_mean_kx", 
                  "data_P_RH_even_mean_kx", 
                  "data_P_RH_odd_mean_kx", 
                  "data_P_RH_phi_even_mean_kx", 
                  "data_P_RH_phi_odd_mean_kx", 
                  "data_P_RH_coll_even_mean_kx", 
                  "data_P_RH_coll_odd_mean_kx"]

data_files_exist = True
for data_filename in data_filenames:
    if not exists(dirname+"/"+data_filename+".dat"):
        data_files_exist = False
        break

#load_from_file = True
load_from_file = False
if load_from_file and data_files_exist:

    E_RH_mean_kx           = np.loadtxt(dirname+"/"+data_filenames[0]+".dat")
    P_RH_num_mean_kx       = np.loadtxt(dirname+"/"+data_filenames[1]+".dat")
    P_RH_even_mean_kx      = np.loadtxt(dirname+"/"+data_filenames[2]+".dat")
    P_RH_odd_mean_kx       = np.loadtxt(dirname+"/"+data_filenames[3]+".dat")
    P_RH_phi_even_mean_kx  = np.loadtxt(dirname+"/"+data_filenames[4]+".dat")
    P_RH_phi_odd_mean_kx   = np.loadtxt(dirname+"/"+data_filenames[5]+".dat")
    P_RH_coll_even_mean_kx = np.loadtxt(dirname+"/"+data_filenames[6]+".dat")
    P_RH_coll_odd_mean_kx  = np.loadtxt(dirname+"/"+data_filenames[7]+".dat")

else:
    # Evaluate data first
    E_RH_mean_kx           = np.zeros_like(kx_all)
    P_RH_num_mean_kx       = np.zeros_like(kx_all)
    P_RH_even_mean_kx      = np.zeros_like(kx_all)
    P_RH_odd_mean_kx       = np.zeros_like(kx_all)
    P_RH_phi_even_mean_kx  = np.zeros_like(kx_all)
    P_RH_phi_odd_mean_kx   = np.zeros_like(kx_all)
    P_RH_coll_even_mean_kx = np.zeros_like(kx_all)
    P_RH_coll_odd_mean_kx  = np.zeros_like(kx_all)
    
    if D_hyper is not None:
        P_RH_hyper_mean_kx = np.zeros_like(kx_all)
    
    for i_kx in range(len(kx_all)):
        print("Evaluating kx %i/%i..." % (i_kx+1, len(kx_all)), end="\r")
    
        if kx_all[i_kx] <= 0 or np.abs(kx_all[i_kx]) > kx_max:
            continue
    
        if plot_all_kx_together:
            idxs_kx = None
        else:
            idxs_kx = np.array([i_kx])
    
        fig, axs, RH_phi_I, time, kx = \
            StellaObj.plot_RH_phi_I(time_min=time_min, time_max=time_max, idxs_kx=idxs_kx, colors=colors)
        fig.suptitle(r"$ k_x \rho_i = %.4f$" % (kx_all[i_kx]))
        plt.tight_layout()
    
        axs[2].set_xlim(xmin=time_min)
        if time_max < 1e5:
            axs[2].set_xlim(xmax=time_max)
        axs[2].set_ylim(ymin=0)
    
        fig.savefig(fig_dir_I_phi_RH+"/fig_RH_phi_I_kx_t" + str_fig +"_%i.pdf" % (i_fig))
    
        plt.close()
    
        # E_RH plot
        fig, ax, E_RH_t_kx, t, kx = StellaObj.plot_E_RH(time_min=time_min, time_max=time_max, idxs_kx=idxs_kx, colors=colors)
    
        P_RH_t_kx_num = np.gradient(E_RH_t_kx, t, axis=0)
        
        log = False
        #log = True
        if log:
            ax.set_yscale('log')
        else:
            ax.set_ylim(ymin=0)
        
            #axs[0].set_ylim(ymin=1e-2, ymax=100)
            #axs[1].set_ylim(ymin=1e-4, ymax=10)
        
        fig.suptitle(r"$ k_x \rho_i = %.4f$" % (kx_all[i_kx]))
        plt.tight_layout()
        fig.savefig(fig_dir_E_RH+"/fig_E_RH_kx_t" + str_fig +"_%i.pdf" % (i_fig))
        plt.close()
    
        # P_RH plot
        fig, axs, P_RH_even_t_kx,  P_RH_odd_t_kx,     \
                  P_RH_phi_even_t_kx,  P_RH_phi_odd_t_kx, \
                  P_RH_apar_even_t_kx, P_RH_apar_odd_t_kx,\
                  P_RH_bpar_even_t_kx, P_RH_bpar_odd_t_kx,\
                  P_RH_coll_even_t_kx, P_RH_coll_odd_t_kx,\
                  P_RH_hyper_t_kx, time, kx = \
            StellaObj.plot_P_RH(passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, idxs_kx=idxs_kx, colors=colors, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll, D_hyper=D_hyper)
    
        mean_P_RH_even = np.mean(P_RH_even_t_kx[:,0])
        mean_P_RH_odd  = np.mean(P_RH_odd_t_kx[ :,0])
        P_RH_phi_t_kx  = P_RH_phi_even_t_kx + P_RH_phi_odd_t_kx
        mean_P_RH_phi = np.mean(P_RH_phi_t_kx[ :,0])
        P_RH_coll_t_kx = P_RH_coll_even_t_kx + P_RH_coll_odd_t_kx
        mean_P_RH_coll = np.mean(P_RH_coll_t_kx[ :,0])
    
        mean_P_RH = mean_P_RH_even + mean_P_RH_odd
        if D_hyper is not None:
            mean_P_RH_hyper  = np.mean(P_RH_hyper_t_kx[ :,0])
            mean_P_RH += mean_P_RH_hyper
        mean_P_RH_num = np.mean(P_RH_t_kx_num[:,0])
    
        if fphi != 0:
            axs[2].axhline(mean_P_RH_phi, c='mediumblue', label=r"$P_{\mathrm{RH}, \varphi}$")
        if fcoll != 0:
            axs[2].axhline(mean_P_RH_coll, c='orange', label=r"$P_{\mathrm{RH}, C}$")
        axs[2].axhline(mean_P_RH,      c='k', label=r"Total")
        axs[2].axhline(mean_P_RH_num,  c='0.5', ls='--', label=r"Numerical")
    
        if ylim_P_RH is not None:
            axs[2].set_ylim(ylim_P_RH)
        
        axs[2].legend()
    
        axs[2].set_yscale('symlog', linthresh=linthresh)
    
        fig.suptitle(r"$ k_x \rho_i = %.4f$" % (kx_all[i_kx]))
        plt.tight_layout()
        fig.savefig(fig_dir_P_RH+"/fig_P_RH_kx_t" + str_fig +"_%i.pdf" % (i_fig))
        
        plt.close()
        
        if i_fig == 0:
            E_RH_t_sumkx       = np.sum(E_RH_t_kx, axis=1)
            P_RH_phi_even_t_sumkx = np.sum(P_RH_phi_even_t_kx, axis=1)
            P_RH_phi_odd_t_sumkx  = np.sum(P_RH_phi_odd_t_kx,  axis=1)
            P_RH_coll_t_sumkx     = np.sum(P_RH_coll_even_t_kx+P_RH_coll_odd_t_kx,  axis=1)
            if D_hyper is not None:
                P_RH_hyper_t_sumkx = np.sum(P_RH_hyper_t_kx,  axis=1)
    
        else:
            E_RH_t_sumkx       += np.sum(E_RH_t_kx, axis=1)
            P_RH_phi_even_t_sumkx += np.sum(P_RH_phi_even_t_kx, axis=1)
            P_RH_phi_odd_t_sumkx  += np.sum(P_RH_phi_odd_t_kx,  axis=1)
            P_RH_coll_t_sumkx     += np.sum(P_RH_coll_even_t_kx+P_RH_coll_odd_t_kx,  axis=1)
            if D_hyper is not None:
                P_RH_hyper_t_sumkx += np.sum(P_RH_hyper_t_kx,  axis=1)
    
        dt = np.gradient(t)
    
        E_RH_mean_kx[idxs_kx]           = np.mean(E_RH_t_kx          *dt[:,None], axis=0)/np.mean(dt)
        P_RH_num_mean_kx[idxs_kx]       = (E_RH_t_kx[-1]-E_RH_t_kx[0])/(t[-1]-t[0])
        P_RH_even_mean_kx[idxs_kx]      = np.mean(P_RH_even_t_kx     *dt[:,None], axis=0)/np.mean(dt)
        P_RH_odd_mean_kx[idxs_kx]       = np.mean(P_RH_odd_t_kx      *dt[:,None], axis=0)/np.mean(dt)
        P_RH_phi_even_mean_kx[idxs_kx]  = np.mean(P_RH_phi_even_t_kx *dt[:,None], axis=0)/np.mean(dt)
        P_RH_phi_odd_mean_kx[idxs_kx]   = np.mean(P_RH_phi_odd_t_kx  *dt[:,None], axis=0)/np.mean(dt)
        P_RH_coll_even_mean_kx[idxs_kx] = np.mean(P_RH_coll_even_t_kx*dt[:,None], axis=0)/np.mean(dt)
        P_RH_coll_odd_mean_kx[idxs_kx]  = np.mean(P_RH_coll_odd_t_kx *dt[:,None], axis=0)/np.mean(dt)
        if D_hyper is not None:
            P_RH_hyper_mean_kx[idxs_kx] = np.mean(P_RH_hyper_t_kx *dt[:,None], axis=0)/np.mean(dt)
    
    
        i_fig += 1
    
        if plot_all_kx_together:
            break

    # Save data
    np.savetxt(dirname+"/"+data_filenames[0]+".dat", E_RH_mean_kx           )
    np.savetxt(dirname+"/"+data_filenames[1]+".dat", P_RH_num_mean_kx       )
    np.savetxt(dirname+"/"+data_filenames[2]+".dat", P_RH_even_mean_kx      )
    np.savetxt(dirname+"/"+data_filenames[3]+".dat", P_RH_odd_mean_kx       )
    np.savetxt(dirname+"/"+data_filenames[4]+".dat", P_RH_phi_even_mean_kx  )
    np.savetxt(dirname+"/"+data_filenames[5]+".dat", P_RH_phi_odd_mean_kx   )
    np.savetxt(dirname+"/"+data_filenames[6]+".dat", P_RH_coll_even_mean_kx )
    np.savetxt(dirname+"/"+data_filenames[7]+".dat", P_RH_coll_odd_mean_kx  )

    # Plot sum over kx's
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, 16))
    
    axs[0].plot(t, E_RH_t_sumkx, c='k', alpha=0.5)
    axs[0].set_ylabel(r"$\sum_{k_x} E_\mathrm{RH}$")
    
    E_RH_sum_kx_mean = np.mean(E_RH_t_sumkx*dt)/np.mean(dt)
    axs[0].axhline(E_RH_sum_kx_mean, ls='--', c='k', alpha=0.5)
    
    P_RH_t_sumkx = P_RH_phi_even_t_sumkx + P_RH_phi_odd_t_sumkx + P_RH_coll_t_sumkx
    axs[1].plot(t, P_RH_phi_even_t_sumkx,  c='crimson')
    axs[1].plot(t, P_RH_phi_odd_t_sumkx,   c='mediumblue')
    axs[1].plot(t, P_RH_coll_t_sumkx,      c='orange')
    if D_hyper is not None:
        axs[1].plot(t, P_RH_hyper_t_sumkx, c='c')
        P_RH_t_sumkx += P_RH_hyper_t_sumkx
    axs[1].plot(t, P_RH_t_sumkx, c='k', alpha=0.5)
    axs[1].set_ylabel(r"$\sum_{k_x} P_\mathrm{RH}$")
    
    P_RH_phi_even_sum_kx_mean = np.mean(P_RH_phi_even_t_sumkx*dt)/np.mean(dt)
    P_RH_phi_odd_sum_kx_mean  = np.mean(P_RH_phi_odd_t_sumkx*dt )/np.mean(dt)
    P_RH_coll_sum_kx_mean     = np.mean(P_RH_coll_t_sumkx*dt    )/np.mean(dt)
    P_RH_sum_kx_mean          = P_RH_phi_even_sum_kx_mean + P_RH_phi_odd_sum_kx_mean + P_RH_coll_sum_kx_mean
    if D_hyper is not None:
        P_RH_hyper_sum_kx_mean  = np.mean(P_RH_hyper_t_sumkx*dt )/np.mean(dt)
        P_RH_sum_kx_mean += P_RH_hyper_sum_kx_mean
    
    P_RH_sum_kx_num = np.gradient(E_RH_t_sumkx, t)
    
    def moving_average(f_t, dt_val=20):
        f_avg_t = np.zeros_like(f_t)
        for i_t, t_val in enumerate(t):
            idx_min =     np.argmin(np.abs(t - (t_val-dt_val/2)))
            idx_max = min(np.argmin(np.abs(t - (t_val+dt_val/2))) + 1, len(t))
            f_avg_t[i_t] = np.sum( f_t[idx_min:idx_max]*dt[idx_min:idx_max] ) / np.sum(dt[idx_min:idx_max])
    
        return f_avg_t
    
    try:
        axs[1].plot(t, moving_average(P_RH_sum_kx_num), c='c', ls='--', alpha=0.5)
    except Exception as e:
        print(e)
    
    axs[1].plot(t, P_RH_sum_kx_num, c='c', ls='--', alpha=0.1)
    
    P_RH_sum_kx_num_mean = np.mean(P_RH_sum_kx_num*dt)/np.mean(dt)
    
    axs[1].axhline(P_RH_phi_even_sum_kx_mean, c='crimson', label=r"$\varphi$ Even")
    axs[1].axhline(P_RH_phi_odd_sum_kx_mean,  c='mediumblue', label=r"$\varphi$ Odd")
    axs[1].axhline(P_RH_coll_sum_kx_mean,     c='orange', label=r"Coll.")
    if D_hyper is not None:
        axs[1].axhline(P_RH_hyper_sum_kx_mean,  c='forestgreen', label=r"Hyper")
    axs[1].axhline(P_RH_sum_kx_mean    ,  c='k', label=r"Total", alpha=0.5)
    axs[1].axhline(P_RH_sum_kx_num_mean,  c='c', label=r"Num", alpha=0.5)
    
    axs[1].legend()
    
    axs[1].set_xlabel(r"$ t v_{T}/a$")
    for ax in axs:
        ax.grid(True)
    
    if ylim_P_RH is not None:
        axs[1].set_ylim(ylim_P_RH)
    
    axs[0].set_ylim(ymin=0)
    
    axs[1].set_yscale('symlog', linthresh=linthresh)
    
    plt.tight_layout()
    
    fig.savefig("fig_E_RH_P_RH_total" + dirname_string + "_" + str_fig + fig_add + ".pdf")
    plt.close()

##################################
### PLOTTING SUMMARY PLOTS #######
##################################

# Plot mean as a function of kx
P_RH_mean_kx      = P_RH_even_mean_kx      + P_RH_odd_mean_kx
P_RH_phi_mean_kx  = P_RH_phi_even_mean_kx  + P_RH_phi_odd_mean_kx
P_RH_coll_mean_kx = P_RH_coll_even_mean_kx + P_RH_coll_odd_mean_kx
if D_hyper is not None:
    P_RH_mean_kx += P_RH_hyper_mean_kx

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7,18))

axs[0].loglog(kx_all[kx_all>0], E_RH_mean_kx[kx_all>0], marker='.', c='k', alpha=0.5)
axs[0].set_ylabel(r"$\langle E_\mathrm{RH} \rangle_t$")

kx_plot = kx_all[kx_all>0]
axs[0].loglog(kx_plot, 5e-4 * kx_plot**(-5/2), ls='--', c='0.5', lw=2)
axs[0].loglog(kx_plot, 5e-4 * kx_plot**(-3),   ls='--', c='0.5', lw=2)
axs[0].loglog(kx_plot, 5e-4 * kx_plot**(-7/2), ls='--', c='0.5', lw=2)

lw=2
plot_avg = False
for i_norm, norm in enumerate([1, 1/E_RH_mean_kx]):

    axs[1+i_norm].semilogx(kx_all, norm*P_RH_phi_even_mean_kx, lw=lw, label=r"$P_{\mathrm{RH}}^{\mathrm{NL},+}$", marker='.', c='crimson')
    axs[1+i_norm].semilogx(kx_all, norm*P_RH_phi_odd_mean_kx,  lw=lw, label=r"$P_{\mathrm{RH}}^{\mathrm{NL},-}$", marker='.', c='mediumblue')
    if fcoll != 0:
        axs[1+i_norm].semilogx(kx_all, norm*P_RH_coll_mean_kx, lw=lw, label=r"$P_{\mathrm{RH}}^{C}$", marker='.', c='orange')
    if D_hyper is not None:
        axs[1+i_norm].semilogx(kx_all, norm*P_RH_hyper_mean_kx,  marker='.', c='0.5', label=r"Hyper")
    axs[1+i_norm].semilogx(kx_all, norm*P_RH_mean_kx ,         lw=lw, label=r"$P_{\mathrm{RH}}^\mathrm{NL}$", marker='.', c='k')#, alpha=0.5)
    axs[1+i_norm].semilogx(kx_all[kx_all>0], (norm*P_RH_num_mean_kx)[kx_all>0],     lw=lw, label=r"$\mathrm{d}E_\mathrm{RH}/\mathrm{d}t$", marker='.', c='0.5')#, alpha=0.5)

    
    if plot_avg:
        axs[1+i_norm].axhline(np.mean(norm*P_RH_phi_even_mean_kx),  c='crimson')
        axs[1+i_norm].axhline(np.mean(norm*P_RH_phi_odd_mean_kx),   c='mediumblue')

        if fcoll != 0:
            axs[1+i_norm].axhline(np.mean(norm*P_RH_coll_mean_kx),      c='orange')
        
        if D_hyper is not None:
            axs[1+i_norm].axhline(np.mean(norm*P_RH_hyper_mean_kx),  c='c')
        
        axs[1+i_norm].axhline(np.mean(norm*P_RH_mean_kx),      c='k', alpha=0.5)

    axs[1+i_norm].set_xlabel(r"$k_x \rho_i$")

#if fcoll != 0:
#    axs[2].semilogx(kx_plot[kx_plot<kmax], ((norm*P_RH_coll_mean_kx)[(kx_all>0) & (kx_all < kmax)])[0]*(kx_plot[kx_plot<kmax]/kx_plot[0])**2, c='0.5', ls='--', lw=2)

#axs[1].legend(fontsize=12)

axs[1].set_ylabel(r"$\langle P_\mathrm{RH}\rangle_t$")
axs[2].set_ylabel(r"$\langle P_\mathrm{RH}\rangle_t/\langle E_\mathrm{RH}\rangle_t$")
axs[2].set_yscale('symlog', linthresh=1e-3)

for ax in axs:
    ax.set_xlim(xmin=0.5*(kx_all[1]-kx_all[0]))
    ax.grid(True)

plt.tight_layout()
fig.savefig("fig_E_RH_P_RH_mean_kx_" + dirname_string + "_" + str_fig + fig_add + ".pdf")

