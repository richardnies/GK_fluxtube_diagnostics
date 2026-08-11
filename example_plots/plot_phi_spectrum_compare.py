import numpy as np
import os
from os.path import exists
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})
fontsize_legend=5

#W_instead_of_phi = True
W_instead_of_phi = False

lw = 0.5
overplot_kx_ky = True

import loadStellaScan as lSS

plot_legend = False

aspect_ratio = 2.778

#plot_alpha_spectrum = True
plot_alpha_spectrum = False

#delta_t_avg=None
#delta_t_avg=20
delta_t_avg=500

#load_from_file = False
load_from_file = True

plot_slides = False
#plot_slides = True

add_inset = False
#add_inset = True

#add_arrows = True
add_arrows = False

if plot_slides:
    figsize = (7,4.5)
else:
    figsize = (4.5,4.5)

############## COLL SCANS ###############
dir_0 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0_higher-vel-res/"
dir_1 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
dir_2 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/"
dir_3 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-4/"
dir_4 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/"
dir_5 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-5/"

dir_q_0 = "2026-06-26_scan_qinp-0.7_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
dir_q_1 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
dir_q_2 = "2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"

for tprim in [3.85, 4.2, 4.9, 5.6, 5.95, 6.3, 7.35, 8.4]:
#for tprim in [7.5, 9.6]:
#for tprim in [5.6, 6.3]:
    
    if W_instead_of_phi:
        add_str = "_W_instead_of_phi"
    else:
        add_str = ""
    
    #### DIRECTORIES
    # nu scan
    #dirnames = [dir_1+"run_tprim-%.4f" % (tprim), dir_4+"run_tprim-%.4f" % (tprim), dir_2+"run_tprim-%.4f" % (tprim)]
    #labels = [r"CBC ($\nu=0$)", r"CBC ($\nu=10^{-4}$)", r"CBC ($\nu=10^{-3}$)"]
    #add_str += "_nu_scan_tprim-%.4f" % (tprim)
    
    # qinp scan
    #dirnames = [dir_q_0+"run_tprim-%.4f" % (tprim), dir_q_1+"run_tprim-%.4f" % (tprim), dir_q_2+"run_tprim-%.4f" % (tprim)]
    #labels = [r"$q=0.7$", r"$q=1.4$", r"$q=2.8$"]
    #add_str += "_qinp_scan_tprim-%.4f" % (tprim)

    # Convergence scan
    #dirnames = [dir_4+"run_tprim-%.4f" % (tprim), dir_4+"run_tprim-%.4f_small_x0" % (tprim)]
    #labels = [r"Base case", r"Larger box"]
    #add_str += "_convergence_scan_tprim-%.4f" % (tprim)

     # nu=0 only
    dirnames = [dir_1+"run_tprim-%.4f" % (tprim)]
    labels = [None]
    add_str += "_tprim-%.4f" % (tprim)
    
    filenames_list = []
    labels_list    = []
    colors_list    = []
    ls_list        = []
    marker_list    = []
    tprim_list     = []
    qinp_list     = []
    codes_list     = []
    
    codes  = ["stella", "stella", "stella", "stella", "stella", "stella", "stella", "stella", "stella", "stella", "stella"]
    filenames = ["CBC", "CBC", "CBC", "CBC", "CBC", "CBC", "CBC", "CBC", "CBC", "CBC", "CBC", "CBC", "CBC"]
    
    colors = ["k", "orange", "crimson", "crimson", "forestgreen", "mediumblue", "purple", "c", "pink"]
    
    colors_list = colors
    labels_list = []
    codes_list  = codes

    for i_dir, dirname in enumerate(dirnames):
        filename = dirname + "/" + filenames[i_dir]
        if exists(filename+".nc") or exists(filename+".out.nc"):
            filenames_list.append(filename)
            tprim_list.append(tprim)
            labels_list.append(labels[i_dir])
            qinp_list.append(1.4)
    
    # Setup
    #for scaling_theory in ["CB", "GCB", "unscaled", "zonal_diffusive"]:
    for scaling_theory in ["unscaled"]:
        
        scanObj = lSS.loadStellaScan(filenames_list, labels_list, codes_list)
        
        scale_kmin = True
        
        # Phi2(ky)
        fig, ax = plt.subplots(figsize=figsize)
        fig, ax = scanObj.plot_phi_k_spectrum(fig=fig, ax=ax, plot_kx=False, color_list=colors_list, tprim_norm_list=tprim_list, qinp_norm_list=qinp_list, scale_kmin=scale_kmin, k_exp=0, scaling_theory=scaling_theory, load_from_file=load_from_file, delta_t_avg=delta_t_avg, plot_alpha_spectrum=plot_alpha_spectrum, lw=lw, W_instead_of_phi=W_instead_of_phi)
        ax.grid(True)
        ax.legend(fontsize=fontsize_legend, ncols=1, loc='lower left', columnspacing=0.7, markerscale=1, handlelength=1.5)
        figname = "fig_phi_ky_spectrum_" + scaling_theory
        if plot_alpha_spectrum:
            figname += "_alpha-spectrum"
            ax.set_ylim([-3,3])
            ax.axhline(-7/3, ls='--', c='0.5')
        else:
            if scaling_theory == "CB":
                if plot_slides:
                    ax.set_ylim([1e-8, 1e-1])
                else:
                    ax.set_ylim(ymin=1e-9)
            if scaling_theory == "GCB":
                if plot_slides:
                    ax.set_ylim([1e-6, 1e1])
                    ax.set_yticks([1e-6, 1e-3, 1e0])
                else:
                    ax.set_ylim(ymin=1e-4, ymax=2e0)
    
        plt.tight_layout()
        plt.savefig(figname+add_str+".pdf")
        plt.close()
    
        # Phi2_NZ(kx)
        fig, ax = plt.subplots(figsize=figsize)
        fig, ax = scanObj.plot_phi_k_spectrum(fig=fig, ax=ax, plot_kx=True, remove_zonal=True, color_list=colors_list, qinp_norm_list=qinp_list,  tprim_norm_list=tprim_list, scale_kmin=scale_kmin, k_exp=0, scaling_theory=scaling_theory, load_from_file=load_from_file, delta_t_avg=delta_t_avg, plot_alpha_spectrum=plot_alpha_spectrum, lw=lw, W_instead_of_phi=W_instead_of_phi)
    
        figname = "fig_phi_kx_spectrum_nonzonal_" + scaling_theory
        if plot_alpha_spectrum:
            figname += "_alpha-spectrum"
            ax.set_ylim([-3,3])
            ax.axhline(-7/3, ls='--', c='0.5')
        else:
            if scaling_theory == "CB":
                if plot_slides:
                    ax.set_ylim([1e-8, 1e-1])
                else:
                    ax.set_ylim(ymin=2e-10)
            if scaling_theory == "GCB":
                if plot_slides:
                    ax.set_ylim([1e-5, 1e2])
                else:
                    ax.set_ylim(ymin=5e-4, ymax=1e1)
    
        if add_arrows:
            
            x_or = 7; y_or = 1e-1
            x_dest = [0.6*7,  0.6*5.6, 0.6*4.9]
            y_dest = [2.8e-2,    5e-2,    6e-2]
            for i in range(len(x_dest)): 
                ax.plot([x_or, x_dest[i]], [y_or, y_dest[i]], c='k')
                #ax.arrow(x_or, y_or, x_dest[i]-x_or, y_dest[i]-y_or, head_width=0.02, shape='right')
            #ax.text(0.6*3.5, y_or*1.3, r"$k_x \rho_i \sim 0.5$", fontsize=fontsize_legend-8)
            #ax.text(x_or/2, y_or*3, r"Local peaks at"+'\n'+r"toroidal secondary"+'\n'+r"scale $k_x \rho_i \sim 0.5$", fontsize=fontsize_legend-8)
    
        #if add_inset:
        #    left, bottom, width, height = [0.2, 0.2, 0.3, 0.4]
        #    ax2 = fig.add_axes([left, bottom, width, height])
        #    _, _ = scanObj.plot_phi_k_spectrum(fig=fig, ax=ax2, plot_kx=True, color_list=colors_list, tprim_norm_list=tprim_list, qinp_norm_list=qinp_list, scale_kmin=scale_kmin, k_exp=0, ls_list=ls_list, scaling_theory="unscaled", load_from_file=load_from_file, delta_t_avg=delta_t_avg, plot_alpha_spectrum=plot_alpha_spectrum, lw=2*lw)
        #    ax2.set_xlim([0.1, 1.5])
        #    ax2.grid(False)
        #    ax2.set_xlabel(r"$k_x \rho_i$", fontsize=fontsize_legend-4)
        #    ax2.set_ylabel(None)
    
        ax.grid(True)
        ax.legend(fontsize=fontsize_legend, ncols=1, loc='lower left', columnspacing=0.7, markerscale=1, handlelength=1.5)
        plt.tight_layout()
        plt.savefig(figname+add_str+".pdf")    
    
        if overplot_kx_ky:
            fig, ax = scanObj.plot_phi_k_spectrum(fig=fig, ax=ax, plot_kx=False, color_list=colors_list, tprim_norm_list=tprim_list, qinp_norm_list=qinp_list, scale_kmin=scale_kmin, k_exp=0, scaling_theory=scaling_theory, load_from_file=load_from_file, delta_t_avg=delta_t_avg, plot_alpha_spectrum=plot_alpha_spectrum, lw=2*lw, W_instead_of_phi=W_instead_of_phi)
            ax.set_xlabel(r"$k_\perp \rho_i$")
            plt.savefig(figname+add_str+"_kxky-overplot.pdf")    
    
        plt.close()
    
        # Phi2_Z(kx)
        fig, ax = plt.subplots(figsize=figsize)
    
        if len(labels_list)==1:
            scanObj.list_labels = [r"$E_\mathrm{RH}$"]
            colors_list = ["crimson"]
            fontsize_legend = 24 
        fig, ax = scanObj.plot_phi_k_spectrum(fig=fig, ax=ax, plot_kx=True, only_zonal=True, color_list=colors_list, qinp_norm_list=qinp_list,  tprim_norm_list=tprim_list, scale_kmin=scale_kmin, k_exp=0, scaling_theory=scaling_theory, load_from_file=load_from_file, delta_t_avg=delta_t_avg, plot_alpha_spectrum=plot_alpha_spectrum, lw=4*lw, W_instead_of_phi=W_instead_of_phi, plot_RH_phi_spectrum=True)#, alpha_plot=0.5)
        if len(labels_list)==1:
            scanObj.list_labels = [r"$E_\varphi$"]
            colors_list = ["mediumblue"]
        fig, ax = scanObj.plot_phi_k_spectrum(fig=fig, ax=ax, plot_kx=True, only_zonal=True, color_list=colors_list, qinp_norm_list=qinp_list,  tprim_norm_list=tprim_list, scale_kmin=scale_kmin, k_exp=0, scaling_theory=scaling_theory, load_from_file=load_from_file, delta_t_avg=delta_t_avg, plot_alpha_spectrum=plot_alpha_spectrum, lw=4*lw,   W_instead_of_phi=W_instead_of_phi, alpha_plot=0.5)
    
        figname = "fig_phi_kx_spectrum_zonal_" + scaling_theory
        if plot_alpha_spectrum:
            figname += "_alpha-spectrum"
            ax.set_ylim([-3,3])
            ax.axhline(-7/3, ls='--', c='0.5')
    #    else:
    #       ax.set_xlim(xmax=1, xmin=0.4)
    #       if scaling_theory == "unscaled":
    #           ax.set_ylim([3e-2,1e2])
    #       elif scaling_theory == "zonal_diffusive":
    #           ax.set_ylim([1e-4,3e-2])
    
        ax.grid(True)
        if plot_legend:
            ax.legend(fontsize=fontsize_legend, ncols=1, loc='lower left', columnspacing=0.7, markerscale=1, handlelength=1.5)
        plt.tight_layout()
        plt.savefig(figname+add_str+".pdf")    
    
        #ax.set_yscale('linear')
        #plt.savefig(figname+add_str+"_lin.pdf")    
    
        plt.close()
