import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib import transforms
import traceback
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})
fontsize_labels = 32

import stellaDiagnostics as sD

# Load data
filename_base = "/CBC"

# Quantities to plot
labels     = [r"$v_E^Z$", r"$v_E^{\mathrm{RH},Z}$",        r"$Q$", r"$v_{Ex}^2$", r"$P_\mathrm{RH}$", r"$P_\varphi$", r"$P_\varphi^+$", r"$P_\varphi^-$",  r"$P_{\mathrm{RH}}^C$", r"$u_\parallel^Z$", r"$\partial_x T^Z$", r"$\partial_x p^Z$", r"$\Pi_\parallel$"]
datanames  = ["vE",        "vE_RH",                          "Q",       "vEx2",        "P_RH",          "P_RH_phi",       "P_phi_even",    "P_phi_odd",          "P_RH_coll", "upar",  "gradTZ", "gradPZ", "Pi_parallel"]

# Setup
sharex=False
#sharex=True

alpha = 0.5
lw    = 2

scan_type = "coll"
#scan_type = "qinp"
#scan_type = "shat"
#scan_type = "eps"

# Data directories
dirnames = []
dirnames_row1 = []
dirnames_row2 = []
dirnames_row3 = []

# SHAT SCAN
if scan_type == "shat":
#    sharex=False
    

    dirnames_row1.append("") 
    dirnames_row1.append("") 
    dirnames_row1.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000") 
    dirnames_row1.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-5.6000")
    dirnames.append(dirnames_row1)

    dirnames_row2.append("../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.2500/")
    dirnames_row2.append("../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.5000/")
    dirnames_row2.append("../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.7500/")
    dirnames_row2.append("../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-2.0000/")
    dirnames.append(dirnames_row2)

    dirnames_row3.append("")
    dirnames_row3.append("")
    dirnames_row3.append("../2026-01-20_ES_ITG_Dimits/2026-05-05_qinp-1.4_shat-0.16_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.7500/")
    dirnames_row3.append("../2026-01-20_ES_ITG_Dimits/2026-05-05_qinp-1.4_shat-0.16_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-2.0000/")
    dirnames.append(dirnames_row3)
    
    col_titles = [r"$R/L_T = 3.5$", r"$R/L_T = 4.2$", r"$R/L_T = 4.9$", r"$R/L_T = 5.6$"]
    row_titles = [r"$\hat s = 0.8$", r"$\hat s = 0.32$", r"$\hat s = 0.16$"]

# QINP SCAN
if scan_type == "qinp":
#    sharex=False
    
    dirnames_row1.append("")
    dirnames_row1.append("../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-2.2500/")
    dirnames_row1.append("../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-2.5000/")
    dirnames_row1.append("../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-3.0000/")
    dirnames.append(dirnames_row1)

    dirnames_row2.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000") 
    dirnames_row2.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000")
    dirnames_row2.append("")
    dirnames_row2.append("")
    dirnames.append(dirnames_row2)
    
    dirnames_row3.append("2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000")
    dirnames_row3.append("2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000")
    dirnames_row3.append("")
    dirnames_row3.append("")
    dirnames.append(dirnames_row3)
    
    col_titles = [r"$R/L_T = 4.9$", r"$R/L_T = 6.3$", r"$R/L_T = 7.0$", r"$R/L_T = 8.4$"]
    row_titles = [r"$q=1.0$", r"$q= 1.4$", r"$q=2.8$"]

# EPS SCAN
elif scan_type == "eps":
    dirnames_row1.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0/run_tprim-4.9000") 
    dirnames_row1.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0/run_tprim-6.3000")
    dirnames.append(dirnames_row1)

    dirnames_row2.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000") 
    dirnames_row2.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000")
    dirnames.append(dirnames_row2)
    
    dirnames_row3.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0/run_tprim-4.9000")
    dirnames_row3.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0/run_tprim-6.3000")
    dirnames.append(dirnames_row3)
    
    col_titles = [r"$R/L_T = 4.9$", r"$R/L_T = 6.3$"]
    row_titles = [r"$\epsilon = 0.045$", r"$\epsilon = 0.18$", r"$\epsilon = 0.36$"]

# COLLISIONALITY SCAN
elif scan_type == "coll":
    dirnames_row1.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-3.8500")
    dirnames_row1.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.2000")
    dirnames_row1.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000") 
    dirnames_row1.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-5.6000")
    dirnames_row1.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-5.9500")
    dirnames_row1.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000")
    dirnames_row1.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-8.4000")
    dirnames.append(dirnames_row1)
    
    dirnames_row2.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-3.8500") 
    dirnames_row2.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.2000") 
    dirnames_row2.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.9000")
    dirnames_row2.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-5.6000")
    dirnames_row2.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-5.9500")
    dirnames_row2.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-6.3000")
    dirnames_row2.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-8.4000")
    dirnames.append(dirnames_row2)
    
    dirnames_row3.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-3.8500") 
    dirnames_row3.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-4.2000") 
    dirnames_row3.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-4.9000")
    dirnames_row3.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-5.6000")
    dirnames_row3.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-5.9500")
    dirnames_row3.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-6.3000")
    dirnames_row3.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-8.4000")
    dirnames.append(dirnames_row3)
    
    
    col_titles = [r"$R/L_T = 3.85$", r"$R/L_T = 4.2$", r"$R/L_T = 4.9$", r"$R/L_T = 5.6$", r"$R/L_T = 5.95$", r"$R/L_T = 6.3$", r"$R/L_T = 8.4$"]
    row_titles = [r"$\nu_{ii}R/v_{Ti} = 0$", r"$\nu_{ii}R/v_{Ti} = 10^{-4}$", r"$\nu_{ii}R/v_{Ti} = 10^{-3}$"]

Nrows = len(row_titles)
Ncols = len(col_titles)

# Loop through quantities
for i_dataname, dataname in enumerate(datanames):
    label = labels[i_dataname]

    fig, axs = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=(8*Ncols, 5*Nrows), sharex=sharex)
    figname = "fig_"+scan_type+"_mean_quantities_x_zed"+"_"+dataname
    
    # Generate plot
    for i_row in range(Nrows):
        for i_col in range(Ncols):
    
            try:
                ax = axs[i_row, i_col]
                dirname = dirnames[i_row][i_col]
                filename = dirname + "/CBC"
                diagObj = sD.stellaDiagnostics(filename)
    
                dl_over_B_avg = diagObj.dl_over_B_avg()
    
                ax.axvline(0, c='k', alpha=0.25)
    
                rot  = transforms.Affine2D().rotate_deg(90)
                base = ax.transData
    
                # Plot quantity
                if dataname == "gradPZ":
                    dxn_zed_x = np.loadtxt(dirname+"/data_zed_x_gradnZ.dat")
                    dxT_zed_x = np.loadtxt(dirname+"/data_zed_x_gradTZ.dat")
                    f_zed_x = dxn_zed_x + dxT_zed_x

                else:
                    f_zed_x = np.loadtxt(dirname+"/data_zed_x_"+dataname+".dat")

    #            x       = np.loadtxt(dirname+"/data_zed_x_x.dat")
    #            zed     = np.loadtxt(dirname+"/data_zed_x_zed.dat")
    #            zed     = np.linspace(-np.pi, np.pi, len(f_zed_x[:,0]), endpoint=False)
    
                _, _, im, x, zed, f_zed_x, zed_avg_x, vmin, vmax = diagObj.plot_quantity_x_zed(quantity=f_zed_x, fig=fig, ax=ax, vmin="symm", cmap="coolwarm")
                cbar = fig.colorbar(im, ax=ax)
    
                # Average value
                f_x = np.sum(f_zed_x*dl_over_B_avg[:,None], axis=0)
                norm = (zed[-1]/2)/np.nanmax(np.abs(f_x))
                ax.plot(x, -f_x*norm, c='purple', alpha=alpha, transform=rot+base, label=r"$\langle f \rangle$", lw=lw)
    
                # Average zed
                ax.plot(x, -zed_avg_x, c='forestgreen', alpha=alpha, transform=rot+base, label=r"$\langle f \zeta \rangle / \langle f \rangle$", lw=lw)

                # Overplot 1D vE profile
                vE_zed_x = np.loadtxt(dirname+"/data_zed_x_vE.dat")
                vE_x = np.sum(vE_zed_x*dl_over_B_avg[:,None], axis=0)
    
                norm = (zed[-1]/2)/np.nanmax(np.abs(vE_x))
                ax.plot(x, -vE_x*norm, c='k', alpha=alpha, transform=rot+base, label=r"$v_E$", lw=lw)

                # RMS in x as a function of zed
                f_RMS_theta = np.sqrt( np.sum(f_zed_x**2, axis=1) )
                norm = x[-1]/f_RMS_theta.max()
                ax.plot(zed, norm*f_RMS_theta + x[0], c='mediumblue', alpha=alpha, label=r"$\langle f^2 \rangle_x^{1/2}$", lw=lw)
                ax.plot(zed, 0.5*(1+np.cos(zed))*x[-1] + x[0], c='mediumblue', alpha=alpha, ls='--', lw=lw)
    
            except Exception as e:
                print("COULD NOT LOAD " + dirname)
                print(e)
            
    # Finish plot
    for i_row in range(Nrows):
        axs[i_row,  0].set_ylabel(row_titles[i_row], fontsize=fontsize_labels)
    for i_col in range(Ncols):
        axs[0,  i_col].set_title(col_titles[i_col], fontsize=fontsize_labels)
        axs[-1, i_col].set_xlabel(r"$\theta$")
    
    axs[0,0].legend(fontsize=18)
    
    plt.tight_layout()
    fig.savefig(figname+".pdf")
    
    plt.close()
