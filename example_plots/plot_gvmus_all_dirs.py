import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
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

# Setup
dt_avg = 300

kx_min = 0
kx_max = 0.2

#kx_min = None
#kx_max = None

if kx_min is None and kx_max is None:
    label_kx = ""
else:
    label_kx = r"$(%.2f < |k_x| < %.2f)$" % (kx_min, kx_max)

nozonal = False
zonal   = True

#nozonal = True
#zonal   = False

sharex=True

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
fig, axs = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=(8*Ncols, 8*Nrows), sharex=sharex)
figname = "fig_gvmus_all_dirs"
if nozonal:
    figname += "_nozonal"
if zonal: 
    figname += "_zonal"
if kx_min is None and kx_max is None:
    figname += "_dtavg-%i" % (dt_avg)
else:
    figname += "_dtavg-%i_kxmin-%.2f_kxmax-%.2f" % (dt_avg, kx_min, kx_max)

# Generate plot
for i_row in range(Nrows):
    for i_col in range(Ncols):

        ax = axs[i_row, i_col]
        dirname = dirnames[i_row][i_col]

        filename = dirname + "/CBC"
        diagObj = sD.stellaDiagnostics(filename)

        try:
            _, _, im = diagObj.plot_contour_gvmu_vpa(time_idx=-1, logarithmic=True, vmin="symm", nozonal=nozonal, zonal=zonal, fig=fig, ax=ax, kx_min=kx_min, kx_max=kx_max, dt_avg=dt_avg)
            plt.colorbar(im, ax=ax)

        except Exception as e:
            print("COULD NOT LOAD " + dirname)
            print(e)

if nozonal:
    fig.suptitle(r"$V^{-1}\int\mathrm{d}^3 r \;g_\mathrm{NZ}^2 / F_M$ " + label_kx)
elif zonal:
    fig.suptitle(r"$V^{-1}\int\mathrm{d}^3 r \;g_\mathrm{Z}^2 / F_M$ " + label_kx)
else:
    fig.suptitle(r"$V^{-1}\int\mathrm{d}^3 r \;g^2 / F_M$ " + label_kx)
        
# Finish plot
for i_row in range(Nrows):
    axs[i_row,  0].set_ylabel(row_titles[i_row], fontsize=fontsize_labels)
for i_col in range(Ncols):
    axs[0,  i_col].set_title(col_titles[i_col], fontsize=fontsize_labels)

#    axs[-1, i_col].set_xlabel(r"$x/\rho_i$")
#    axs[-1, i_col].set_xlabel(r"$v_E^Z$")

#axs[0,0].legend(fontsize=18)

plt.tight_layout()
fig.savefig(figname+"_"+scan_type+".pdf")
