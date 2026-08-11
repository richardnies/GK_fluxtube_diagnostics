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
quantities_plot = "P_RH_scatter"
#quantities_plot = "Q"
quantities_plot = "P_RH"
quantities_plot = "Pi_RH"
#quantities_plot = "Z_profiles"

scatter_vE_norm_pow = 2
scatter_vE_excl_frac = 0.3

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

if quantities_plot in ["P_RH_scatter"]:
    fig_extra, ax_extra = plt.subplots(figsize=(9,5))
    sharex=False

Nrows = len(row_titles)
Ncols = len(col_titles)
fig, axs = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=(10*Ncols, 4*Nrows), sharex=sharex)
#fig, axs = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=(8*Ncols, 5*Nrows), sharex=sharex)
figname = "fig_mean_quantities_x"

#if Nrows == 1 and Ncols == 1:
#    axs = [[axs]]
#elif Nrows == 1 or Ncols == 1:
#    axs = [[axs]

#if quantities_plot == "Q":
#    fig.suptitle(r"$Q/Q_\mathrm{gB}$")

#labels     = [r"$v_E^Z$", r"$v_E^{\mathrm{RH},Z}$",        r"$Q$", r"$v_{Ex}^2$", r"$P_\mathrm{RH}$", r"$P_\varphi$", r"$P_\varphi^+$", r"$P_\varphi^-$",  r"$P_{\mathrm{RH}}^C$", r"$u_\parallel^Z$",r"$u_\parallel^Z\cos\theta$", r"$\partial_x T^Z$"]
#datanames  = ["vE",        "vE_RH",                          "Q",       "vEx2",        "P_RH",          "P_RH_phi",       "P_phi_even",    "P_phi_odd",          "P_RH_coll",           "upar",          "upar_cos",     "gradTZ"]

# Generate plot
for i_row in range(Nrows):
    for i_col in range(Ncols):

        ax = axs[i_row, i_col]
        dirname = dirnames[i_row][i_col]

        ax.grid(True, alpha=0.75)
        ax.axhline(0, c='k', alpha=0.75)

        try:
            x      = np.loadtxt(dirname+"/data_x.dat")
            vE     = np.loadtxt(dirname+"/data_vE.dat")
            vE_RH  = np.loadtxt(dirname+"/data_vE_RH.dat")

            # Heat flux plot
            if quantities_plot == "Q":
                Q = np.loadtxt(dirname+"/data_Q.dat")
                ax.plot(x, Q, c='crimson', lw=4, label=r"$Q/Q_\mathrm{gB}$")

                vE_norm = np.abs(Q).max()/np.abs(vE).max()

            # P_RH plots
            if quantities_plot == "P_RH_scatter":
                P_phi_even = np.loadtxt(dirname+"/data_P_phi_even.dat")
                P_phi_odd  = np.loadtxt(dirname+"/data_P_phi_odd.dat")
                P_RH_phi   = np.loadtxt(dirname+"/data_P_RH_phi.dat")
                P_RH_coll  = np.loadtxt(dirname+"/data_P_RH_coll.dat")
                P_RH_tot   = np.loadtxt(dirname+"/data_P_RH.dat")

                if scatter_vE_norm_pow > 0:

                    #vE_excl    = vE[        np.abs(vE)>scatter_vE_excl_frac*np.abs(vE).max()]
                    vE_excl    = vE_RH[     np.abs(vE)>scatter_vE_excl_frac*np.abs(vE).max()]
                    P_phi_even = P_phi_even[np.abs(vE)>scatter_vE_excl_frac*np.abs(vE).max()]/np.abs(vE_excl)**scatter_vE_norm_pow 
                    P_phi_odd  = P_phi_odd[ np.abs(vE)>scatter_vE_excl_frac*np.abs(vE).max()]/np.abs(vE_excl)**scatter_vE_norm_pow
                    P_RH_phi   = P_RH_phi[  np.abs(vE)>scatter_vE_excl_frac*np.abs(vE).max()]/np.abs(vE_excl)**scatter_vE_norm_pow
                    P_RH_coll  = P_RH_coll[ np.abs(vE)>scatter_vE_excl_frac*np.abs(vE).max()]/np.abs(vE_excl)**scatter_vE_norm_pow
                    P_RH_tot   = P_RH_tot[  np.abs(vE)>scatter_vE_excl_frac*np.abs(vE).max()]/np.abs(vE_excl)**scatter_vE_norm_pow
                    vE         = vE_excl

                    label_add = r"$/v_E^%i$" % (scatter_vE_norm_pow)

                    scan_type += "_norm-vE-%i" % (scatter_vE_norm_pow)

                else:
                    label_add = ""

                ax.scatter(vE, P_phi_even, c='crimson',     alpha=0.75, label=r"$P_\mathrm{RH, \varphi}^+$"+label_add)
                ax.scatter(vE, P_phi_odd,  c='mediumblue',  alpha=0.75, label=r"$P_\mathrm{RH, \varphi}^-$"+label_add)
                ax.scatter(vE, P_RH_phi,   c='forestgreen', alpha=0.75, label=r"$P_\mathrm{RH, \varphi}$"+label_add)
                ax.scatter(vE, P_RH_coll,  c='orange',      alpha=0.75, label=r"$P_\mathrm{RH}^C$"+label_add)
                ax.scatter(vE, P_RH_tot,   c='k',           alpha=0.75, label=r"$P_\mathrm{RH}^\mathrm{tot}$"+label_add)

                # Extra plot of P_RH_C/nu for all dirs
                try:
                    diagObj = sD.stellaDiagnostics(dirname+"/CBC")
                    nu = diagObj.ncdata['vnew'][0]
                    ax_extra.scatter(vE, -P_RH_coll/nu, alpha=0.75, label=dirname)
                except Exception as e:
                    print(e)
                    print("Could not plot P/nu for " + dirname)
        

            # P_RH plots
            if quantities_plot == "P_RH":
                P_phi_even = np.loadtxt(dirname+"/data_P_phi_even.dat")
                P_phi_odd  = np.loadtxt(dirname+"/data_P_phi_odd.dat")
                P_RH_phi   = np.loadtxt(dirname+"/data_P_RH_phi.dat")
                P_RH_coll  = np.loadtxt(dirname+"/data_P_RH_coll.dat")
                P_RH_tot   = np.loadtxt(dirname+"/data_P_RH.dat")

                ax.plot(x, P_phi_even, c='crimson',     lw=4, label=r"$P_\mathrm{RH, \varphi}^+$")
                ax.plot(x, P_phi_odd,  c='mediumblue',  lw=4, label=r"$P_\mathrm{RH, \varphi}^-$")
#                ax.plot(x, P_RH_phi,   c='forestgreen', lw=4, label=r"$P_\mathrm{RH, \varphi}$")
                ax.plot(x, P_RH_coll,  c='orange',      lw=4, label=r"$P_\mathrm{RH}^C$")
                ax.plot(x, P_RH_tot,   c='k',           lw=4, label=r"$P_\mathrm{RH}^\mathrm{tot}$")

                vE_norm = (np.abs(P_phi_even).max())/np.abs(vE).max()

            # Pi_RH plots
            if quantities_plot == "Pi_RH":
                Pi_RH_NL    = np.loadtxt(dirname+"/data_Pi_RH_NL.dat")
                Pi_RH_even  = np.loadtxt(dirname+"/data_Pi_RH_even.dat")
                Pi_RH_odd   = np.loadtxt(dirname+"/data_Pi_RH_odd.dat")
                Pi_parallel = np.loadtxt(dirname+"/data_Pi_parallel.dat")

                ax.plot(x, Pi_RH_even,  c='crimson',     lw=4, label=r"$\Pi_\mathrm{RH}^+$")
                ax.plot(x, Pi_RH_odd,   c='mediumblue',  lw=4, label=r"$\Pi_\mathrm{RH}^-$")
                ax.plot(x, Pi_RH_NL,    c='k',           lw=4, label=r"$\Pi_\mathrm{RH}$")
                ax.plot(x, Pi_parallel, c='purple',      lw=4, label=r"$\Pi_\parallel$", alpha=0.5)

                vE_norm = (np.abs(Pi_RH_even).max())/np.abs(vE).max()

            # Z_profile plots
            if quantities_plot == "Z_profiles":
                upar     = np.loadtxt(dirname+"/data_upar.dat")
                upar_cos = np.loadtxt(dirname+"/data_upar_cos.dat")
                dxTZ     = np.loadtxt(dirname+"/data_gradTZ.dat")

                gammaE_RH = np.gradient(vE_RH, x)

                ax.plot(x, vE_RH,        c='k',           lw=4, label=r"$v_E^\mathrm{RH}$")
                ax.plot(x, gammaE_RH*10, c='purple',      lw=8, label=r"$10\gamma_E^\mathrm{RH}$")
                ax.plot(x, dxTZ*5,       c='crimson',     lw=4, label=r"$5\partial_x T^Z$")
                ax.plot(x, upar,         c='mediumblue',  lw=4, label=r"$\langle u_\parallel \rangle_\psi$")
                ax.plot(x, upar_cos,     c='forestgreen', lw=4, label=r"$\langle u_\parallel \cos\theta\rangle_\psi$")

                vE_norm = 1

            if quantities_plot in ["Q", "P_RH", "Z_profiles", "Pi_RH"]:
                ax.plot(x, vE*vE_norm, c='k', alpha=0.75, lw=2, label=r"$v_E^Z$ (a.u.)", ls='--')
                ax.set_xlim([x[0], x[-1]])

        except Exception as e:
            print("COULD NOT LOAD " + dirname)
            print(e)
        
# Finish plot
for i_row in range(Nrows):
    axs[i_row,  0].set_ylabel(row_titles[i_row], fontsize=fontsize_labels)
for i_col in range(Ncols):
    axs[0,  i_col].set_title(col_titles[i_col], fontsize=fontsize_labels)

    if quantities_plot in ["Q", "P_RH", "Z_profiles"]:
        axs[-1, i_col].set_xlabel(r"$x/\rho_i$")
    elif quantities_plot in ["P_RH_scatter"]:
        axs[-1, i_col].set_xlabel(r"$v_E^Z$")

axs[0,0].legend(fontsize=18)

fig.tight_layout()
fig.savefig(figname+"_"+quantities_plot+"_"+scan_type+".pdf")

if quantities_plot in ["P_RH_scatter"]:
   # ax_extra.legend(fontsize=8) 
    ax_extra.set_xlabel(r"$v_E$")
    ax_extra.set_ylabel(r"$-\nu_{ii}^{-1} P_\mathrm{RH}^C$"+label_add)
    ax_extra.set_yscale('log')
    fig_extra.tight_layout()
    fig_extra.savefig("fig_PRH_nu_"+scan_type+".pdf")
