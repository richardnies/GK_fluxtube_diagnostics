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

def get_aLT_lin_analytic(rhoc, q, shat, eps=1, tau=1):
    return (1+tau)*(1.33 + 1.91*shat/q) * (1 - 1.5*eps*rhoc)*eps

import stellaDiagnostics as sD

# Load data
filename_base = "/CBC"

from glob import glob
import seaborn as sns
import json

base_dirs = ["2026-06-26_scan_qinp-0.7_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0",
             "2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001"]
base_labels = [r"$q=0.7$", r"$q = 1.4$", r"$q = 2.8$", r"$\epsilon = 0.045$", r"$\epsilon = 0.36$", r"$\nu = 10^{-3}$"]
figname_add = "_geo_scan"
aLT_lin_vals = get_aLT_lin_analytic( rhoc = np.array([0.18,0.18,0.18,0.045,0.36,0.18]),
                                     q    = np.array([0.7, 1.4, 2.8, 1.4,   1.4, 1.4]),
                                     shat = np.array([0.8, 0.8, 0.8, 0.8,   0.8, 0.8]) )

base_dirs = ["2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0"]
base_labels = [r"$\epsilon = 0.045$",r"$\epsilon = 0.18$", r"$\epsilon = 0.36$"]
figname_add = "_eps_scan"
aLT_lin_vals = get_aLT_lin_analytic( rhoc = np.array([0.045,0.18,0.36]),
                                     q    = np.array([ 1.4,  1.4, 1.4]),
                                     shat = np.array([ 0.8,  0.8, 0.8]) )
tprim_crit_approx_vals = [6, 6, 6, 6, 6, 6]

base_dirs = ["2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-5",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-5",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-4",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-3",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.01"]
base_labels = [r"$\nu^*=0$", r"$\nu^*=10^{-5}$", r"$\nu^*=3\cdot 10^{-5}$", r"$\nu^*=10^{-4}$", r"$\nu^*=3\cdot 10^{-4}$", r"$\nu^*=10^{-3}$", r"$\nu^*=3\cdot 10^{-3}$", r"$\nu^*=10^{-2}$"]
figname_add = "_nu_scan"

base_dirs = ["2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001",
             "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001"]
base_labels = [r"$\nu_{ii} R / v_{Ti} = 0$", r"$\nu_{ii} R / v_{Ti}=10^{-4}$", r"$\nu_{ii} R / v_{Ti}=10^{-3}$"]
figname_add = "_nu_scan_red"

tprim_crit_approx_vals = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
aLT_lin_vals = get_aLT_lin_analytic( rhoc = np.array([0.18,0.18,0.18,0.18,0.18, 0.18, 0.18, 0.18, 0.18, 0.18]),
                                     q    = np.array([1.4, 1.4, 1.4, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4]),
                                     shat = np.array([0.8, 0.8, 0.8, 0.8,  0.8,  0.8,  0.8,  0.8,  0.8,  0.8]) )

tprim_exclude = []
#tprim_exclude = [3.85, 4.2, 4.9, 5.25, 5.6, 5.95, 6.3]

base_colors = sns.color_palette("rocket", len(base_dirs))
#base_colors = sns.color_palette("hls", len(base_dirs))

#aLT_lin_vals = np.zeros_like(aLT_lin_vals)
#aLT_lin_vals = np.array([1.9, 1.4, 1.1, 0.7, 0.8, 1.4, 1.4, 1.4])

substract_lin = False

#tprim_val = 8.4
tprim_val = None

xlim = [3.6, 8.7]

markersize = 10

nrows=15
fig, axs = plt.subplots(nrows=nrows, figsize=(8, 4*nrows))#, sharex=True)

for i_base, base_dir in enumerate(base_dirs):

    label = base_labels[i_base]
    color = base_colors[i_base]
    tprim_crit_approx = tprim_crit_approx_vals[i_base]

    if tprim_val is None:
        dirnames = sorted(glob(base_dir+"/run_tprim*00/"))
    else:
        dirnames = [base_dir+"/run_tprim-%.4f/" % (tprim_val)]

    ndirs = len(dirnames)

    # Load data
    tprim_vals             = np.zeros(ndirs)
    qinp_vals              = np.zeros(ndirs)
    eps_vals               = np.zeros(ndirs)
    qflx_avg_vals          = np.zeros(ndirs)
    qflx_std_vals          = np.zeros(ndirs)
    gammaE_avg_vals        = np.zeros(ndirs)
    gammaE_std_vals        = np.zeros(ndirs)
    gammaE_LW_avg_vals     = np.zeros(ndirs)
    gammaE_LW_std_vals     = np.zeros(ndirs)
    upar_avg_vals          = np.zeros(ndirs)
    vE_avg_vals            = np.zeros(ndirs)
    vE_RH_avg_vals         = np.zeros(ndirs)
    uparcos_avg_vals       = np.zeros(ndirs)
    dxT_avg_vals           = np.zeros(ndirs)
    gammaE_RH_avg_vals     = np.zeros(ndirs)
    gammaE_RH_std_vals     = np.zeros(ndirs)
    gammaE_RH_LW_avg_vals  = np.zeros(ndirs)
    gammaE_RH_LW_std_vals  = np.zeros(ndirs)
    gamma_lin_max_vals     = np.zeros(ndirs)
    P_RH_even_avg_vals_alt = np.zeros(ndirs)
    P_RH_odd_avg_vals_alt  = np.zeros(ndirs)
    P_RH_even_avg_vals     = np.zeros(ndirs)
    P_RH_odd_avg_vals      = np.zeros(ndirs)
    P_RH_coll_avg_vals     = np.zeros(ndirs)
    P_RH_even_avg_LW_vals  = np.zeros(ndirs)
    P_RH_odd_avg_LW_vals   = np.zeros(ndirs)
    P_RH_coll_avg_LW_vals  = np.zeros(ndirs)
    dyphi2_avg_vals        = np.zeros(ndirs)
    dyphi2_avg_vEpos_vals  = np.zeros(ndirs)
    dyphi2_avg_vEneg_vals  = np.zeros(ndirs)

    for i_dir, dirname in enumerate(dirnames):
    
        try:
            f = open(dirname+'data_Dimits.json')
            data_dict = json.load(f)

            print("-----------------------------------")
            print(dirname + ":")
            print(data_dict)
            print("-----------------------------------")

            tprim_vals[i_dir]             = data_dict['tprim']
            qinp_vals[i_dir]              = data_dict['qinp']
            eps_vals[i_dir]               = data_dict['eps']
            qflx_avg_vals[i_dir]          = data_dict['qflx_avg']
            qflx_std_vals[i_dir]          = data_dict['qflx_std']
            gammaE_avg_vals[i_dir]        = data_dict['gammaE_avg']
            upar_avg_vals[i_dir]          = data_dict['upar_avg']
            uparcos_avg_vals[i_dir]       = data_dict['uparcos_avg']
            dxT_avg_vals[i_dir]           = data_dict['dxT_avg']
            gammaE_RH_avg_vals[i_dir]     = data_dict['gammaE_RH_avg']
            vE_avg_vals[i_dir]            = data_dict['vE_avg']
            vE_RH_avg_vals[i_dir]         = data_dict['vE_RH_avg']
            gamma_lin_max_vals[i_dir]     = data_dict['gamma_lin_max']
            P_RH_odd_avg_vals_alt[i_dir]  = data_dict['P_RH_odd_avg_alt']
            P_RH_even_avg_vals_alt[i_dir] = data_dict['P_RH_even_avg_alt']
            P_RH_odd_avg_vals[i_dir]      = data_dict['P_RH_odd_avg']
            P_RH_even_avg_vals[i_dir]     = data_dict['P_RH_even_avg']
            P_RH_odd_avg_LW_vals[i_dir]   = data_dict['P_RH_odd_avg_LW']
            P_RH_even_avg_LW_vals[i_dir]  = data_dict['P_RH_even_avg_LW']
    
        except Exception as e:
            print(e)
            print("Could not load file for " + dirname)
    
            tprim_vals[i_dir]             = np.nan 
            qinp_vals[i_dir]              = np.nan 
            eps_vals[i_dir]               = np.nan 
            qflx_avg_vals[i_dir]          = np.nan 
            qflx_std_vals[i_dir]          = np.nan 
            gammaE_avg_vals[i_dir]        = np.nan 
            upar_avg_vals[i_dir]          = np.nan 
            uparcos_avg_vals[i_dir]       = np.nan 
            dxT_avg_vals[i_dir]           = np.nan 
            gammaE_RH_avg_vals[i_dir]     = np.nan 
            vE_avg_vals[i_dir]            = np.nan 
            vE_RH_avg_vals[i_dir]         = np.nan 
            gamma_lin_max_vals[i_dir]     = np.nan 
            P_RH_odd_avg_vals[i_dir]      = np.nan 
            P_RH_even_avg_vals[i_dir]     = np.nan 
            P_RH_odd_avg_LW_vals[i_dir]   = np.nan 
            P_RH_even_avg_LW_vals[i_dir]  = np.nan 

        try:
            gammaE_LW_avg_vals[i_dir]    = data_dict['gammaE_LW_avg']
            gammaE_RH_LW_avg_vals[i_dir] = data_dict['gammaE_RH_LW_avg']
        except:
            gammaE_LW_avg_vals[i_dir]    = np.nan 
            gammaE_RH_LW_avg_vals[i_dir] = np.nan 

        try:
            dyphi2_avg_vals[i_dir]        = data_dict['dyphi2_avg'] 
            dyphi2_avg_vEpos_vals[i_dir]  = data_dict['dyphi2_avg_vEpos'] 
            dyphi2_avg_vEneg_vals[i_dir]  = data_dict['dyphi2_avg_vEneg'] 
        except:
            dyphi2_avg_vals[i_dir]        = np.nan 
            dyphi2_avg_vEpos_vals[i_dir]  = np.nan 
            dyphi2_avg_vEneg_vals[i_dir]  = np.nan 

        try:
            gammaE_std_vals[i_dir]        = data_dict['gammaE_std']
            gammaE_RH_std_vals[i_dir]     = data_dict['gammaE_RH_std']
        except:
            gammaE_std_vals[i_dir]        = 0
            gammaE_RH_std_vals[i_dir]     = 0

        try:
            gammaE_LW_avg_vals[i_dir]     = data_dict['gammaE_LW_avg']
            gammaE_RH_LW_avg_vals[i_dir]  = data_dict['gammaE_RH_LW_avg']
            gammaE_LW_std_vals[i_dir]     = data_dict['gammaE_LW_std']
            gammaE_RH_LW_std_vals[i_dir]  = data_dict['gammaE_RH_LW_std']
        except:
            gammaE_LW_avg_vals[i_dir]     = np.nan 
            gammaE_RH_LW_avg_vals[i_dir]  = np.nan 
            gammaE_LW_std_vals[i_dir]     = np.nan 
            gammaE_RH_LW_std_vals[i_dir]  = np.nan 

        try:
            P_RH_coll_avg_vals[i_dir]     = data_dict['P_RH_coll_avg']
            P_RH_coll_avg_LW_vals[i_dir]  = data_dict['P_RH_coll_avg_LW']
            if P_RH_coll_avg_vals[i_dir] == 0:
                P_RH_coll_avg_vals[i_dir]     = np.nan 
                P_RH_coll_avg_LW_vals[i_dir]  = np.nan 
        except:
            P_RH_coll_avg_vals[i_dir]     = np.nan 
            P_RH_coll_avg_LW_vals[i_dir]  = np.nan 

    if substract_lin:
        tprim_vals += -aLT_lin_vals[i_base]
        tprim_crit_approx += -aLT_lin_vals[i_base]

   # mask = (np.isfinite(qflx_avg_vals)) & (tprim_vals not in tprim_exclude)
    mask = (
    np.isfinite(qflx_avg_vals)
    & ~np.isin(tprim_vals, tprim_exclude)
    )

    ##### Plot
    # Heat flux
    i=0
    ax = axs[i]
    ax.errorbar(tprim_vals[mask], qflx_avg_vals[mask], qflx_std_vals[mask], c=color, label=label, lw=2, marker='o', markersize=markersize)
    ax.set_ylabel(r"$Q/Q_\mathrm{gB}$")
    ax.set_ylim(ymin=0)

#    # Heat flux normalised by vEx^2
    # Heat flux on log scale
    i+=1
    ax = axs[i]
    #ax.plot(tprim_vals, qflx_avg_vals/dyphi2_avg_vals, c=color, label=label, lw=2, marker='o', markersize=markersize)
    #ax.set_ylabel(r"$Q/v_{Ex}^2$")
    ax.errorbar(tprim_vals[mask], qflx_avg_vals[mask], qflx_std_vals[mask], c=color, label=label, lw=2, marker='o', markersize=markersize)
#    ax.set_xscale('log')
    ax.set_yscale('log')
#    tprim_plot = np.linspace(tprim_vals.min(), tprim_vals.max(), 100)
#    ax.plot(tprim_plot, tprim_plot**5, ls='--', c='k')
#    ax.plot(tprim_plot, tprim_plot*10, ls='--', c='k')
    ax.set_ylabel(r"$Q/Q_\mathrm{gB}$")

    # Flow shear and max linear growth rate
    i+=1
    ax = axs[i]
    if i_base == 0:
        labels = [r"$\gamma_E \, R/v_{Ti}$", r"$\gamma_{E, \mathrm{RH}} \, R/v_{Ti}$", r"$\gamma_\mathrm{lin}^\mathrm{max} \, R / v_{Ti}$"]
    else:
        labels = [None, None, None]
    ax.errorbar(tprim_vals[mask], gammaE_avg_vals[mask],    0.5*gammaE_std_vals[mask],    lw=2, c=color, ls='--',  marker='o', markersize=markersize)
    ax.errorbar(tprim_vals[mask], gammaE_RH_avg_vals[mask], 0.5*gammaE_RH_std_vals[mask], lw=2, c=color, ls='--',  marker='s', markersize=markersize, alpha=0.3)
    #ax.plot(tprim_vals[mask], gammaE_RH_LW_avg_vals[mask], lw=2, c=color, ls='--',  marker='s', markersize=markersize, alpha=0.5)
    ax.plot(tprim_vals[mask], gamma_lin_max_vals[mask], lw=4, c=color, ls='-', marker='x', markersize=markersize, alpha=0.5)

    ax.plot([], [], c='k', ls='None',  lw=2, marker='o', label=labels[0], markersize=markersize)
    ax.plot([], [], c='k', ls='--',  lw=2, marker='s', label=labels[1], markersize=markersize, alpha=0.3)
    ax.plot([], [], c='k', ls='-', lw=4, marker='x', label=labels[2], markersize=markersize)

    ax.set_ylim(ymin=0)

    # LW Flow shear and max linear growth rate
    i+=1
    ax = axs[i]
    if i_base == 0:
        labels = [r"$\gamma_E^\mathrm{LW} \, R/v_{Ti}$", r"$\gamma_{E, \mathrm{RH}}^\mathrm{LW} \, R/v_{Ti}$", r"$\gamma_\mathrm{lin}^\mathrm{max} \, R / v_{Ti}$"]
        #labels = [r"$\gamma_E \, R/v_{Ti}$", r"$\gamma_{E, \mathrm{RH}} \, R/v_{Ti}$", r"$\gamma_\mathrm{lin}^\mathrm{max} \, R / v_{Ti}$"]
    else:
        labels = [None, None, None]
    ax.errorbar(tprim_vals[mask], gammaE_LW_avg_vals[mask],    0.5*gammaE_LW_std_vals[mask],    lw=2, c=color, ls='--',  marker='o', markersize=markersize)
    #ax.errorbar(tprim_vals[mask], gammaE_RH_LW_avg_vals[mask], 0.5*gammaE_RH_LW_std_vals[mask], lw=2, c=color, ls='--',  marker='s', markersize=markersize, alpha=0.3)
    #ax.plot(tprim_vals[mask], gammaE_RH_LW_avg_vals[mask], lw=2, c=color, ls='--',  marker='s', markersize=markersize, alpha=0.5)
    ax.plot(tprim_vals[mask], gamma_lin_max_vals[mask], lw=4, c=color, ls='-', marker='x', markersize=markersize, alpha=0.5)

    ax.plot([], [], c='k', ls='--',  lw=2, marker='o', label=labels[0], markersize=markersize)
    #ax.plot([], [], c='k', ls='--',  lw=2, marker='s', label=labels[1], markersize=markersize, alpha=0.3)
    ax.plot([], [], c='k', ls='-', lw=4, marker='x', label=labels[2], markersize=markersize)

    ax.set_ylim(ymin=0, ymax=0.25)

    # P_RH/P_RH^+
    i+=1
    ax = axs[i]
    if i_base == 0:
        labels = [r"$-P_\mathrm{RH}^C/P_\mathrm{RH}^+$", r"$-P_\mathrm{RH}^-/P_\mathrm{RH}^+$", r"$-(P_\mathrm{RH}^C+P_\mathrm{RH}^-)/P_\mathrm{RH}^+$"]
    else:
        labels = [None, None, None]

    ax.plot(tprim_vals,-P_RH_coll_avg_vals / P_RH_even_avg_vals,  c=color, ls='-.',lw=2, marker='v', markersize=markersize, alpha=0.5)
    ax.plot(tprim_vals,-P_RH_odd_avg_vals  / P_RH_even_avg_vals,  c=color, ls=':', lw=2, marker='s', markersize=markersize, alpha=0.5)
    ax.plot(tprim_vals,-(P_RH_coll_avg_vals+P_RH_odd_avg_vals) / P_RH_even_avg_vals,  c=color, ls='-',lw=4, marker='o', markersize=2*markersize)

    ax.plot([], [], c='k', ls='-.', lw=2, marker='v', label=labels[0], markersize=markersize, alpha=0.5)
    ax.plot([], [], c='k', ls=':',  lw=2, marker='s', label=labels[1], markersize=markersize, alpha=0.5)
    ax.plot([], [], c='k', ls='-',  lw=4, marker='o', label=labels[2], markersize=markersize)

    # P_RH +-
    i+=1
    ax = axs[i]
    if i_base == 0:
        labels = [r"$P_\mathrm{RH}^+$", r"$-P_\mathrm{RH}^-$", r"$-P_\mathrm{RH}^C$"]
    else:
        labels = [None, None, None]

    ax.plot(tprim_vals, P_RH_even_avg_vals,    c=color, ls='-', lw=2, marker='o', markersize=markersize)
    ax.plot(tprim_vals,-P_RH_odd_avg_vals,     c=color, ls=':', lw=2, marker='s', markersize=markersize)
    ax.plot(tprim_vals,-P_RH_coll_avg_vals,    c=color, ls='-.',lw=2, marker='v', markersize=markersize)

    ax.plot([], [], c='k', ls='-',  lw=2, marker='o', label=labels[0], markersize=markersize)
    ax.plot([], [], c='k', ls=':',  lw=2, marker='s', label=labels[1], markersize=markersize)
    ax.plot([], [], c='k', ls='-.', lw=2, marker='v', label=labels[2], markersize=markersize)
    #ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_yscale("symlog", linthresh=1e-5)

   # ax.plot([], [], c='k', ls='-',  lw=2, marker='o', label=labels[0], markersize=markersize)
    #ax.set_yscale("symlog", linthresh=1e-2)
   # ax.set_yscale("symlog", linthresh=1e-5)

    # P_RH +- / vEx^2
    # P_RH (linear)
    i+=1
    ax = axs[i]
    if i_base == 0:
        labels = [r"$P_\mathrm{RH}^+$", r"$P_\mathrm{RH}^-$", r"$P_\mathrm{RH}$"]
        #labels = [r"$P_\mathrm{RH}^+ / v_{Ex}^2$", r"$-P_\mathrm{RH}^- /  v_{Ex}^2$"]
    else:
        labels = [None, None, None]

    ax.plot(tprim_vals, P_RH_even_avg_vals, c=color, ls='-', lw=2, marker='o', markersize=markersize)
    ax.plot(tprim_vals, P_RH_odd_avg_vals , c=color, ls=':', lw=2, marker='s', markersize=markersize)
    P_RH_tot_avg_vals = P_RH_even_avg_vals + P_RH_odd_avg_vals
    ax.plot(tprim_vals, P_RH_tot_avg_vals,  c=color, ls='-.', lw=2, marker='.', markersize=markersize)

    ax.plot([], [], c='k', ls='-',  lw=2, marker='o', label=labels[0], markersize=markersize)
    ax.plot([], [], c='k', ls=':',  lw=2, marker='s', label=labels[1], markersize=markersize)
    ax.plot([], [], c='k', ls='-.', lw=2, marker='.', label=labels[2], markersize=markersize)

    # P_RH +- (LW)
    i+=1
    ax = axs[i]
    if i_base == 0:
        labels = [r"$P_\mathrm{RH}^+$ (LW)", r"$-P_\mathrm{RH}^-$ (LW)", r"$-P_\mathrm{RH}^C$ (LW)"]
    else:
        labels = [None, None, None]

    ax.plot(tprim_vals, P_RH_even_avg_LW_vals,    c=color, ls='-', lw=2, marker='o', markersize=markersize)
    ax.plot(tprim_vals,-P_RH_odd_avg_LW_vals,     c=color, ls=':', lw=2, marker='s', markersize=markersize)
    ax.plot(tprim_vals,-P_RH_coll_avg_LW_vals,    c=color, ls='-.', lw=2, marker='v', markersize=markersize)

    ax.plot([], [], c='k', ls='-',  lw=2, marker='o', label=labels[0], markersize=markersize)
    ax.plot([], [], c='k', ls=':',  lw=2, marker='s', label=labels[1], markersize=markersize)
    ax.plot([], [], c='k', ls='-.', lw=2, marker='v', label=labels[2], markersize=markersize)
    ax.set_yscale("symlog", linthresh=1e-5)

    # Zonal parallel flow
    i+=1
    ax = axs[i]
    if i_base == 0:
        labels = [r"$u_\parallel / (q/\epsilon \cdot v_E)$", r"$2 u_\parallel \cos\theta / (q v_E)$"]
        #labels = [r"$u_\parallel / (1.6 \epsilon^{1/2} q v_E)$", r"$2 u_\parallel \cos\theta / (q v_E)$"]
    else:
        labels = [None, None, None]
    ax.plot(tprim_vals, 2*upar_avg_vals/(qinp_vals/eps_vals*vE_RH_avg_vals),    lw=2, c=color, ls='-',  marker='o', markersize=markersize)
    ax.plot(tprim_vals, 2*2*uparcos_avg_vals/(2*qinp_vals*vE_RH_avg_vals), lw=2, c=color, ls=':',  marker='s', markersize=markersize)

    ax.plot([], [], c='k', ls='-',  lw=2, marker='o', label=labels[0], markersize=markersize)
    ax.plot([], [], c='k', ls=':',  lw=2, marker='s', label=labels[1], markersize=markersize)
    ax.set_ylim(ymin=0)

    # Ratio of <upar>/<upar*cos(theta)>
    i+=1
    ax = axs[i]
    #ax.plot(tprim_vals, 2*eps_vals * upar_avg_vals/uparcos_avg_vals, lw=2, c=color, ls='-',  marker='o', markersize=markersize)
    #ax.set_ylabel(r"$2\epsilon \langle u_\parallel \rangle_\theta / \langle u_\parallel \cos\theta \rangle_\theta$")
    ax.plot(tprim_vals, 0.5* upar_avg_vals/uparcos_avg_vals, lw=2, c=color, ls='-',  marker='o', markersize=markersize)
    ax.set_ylabel(r"$\langle u_\parallel \rangle_\theta / 2\langle u_\parallel \cos\theta \rangle_\theta$")

    # <upar>
    i+=1
    ax = axs[i]
    ax.plot(tprim_vals, upar_avg_vals, lw=2, c=color, ls='-',  marker='o', markersize=markersize)
    ax.set_ylabel(r"$ \langle u_\parallel \rangle_\theta $")
#    ax.set_ylim(ymin=0)

    # Zonal temperature gradient
    i+=1
    ax = axs[i]
    if i_base == 0:
        labels = [r"$v_\mathrm{dia}/v_E$"]
        #labels = [r"$\partial_x T / \kappa_T$"]
    else:
        labels = [None, None, None]

    ax.plot(tprim_vals, dxT_avg_vals/vE_avg_vals,     lw=2, c=color, ls='--', marker='x', markersize=markersize)
    #ax.plot(tprim_vals, dxT_avg_vals/tprim_vals,     lw=2, c=color, ls='--', marker='x', markersize=markersize)
    ax.plot([], [], c='k', ls='--', lw=2, marker='x', label=labels[0], markersize=markersize)
    ax.set_ylim(ymin=0)

    # Plot ratio of LW and SW ZF shear
    i+=1
    ax = axs[i]
    #ax.plot(tprim_vals, gammaE_LW_avg_vals/gammaE_avg_vals, lw=2, c=color,  marker='s', markersize=markersize)
    #ax.plot(tprim_vals, gammaE_RH_LW_avg_vals/gammaE_RH_avg_vals, lw=2, c=color,  marker='s', markersize=markersize, alpha=0.5)
    #ax.set_ylabel(r"$\gamma_{E}^\mathrm{LW}/\gamma_{E}$")
    ax.plot(tprim_vals, np.abs(gammaE_RH_avg_vals-gammaE_avg_vals)/gammaE_avg_vals, lw=2, c=color,  marker='s', markersize=markersize)
    ax.set_ylabel(r"$|\gamma_{E}-\gamma_{E, \mathrm{EH}}|/\gamma_{E}$")
    ax.set_ylim(ymax=1)

#    # Plot ratio of turbulence at vE >< 0
    # Plot turbulence amplitude (total and vE><0)
    i+=1
    ax = axs[i]
    ax.semilogy(tprim_vals, qflx_avg_vals*10,      lw=2, c=color, marker='o', markersize=markersize, alpha=0.5)
    ax.semilogy(tprim_vals, dyphi2_avg_vals,       lw=2, c=color, marker='s', markersize=markersize)
#    ax.semilogy(tprim_vals, dyphi2_avg_vEpos_vals, lw=1, c=color, marker='s', markersize=markersize, ls='--', alpha=0.5)
#    ax.semilogy(tprim_vals, dyphi2_avg_vEneg_vals, lw=1, c=color, marker='s', markersize=markersize, ls='-.', alpha=0.5)
    ax.set_ylabel(r"$\tilde v_{Ex}^2$")
#    ax.set_ylim([1e-1, 1e1])

    # Plot ratio of vEx^2/vEZ^2
    i+=1
    ax = axs[i]
    ax.semilogy(tprim_vals, dyphi2_avg_vals/vE_avg_vals**2,       lw=2, c=color, marker='s', markersize=markersize)
    ax.set_ylabel(r"$(\tilde v_{Ex}/v_E^Z)^2$")
#    ax.set_ylim([1e-1, 1e1])

#    for ax in axs:
#        ax.axvline(tprim_crit_approx, c=color, alpha=0.25, lw=8)

# Beautify and save plot
for ax in axs:
    if substract_lin:
        ax.set_xlabel(r"$R/L_T-(R/L_T)_\mathrm{lin}$")
        ax.set_xlim(xmin=0)#.2)
    else:
        ax.set_xlabel(r"$R/L_T$")
    ax.grid(True, alpha=0.5)
    ax.legend(fontsize=22)

    if xlim is not None:
        ax.set_xlim(xlim)

plt.tight_layout()
fig.savefig("fig_param_scan_Dimits"+figname_add+".pdf")
