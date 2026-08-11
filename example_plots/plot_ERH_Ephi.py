import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import scipy.special as specialfunc
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
base_labels = [r"$\nu^*=0$", r"$\nu^*=10^{-4}$", r"$\nu^*=10^{-3}$"]
figname_add = "_nu_scan_red"

aLT_lin_vals = get_aLT_lin_analytic( rhoc = np.array([0.18,0.18,0.18,0.18,0.18, 0.18, 0.18, 0.18, 0.18, 0.18]),
                                     q    = np.array([1.4, 1.4, 1.4, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4]),
                                     shat = np.array([0.8, 0.8, 0.8, 0.8,  0.8,  0.8,  0.8,  0.8,  0.8,  0.8]) )

#base_colors = sns.color_palette("hls", len(base_dirs))
base_colors = sns.color_palette("rocket", len(base_dirs))

#aLT_lin_vals = np.zeros_like(aLT_lin_vals)
#aLT_lin_vals = np.array([1.9, 1.4, 1.1, 0.7, 0.8, 1.4, 1.4, 1.4])

markersize = 10

nrows=5
fig, axs = plt.subplots(nrows=nrows, figsize=(9, 5*nrows))#, sharex=True)

dt_avg = 800
figname_add += "_dtavg-%i" % (dt_avg)

chihat_norm = None

for i_base, base_dir in enumerate(base_dirs):

    label = base_labels[i_base]
    color = base_colors[i_base]

    dirnames = sorted(glob(base_dir+"/run_tprim*00/"))
    #dirnames = sorted(glob(base_dir+"/run_tprim*00/"))

    ndirs = len(dirnames)

    # Load data
    tprim_vals             = np.zeros(ndirs)
    qinp_vals              = np.zeros(ndirs)
    eps_vals               = np.zeros(ndirs)
    qflx_avg_vals          = np.zeros(ndirs)
    gammaE_avg_vals        = np.zeros(ndirs)
    gammaE_std_vals        = np.zeros(ndirs)
    ERH_vals               = np.zeros(ndirs)
    Ephi_vals              = np.zeros(ndirs)

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
            gammaE_avg_vals[i_dir]        = data_dict['gammaE_avg']
            gammaE_std_vals[i_dir]        = data_dict['gammaE_std']

            # Load ERH and Ephi
            filename = dirname + "/CBC"
            diagObj = sD.stellaDiagnostics(filename)
    
            time_min = diagObj.ncdata.variables['t'][-1]-dt_avg
            time_max = diagObj.ncdata.variables['t'][-1]

            E_RH_t_kx, RH_time, RH_kx = diagObj.get_E_RH_t_kx(time_min=time_min, time_max=time_max)
            ERH_vals[i_dir]  = np.sum(E_RH_t_kx*np.gradient(RH_time)[:,None])/np.sum(np.gradient(RH_time))

            phi2_t_kx_ky, time, kx, ky = diagObj.read_phi2_spectra(time_min=time_min, time_max=time_max)
            Gamma0 = specialfunc.iv(0, kx**2/2) * np.exp(-kx**2/2)
            Ephi_vals[i_dir] = 0.5*np.sum((1-Gamma0)[None,:]*phi2_t_kx_ky[:,:,0]*np.gradient(time)[:,None])/np.sum(np.gradient(time))

        except Exception as e:
            print(e)
            print("Could not load file for " + dirname)
    
            tprim_vals[i_dir]             = np.nan 
            qinp_vals[i_dir]              = np.nan 
            eps_vals[i_dir]               = np.nan 
            qflx_avg_vals[i_dir]          = np.nan 
            gammaE_avg_vals[i_dir]        = np.nan 
            gammaE_std_vals[i_dir]        = np.nan 
            ERH_vals[i_dir]               = np.nan 
            Ephi_vals[i_dir]              = np.nan 

    tprim_vals += -aLT_lin_vals[i_base]

    ##### Plot
    # ERH
    i=0
    ax = axs[i]
    ax.plot(tprim_vals, ERH_vals, c=color, marker='o', markersize=markersize, label=label)
    ax.set_ylabel(r"$E_\mathrm{RH}$")
    ax.set_yscale('log')

    # Ephi
    i=1
    ax = axs[i]
    ax.plot(tprim_vals, Ephi_vals, c=color, marker='o', markersize=markersize, label=label)
    ax.set_ylabel(r"$E_\varphi$")
    ax.set_yscale('log')

    # ERH/Ephi
    i=2
    ax = axs[i]
    ax.plot(tprim_vals, ERH_vals/Ephi_vals, c=color, marker='o', markersize=markersize, label=label)
    ax.set_ylabel(r"$E_\mathrm{RH}/E_\varphi$")

    # chihat/chihat(large R/LT)
    i=3
    ax = axs[i]
    chihat = qflx_avg_vals/tprim_vals
    if chihat_norm is None:
        chihat_norm = chihat[-1]
    ax.plot(tprim_vals, chihat/chihat_norm, c=color, marker='o', markersize=markersize, label=label)
    ax.set_ylabel(r"$\hat \chi_i / \hat \chi_i(R/L_T)_\mathrm{max}$")

    # ERH/Ephi
    i=4
    ax = axs[i]
    ax.errorbar(tprim_vals, gammaE_avg_vals, gammaE_std_vals, c=color, marker='o', markersize=markersize, label=label)
    ax.set_ylabel(r"$\langle \gamma_E^2 \rangle^{1/2}$")

    for ax in axs:
        ax.set_xlabel(r"$a/L_T-(a/L_T)_\mathrm{lin}$")

# Beautify and save plot

#axs[-1].set_xlabel(r"$a/L_{Ti}-a/L_{Ti}^\mathrm{Jenko}$")

for ax in axs:
    ax.grid(True)
    ax.legend(fontsize=22)
    ax.set_xlim(xmin=0)#.2)

plt.tight_layout()
fig.savefig("fig_ERH_Ephi_Dimits"+figname_add+".pdf")
