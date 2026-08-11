import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12, 
    "axes.titlepad": 15,
})

# Stella diagnostics directory
dir_stella_diagnostics = '/home/rnies/stella_diagnostics'
sys.path.append(dir_stella_diagnostics)
import stellaDiagnostics as sD

# Setup
filename_base = "/precise_QA_NL"

base_dirs = ["./"]
tprim_vals   = np.array([2,3])
zeta_center_vals = [0, 0.5] # Multiples of 2*pi/Nfp

import seaborn as sns
colors = sns.color_palette("rocket", len(tprim_vals))
markers = ["o", "x"]
ls_base_dirs = ["-", "--"]
colors_tube = ["r", "g"]

dirnames         = []
labels           = []
color_list       = []
marker_list      = []
ls_list          = []
tprims_list      = []
colors_tube_list = []

for i_base_dir, base_dir in enumerate(base_dirs):
    for i_tprim, tprim_val in enumerate(tprim_vals):
        for i_zetactr, zeta_center_val in enumerate(zeta_center_vals):
                dirnames.append(base_dir+"run_tprim-%.4f_zeta_center-%.4f" % (tprim_val, zeta_center_val) )
                if i_base_dir == 0 and i_zetactr == 0:
                    labels.append(r"$a/L_T = %.1f$" % (tprim_val))
                else:
                    labels.append(None)
                ls_list.append(ls_base_dirs[i_base_dir])
                color_list.append(colors[i_tprim])
                marker_list.append(markers[i_base_dir])
                tprims_list.append(tprim_val)
                colors_tube_list.append(colors_tube[i_zetactr])

take_last = False
#take_last = True
time_avg  = 350
kx_rhoi_O_avg = np.zeros(len(dirnames))

# Load data
plt.figure(figsize=(5,3))
for i_dir, dirname in enumerate(dirnames):
    filename = dirname+filename_base
    diagObj = sD.stellaDiagnostics(filename)

    kx_rhoi_O, time = diagObj.read_avg_kx_rhoi()

    plt.plot(time, kx_rhoi_O, label=labels[i_dir], ls=ls_list[i_dir], c=color_list[i_dir])

    if take_last:
        kx_rhoi_O_avg[i_dir] = kx_rhoi_O[-1]
    else:
        if time[-1] > time_avg:
            kx_rhoi_O_avg[i_dir] = np.average(kx_rhoi_O[time>time_avg])
        else:
            kx_rhoi_O_avg[i_dir] = np.nan

plt.grid()
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\langle (k_x \rho_i)^{-1} \rangle^{-1}$")
plt.xlim(xmin=0)
plt.ylim(ymin=0)
#plt.ylim(ymin=0)
plt.tight_layout()
plt.savefig("fig_kxrhoi_outer_time_QA.eps")

plt.close()
plt.figure(figsize=(5,3))
plt.loglog(tprim_vals, kx_rhoi_O_avg[0]*tprim_vals[0]/tprim_vals, ls=':', c='k', label=r"$(q \kappa)^{-1}$")
plt.loglog(tprim_vals, kx_rhoi_O_avg[0]*np.sqrt(tprim_vals[0]/tprim_vals), ls=':', c='0.5', label=r"$(q \kappa)^{-1/2}$")
for i in range(len(dirnames)):
    plt.scatter(tprims_list[i], kx_rhoi_O_avg[i], marker=marker_list[i], c=colors_tube_list[i])
plt.legend()
plt.xlabel(r"$a/L_T$")
plt.ylabel(r"$\langle (k_x \rho_i)^{-1} \rangle^{-1}$")
plt.tight_layout()
plt.savefig("fig_kxrhoi_outer_QA.eps")
