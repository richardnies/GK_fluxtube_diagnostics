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

vnew_vals  = [0.0001, 0.001, 0.01]
tprim_vals = [4.2, 4.9, 5.6, 5.95, 6.3, 8.4]

eps = 0.18

fig, ax = plt.subplots(figsize=(9,6))

basedir = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-"

for vnew in vnew_vals:
    if vnew == 0.0001:
        vnew_dir = "0.0001"
        vnew_str = r"$10^{-4}$"
    elif vnew == 0.001:
        vnew_dir = "0.001"
        vnew_str = r"$10^{-3}$"
    elif vnew == 0.01:
        vnew_dir = "0.01"
        vnew_str = r"$10^{-2}$"

    for tprim in tprim_vals:
        dirname = basedir + vnew_dir + "/" + "run_tprim-%.4f" % (tprim)

        try:
            E_RH_mean_kx           = np.loadtxt(dirname+"/data_ERH_mean_kx.dat")
            P_RH_coll_even_mean_kx = np.loadtxt(dirname+"/data_P_RH_coll_even_mean_kx.dat")
            P_RH_coll_odd_mean_kx  = np.loadtxt(dirname+"/data_P_RH_coll_odd_mean_kx.dat")

            P_RH_coll_mean_kx = P_RH_coll_even_mean_kx + P_RH_coll_odd_mean_kx

            if np.sum(np.abs(P_RH_coll_mean_kx))<1e-14:
                continue

            P_RH_coll_mean_kx_norm = P_RH_coll_mean_kx / (vnew*E_RH_mean_kx) * eps**2

            StellaObj = sD.stellaDiagnostics(dirname+"/CBC")
            kx_all = StellaObj.ncdata['kx'][:]

            ax.semilogx(kx_all[kx_all>0], -P_RH_coll_mean_kx_norm[kx_all>0], marker='.', label=r"$\nu_{ii} R/v_{Ti} = $" + vnew_str + r"$, R/L_T = %.2f$" % (tprim))

        except Exception as e:
            print(e)

ax.set_xlabel(r"$k_x \rho_i$")
ax.set_ylabel(r"$-P_\mathrm{RH}^C / (\epsilon^{-2} \nu_{ii} E_\mathrm{RH})$")

ax.grid(True)
ax.legend(fontsize=14)
ax.set_ylim(ymin=0)

plt.tight_layout()
fig.savefig("fig_P_RH_C_normalised.pdf")
