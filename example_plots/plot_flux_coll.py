import numpy as np
import os
import scipy.special as specialfunc
import json
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})

import stellaDiagnostics as sD

def get_aLT_lin_analytic(rhoc, q, shat, eps=1, tau=1):
    return (1+tau)*(1.33 + 1.91*shat/q) * (1 - 1.5*eps*rhoc)*eps

dt_avg = 200

nu_shift = 1e-7

# Setup
code  = "stella"
filename = "CBC"
colors = ["k", "orange", "crimson", "forestgreen", "mediumblue", "purple", "c", "pink", "0.5", "yellow"]

tprim_vals = [3.85, 4.2, 4.9, 5.6, 5.95, 6.3, 7.35, 8.4]
colors_tprim = sns.color_palette("rocket", len(tprim_vals))

dirs_nu = []
vals_nu = []

dirs_nu.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/")
vals_nu.append(nu_shift)
#vals_nu.append(0)
dirs_nu.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-6/")
vals_nu.append(1e-6)
dirs_nu.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-5/")
vals_nu.append(1e-5)
dirs_nu.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-5/")
vals_nu.append(3e-5)
dirs_nu.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/")
vals_nu.append(1e-4)
dirs_nu.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-4/")
vals_nu.append(3e-4)
dirs_nu.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/")
vals_nu.append(1e-3)
dirs_nu.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-3/")
vals_nu.append(3e-3)
dirs_nu.append("2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.01/")
vals_nu.append(1e-2)

assert(len(vals_nu)==len(dirs_nu))

# Plot for Qflx(t)
fig, axs = plt.subplots(nrows=5,figsize=(9,25))
plt.figtext(0.5,0.99,os.path.abspath('.'),ha='center',va='top',fontsize=6, usetex=False)

for i_tprim, tprim in enumerate(tprim_vals):

    qflx_avg_nu  = []
    qflx_std_nu  = []
    vE_RH_avg_nu = []
    gammaE_avg_nu    = []
    vE_avg_nu    = []
#    vE_RH_std_nu = []
    upar_avg_nu  = []
    nu_plot      = []

    for i_nu, nu in enumerate(vals_nu):

        dirname = dirs_nu[i_nu]+"/"+"run_tprim-%.4f" % (tprim)

        try:
            ncol=1
            diagObj  = sD.stellaDiagnostics(dirname +"/"+ filename, code=code)
        
            _, vflx, qflx, time = diagObj.get_fluxes_over_time(norm=False)

            qflx = qflx[time > time[-1]-dt_avg]
            time = time[time > time[-1]-dt_avg]

            qflx_avg_nu.append(np.sum(qflx*np.gradient(time))/np.sum(np.gradient(time)))
            qflx_std_nu.append(np.std(qflx))
            nu_plot.append(nu)

        except Exception as e:  
            print(e)

            print("TRYING TO LOAD HEAT FLUX FROM DATA_DIMITS.JSON")
            try:
                f = open(dirname+'/data_Dimits.json')
                data_dict = json.load(f)
                qflx_avg  = data_dict['qflx_avg']
                qflx_std  = data_dict['qflx_std']
                qflx_avg_nu.append(qflx_avg)
                qflx_std_nu.append(qflx_std)
                nu_plot.append(nu)
            except Exception as e:
                print(e)
                continue

        try:
            f = open(dirname+'/data_Dimits.json')
            data_dict = json.load(f)
            vE_RH_avg = data_dict['vE_RH_avg']
            vE_avg = data_dict['vE_avg']
            gammaE_avg = data_dict['gammaE_avg']
#            vE_RH_std = data_dict['vE_RH_std']
            upar_avg  = data_dict['upar_avg']
            vE_RH_avg_nu.append(vE_RH_avg)
            vE_avg_nu.append(vE_avg)
            gammaE_avg_nu.append(gammaE_avg)
#            vE_RH_std_nu.append(vE_RH_std)
            upar_avg_nu.append(upar_avg)
        except Exception as e:
            print(e)
            vE_RH_avg_nu.append(None)
            vE_avg_nu.append(None)
            gammaE_avg_nu.append(None)
#            vE_RH_std_nu.append(None)
            upar_avg_nu.append(None)

    # tprim-tprim_Jenko
    Delta_tprim = tprim - get_aLT_lin_analytic(rhoc=0.18, q=1.4, shat=0.8)
#    print(Delta_tprim)
#    print(qflx_avg_nu)

    # Sort
    idx_sort = np.argsort(nu_plot)
    nu_plot     = np.array(nu_plot)[    idx_sort]
    qflx_avg_nu  = np.array(qflx_avg_nu)[idx_sort]
    qflx_std_nu  = np.array(qflx_std_nu)[idx_sort]
    vE_RH_avg_nu = np.array(vE_RH_avg_nu)[idx_sort]
    vE_avg_nu = np.array(vE_avg_nu)[idx_sort]
    gammaE_avg_nu = np.array(gammaE_avg_nu)[idx_sort]
#    vE_RH_std_nu = np.array(vE_RH_std_nu)[idx_sort]
    upar_avg_nu  = np.array(upar_avg_nu)[idx_sort]

#    ax.fill_between(nu_plot, (qflx_avg_nu[0]-0.5*qflx_std_nu[0])/Delta_tprim, (qflx_avg_nu[0]+0.5*qflx_std_nu[0])/Delta_tprim, color=colors_tprim[i_tprim], alpha=0.5)

    ax = axs[0]
    ax.errorbar(nu_plot[0],  qflx_avg_nu[0] /Delta_tprim, qflx_std_nu[0] /Delta_tprim, c=colors_tprim[i_tprim], marker='s', markersize=20)
    ax.errorbar(nu_plot[1:], qflx_avg_nu[1:]/Delta_tprim, qflx_std_nu[1:]/Delta_tprim, c=colors_tprim[i_tprim], marker='o', label=r"$R/L_T = %.2f$" % (tprim))

    try:
        ax = axs[1]
        #ax.plot(nu_plot[0],  vE_avg_nu[0] , c=colors_tprim[i_tprim], marker='s', markersize=20)
        #ax.plot(nu_plot[1:], vE_avg_nu[1:], c=colors_tprim[i_tprim], marker='o', label=r"$R/L_T = %.2f$" % (tprim))
        ax.plot(nu_plot[0],  gammaE_avg_nu[0] , c=colors_tprim[i_tprim], marker='s', markersize=20)
        ax.plot(nu_plot[1:], gammaE_avg_nu[1:], c=colors_tprim[i_tprim], marker='o', label=r"$R/L_T = %.2f$" % (tprim))

        ax = axs[2]
        ax.plot(nu_plot[0],  vE_RH_avg_nu[0] , c=colors_tprim[i_tprim], marker='s', markersize=20)
        ax.plot(nu_plot[1:], vE_RH_avg_nu[1:], c=colors_tprim[i_tprim], marker='o', label=r"$R/L_T = %.2f$" % (tprim))

        ax = axs[3]
        ax.plot(nu_plot[0],  upar_avg_nu[0] ,  c=colors_tprim[i_tprim], marker='s', markersize=20)
        ax.plot(nu_plot[1:], upar_avg_nu[1:],  c=colors_tprim[i_tprim], marker='o', label=r"$R/L_T = %.2f$" % (tprim))

        ax = axs[4]
        ax.plot(nu_plot[0],  upar_avg_nu[0] /vE_RH_avg_nu[0],  c=colors_tprim[i_tprim], marker='s', markersize=20)
        ax.plot(nu_plot[1:], upar_avg_nu[1:]/vE_RH_avg_nu[1:], c=colors_tprim[i_tprim], marker='o', label=r"$R/L_T = %.2f$" % (tprim))
    except:
        continue

nu_th = np.linspace(1e-4, 1e-2, 100)
axs[1].plot(nu_th, (nu_th/1e-4)**(-1/2), ls='--', c='0.5', label=r"$\propto \nu_{ii}^{-1/2}$")
axs[2].plot(nu_th, (nu_th/1e-4)**(-1/2), ls='--', c='0.5', label=r"$\propto \nu_{ii}^{-1/2}$")
axs[3].plot(nu_th, (nu_th/1e-4)**(-1/2), ls='--', c='0.5', label=r"$\propto \nu_{ii}^{-1/2}$")

#axs[0].set_yscale('log')
for ax in axs:
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    ax.set_xlabel(r"$\nu_{ii}R/v_{Ti}$")
    ax.set_xlim(xmin=nu_shift)
    ax.grid()

axs[0].set_ylabel(r"$(Q/Q_\mathrm{gB})/(R/L_T - (R/L_T)_\mathrm{Jenko})$")
axs[1].set_ylabel(r"$\langle \gamma_{E}^2 \rangle^{1/2}$")
#axs[1].set_ylabel(r"$\langle v_{E}^2 \rangle^{1/2}$")
axs[2].set_ylabel(r"$\langle v_{E,\mathrm{RH}}^2 \rangle^{1/2}$")
axs[3].set_ylabel(r"$\langle u_\parallel^2 \rangle^{1/2}$")
axs[4].set_ylabel(r"$\langle u_\parallel^2 \rangle^{1/2} / \langle v_{E,\mathrm{RH}}^2 \rangle^{1/2}$")

plt.tight_layout()
plt.savefig("fig_Q_nu_dtavg-%i.pdf" % (dt_avg))
plt.close()
