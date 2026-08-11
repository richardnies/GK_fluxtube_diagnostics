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
fontsize_legend=16

import stellaDiagnostics as sD

filenames_list = []
labels_list    = []
colors_list    = []
ls_list        = []
marker_list    = []
tprim_list     = []
qinp_list     = []
codes_list     = []

add_str = ""

aspect_ratio = 1

############## TPRIM SCANS ###############
dirnames = ["2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"]
labels = [r"$\nu_{ii}R/v_{Ti}=0$"]
filenames = ["CBC", "CBC", "CBC"]
tprim_vals = np.array([3.85, 4.9, 6.4])
ls = ["--", "-", ":"]
markers = ["o", "x", "s"]
colors_tprim = sns.color_palette("rocket", len(tprim_vals))

for i_dir, dirname in enumerate(dirnames):
    for i_tprim, tprim_val in enumerate(tprim_vals):
        filename = dirname + "/run_tprim_val-%.4f/" % (tprim_val) + filenames[i_dir]
        if exists(filename+".nc") or exists(filename+".out.nc"):
            filenames_list.append(filename)
            if i_dir == 0 and (i_tprim == 0 or i_tprim == len(tprim_vals)-1):
                label = r"$\kappa = %.1f$" % (tprim_val*aspect_ratio)
            else:
                label = None
            labels_list.append(label)
            colors_list.append(colors_tprim[i_tprim])
            tprim_list.append(tprim_val)
            qinp_list.append(1.4)
            ls_list.append(ls[i_dir])
            marker_list.append(markers[i_dir])

fig, ax = plt.subplots(figsize=(4.5,4.5))

# Linear fit
c='green'
qkappa2_plot = np.linspace(6e1, 4e3, 100)
#qkappa2_plot = np.linspace(qkappa2_data.min()/1.2, qkappa2_data.max()*1.2, 100)
ax.plot(qkappa2_plot, qkappa2_plot/2.8, c=c, alpha=0.75)
ax.text(1.3e3, 1.2e3, r"$\sim q\kappa^2$", c=c, fontsize=fontsize_legend, alpha=0.75)

# Extract and plot data
kxmin =0.3
kxmax =1e4

fac_rescale_phys = True

if fac_rescale_phys:
    fac_rescale = 2*2.8**2
else:
    fac_rescale = 5e1

qkappa2_data = []
Ezonal_data  = []
for i_file in range(len(filenames_list)):

    #stellaObj = sD.stellaDiagnostics(filenames_list[i_file], code=codes_list[i_file])
    #dxphiZ_kx_ky, kx, _, _ = stellaObj.get_quantity_kx_ky(quantity="phi", kx_order=1, only_zonal=True)
    #phi2k = np.abs(dxphiZ_kx_ky[:,0])**2

    Ezonal_datafile = filenames_list[i_file]+"_kx_zonal.dat" #(1-Gamma0)|phiZ_kx|^2
    phi2k, kx = np.loadtxt(Ezonal_datafile)
    Ezonal_stddev_datafile = filenames_list[i_file]+"_kx_zonal_stddev.dat"
    phi2k_stddev, kx = np.loadtxt(Ezonal_stddev_datafile)

    idx_min =  np.argmin(np.abs(kx-kxmin))
    idx_max =  np.argmin(np.abs(kx-kxmax))
    Ezonal = np.trapz(y=phi2k[idx_min:idx_max], x=kx[idx_min:idx_max]) * fac_rescale
    Ezonal_stddev = np.trapz(y=phi2k_stddev[idx_min:idx_max], x=kx[idx_min:idx_max]) * fac_rescale

    qkappa2 = qinp_list[i_file]*(tprim_list[i_file]*2.8)**2

#    ax.scatter(qkappa2, Ezonal, marker=marker_list[i_file], color=colors_list[i_file])
    ax.errorbar(qkappa2, Ezonal, yerr=Ezonal_stddev, marker=marker_list[i_file], color=colors_list[i_file], label=labels_list[i_file], ls='None')

    qkappa2_data.append(qkappa2)
    Ezonal_data.append(Ezonal)

qkappa2_data = np.array(qkappa2_data)
Ezonal_data  = np.array(Ezonal_data)

idx_sort = np.argsort(qkappa2_data)
qkappa2_data = qkappa2_data[idx_sort]
Ezonal_data  = Ezonal_data[idx_sort]

#ax.plot(qkappa2_plot, 1.5e1*(qkappa2_plot/1e2), c='g')
#ax.plot(qkappa2_plot, 1e1*(qkappa2_plot/1e2)**1.5, c='y')
#ax.plot(qkappa2_plot, 1e1*(qkappa2_plot/1e2)**2, c='r')

#from scipy.stats import linregress
#linfit = linregress(qkappa2_data,Ezonal_data)
#ax.plot(qkappa2_plot, linfit.intercept+qkappa2_plot*linfit.slope, c='k', ls='--', label="Linear fit")
#print("Intercept: %e" % (linfit.intercept))
#print("Slope:     %e" % (linfit.slope))
#print("R2-value:  %e" % (linfit.rvalue**2))

ax.set_xlabel(r"$q \kappa^2$")
if kxmax > 1e3:
    if fac_rescale_phys:
        ax.set_ylabel(r"$ (R/\rho_i)^2 (E^\mathrm{ZF}/T_i)_{|k_x \rho_i| > %.1f}$" % (kxmin))
        #ax.set_ylabel(r"$ (R/\rho_i)^2 E^\mathrm{ZF}_{|k_x \rho_i| > %.1f}/T_i$" % (kxmin))
    else:
        ax.set_ylabel(r"$E^\mathrm{ZF}_{|k_x \rho_i| > %.1f}$ (arb. units)" % (kxmin))
    #ax.set_ylabel(r"$(v_E^Z)^2\;$ for $|k_x \rho_i| > %.1f$" % (kxmin))
else:
    ax.set_ylabel(r"$(v_E^Z)^2\; (|k_x \rho_i| \in [%.1f,%.1f])$" % (kxmin, kxmax))
#ax.set_ylabel(r"$E^Z_{|k_x \rho_i| \in [%.1f,%.1f]}$ (arb. units)" % (kxmin, kxmax))
#ax.set_ylabel(r"$E^Z (%.1f \leq |k_x \rho_i| \leq %.1f)$" % (kxmin, kxmax))
#ax.set_ylabel(r"$E^Z (%.1f \leq k_x \rho_i \leq %.1f)$ (a.u.)" % (kxmin, kxmax))
ax.grid()
ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_xlim([qkappa2_data.min()/2, qkappa2_data.max()*2])
#ax.set_ylim(ymax=qkappa2_data.max()*1.1, ymin=qkappa2_data.min()/4)
#ax.set_ylim([qkappa2_data.min()/2, qkappa2_data.max()*2])
ax.legend(fontsize=fontsize_legend, handlelength=0.3, handletextpad=0.4, borderaxespad=0.4, labelspacing=0.4, borderpad=0.4)
ax.set_ylim(ymin=4)

plt.tight_layout()
fig.savefig("fig_phiZ_TS_qkappa2.pdf")

#ax.set_xscale('linear')
#ax.set_yscale('linear')
#fig.savefig("fig_phiZ_TS_qkappa2_lin.pdf")
#    
