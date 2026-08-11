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

import stellaDiagnostics as sD

# Default values
time_min = 500
time_idx_step = 2
time_max = 1e6

# Load data
filename_base = "/CBC"

#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_no-upwind_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_long_tprim-4.9000"

#### eps = 0.045
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0/run_tprim-6.3000"

#### eps = 0.36
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0/run_tprim-6.3000"
#time_min = 1000

#### shat = 0.32
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.2500/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.5000/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.7500/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-2.0000/"
#time_idx_step = 1

#### shat = 0.16
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-05-05_qinp-1.4_shat-0.16_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.7500"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-05-05_qinp-1.4_shat-0.16_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-2.0000"
#time_idx_step = 3

#### eps = 0.342
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-09_scan_qinp-1.4_shat-0.8_rmaj-2.778_rhoc-0.95_fprim-0.8_vnew-0/run_tprim-2.5000"
#time_min = 4000
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-09_scan_qinp-1.4_shat-0.8_rmaj-2.778_rhoc-0.95_fprim-0.8_vnew-0/run_tprim-2.2500"
#time_min = 1500
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-09_scan_qinp-1.4_shat-0.8_rmaj-2.778_rhoc-0.95_fprim-0.8_vnew-0/run_tprim-2.0000"
#time_min = 1000

#### qinp = 1.0
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-2.0000/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-2.2500/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-2.5000/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-3.0000/"
#time_idx_step = 3

### HR
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-01-26_tprim_scan_adiab_el_HR/run_tprim-2.5000_higher_nx/"

#### qinp = 2.8
#dirname = "2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000"

#### restarts
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-01-20_tprim_scan_adiab_el/run_tprim-2.5000_restart_scaleZ-10_kmax-0.1"
#time_min = 1.0807E+04; time_max = time_min + 500
#time_idx_step = 3

#### vnew = 0
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-3.8500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.2000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-5.6000"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-5.9500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-8.4000"

#### vnew = 1e-4
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-3.8500"
#time_min = 800
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.2000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.2000_HR"
#time_min = 500
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.2000_restart_linear"
time_min = 2.2241E+03
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.9000"
#time_min = 0
#time_max = 1000
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-5.6000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-5.9500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-6.3000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-8.4000"
#time_idx_step = 5

#### vnew = 1e-3
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-3.8500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-4.2000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-5.6000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-5.9500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-6.3000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-8.4000"

overplot_time_trace = True
overplot_quantity_idx = 0

#dirname = "2026-01-20_tprim_scan_adiab_el/run_tprim-2.0000"
#time_min  = 3800
#time_max  = 4500
#time_min  = 4500
#time_max  = 4700

dirname_string = dirname.replace("/","_")
filename = dirname+filename_base
diagObj = sD.stellaDiagnostics(filename)

# Setup

time_idx_min = diagObj.get_time_idx(time_min)
time_idx_max = diagObj.get_time_idx(time_max)
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)

rerun_all = True
#rerun_all = False

# Quantities
quantities = ["phi",      "RH_phi",                 "dyphi-upar",   "dyphi-T",     "dyphi2",      "P_RH_tot",        "P_RH_NL",       "P_RH_even",       "P_RH_odd",       "P_RH_coll",  "upar",  "temperature"]
kx_orders  = [    1,             1,                            0,           0,            0,               0,                0,                 0,                0,                 0,  0,       1]
mults      = [   -1,            -1,                            1,           1,            1,               1,                1,                 1,                1,                 1,  1,       1]
labels     = [r"$v_E^Z$", r"$v_E^{\mathrm{RH},Z}$",r"$\Pi_\parallel$", r"$Q$", r"$v_{Ex}^2$", r"$P_\mathrm{RH}$", r"$P_\varphi$", r"$P_\varphi^+$", r"$P_\varphi^-$",  r"$P_{\mathrm{RH}}^C$", r"$u_\parallel^Z$", r"$\partial_x T^Z$"]
datanames  = ["vE",        "vE_RH",                "Pi_parallel",         "Q",       "vEx2",        "P_RH",          "P_RH_phi",       "P_phi_even",    "P_phi_odd",          "P_RH_coll", "upar",  "gradTZ"]

#quantities = [ "dyphi-upar",      "upar"             ,  "density"          ]
#kx_orders  = [            0,           0             ,       1             ]
#mults      = [            1,           1             ,       1             ]
#labels     = [r"$\Pi_\parallel$", r"$u_\parallel^Z$" ,  r"$\partial_x n^Z$"]
#datanames  = ["Pi_parallel",     "upar"              , "gradnZ"            ]

# Directory with images
img_dir = "fig_" + dirname_string + "_quantities_x_zed"

import os
os.system("mkdir -p " + img_dir)
if rerun_all:
    os.system('rm -rf ' + img_dir + '/*')
else:
    os.system('rm -rf ' + img_dir + '/video*')

# Prepare plot that will show time-averaged quantities
fig_tavg, axs_tavg = plt.subplots(ncols=len(quantities), figsize=(8*len(quantities), 8), sharey=True)
nx   = len(diagObj.ncdata.variables['kx'])
nzed = len(diagObj.ncdata.variables['zed'])
avg_quantities_zed_x = np.zeros((len(quantities), nzed, nx))

dt_sum = 0

time = diagObj.ncdata.variables['t'][:]

# Plot for each timestep
for i_time_idx, time_idx_val in enumerate(time_idx_vals):
    print("Plotting figure %i/%i..." % (i_time_idx+1, len(time_idx_vals)), end="\r")

    fig_filename = img_dir+"/fig_t-%.3i.png" % (i_time_idx)

    if not rerun_all and exists(fig_filename):
        continue

    plt.close()
    try:

        fig, axs = plt.subplots(ncols=len(quantities), figsize=(8*len(quantities), 8), sharey=True)
        for i_quantity, quantity in enumerate(quantities):

            ax = axs[i_quantity]

            kx_order = kx_orders[i_quantity]
            label    = labels[   i_quantity]
            mult     = mults[    i_quantity]

            _, _, im, x, zed, f_zed_x, zed_avg_x, vmin, vmax = diagObj.plot_quantity_x_zed(quantity=quantity, time_idx=time_idx_val, kx_order=kx_order, fig=fig, ax=ax, mult_fac=mult, vmin="symm", cmap="coolwarm", only_zonal=True)
            ax.axvline(0, c='k', alpha=0.75)
            cbar = fig.colorbar(im, ax=ax)
            ax.set_title(label)

            # For time-average plots
            if i_time_idx > 0:
                if i_quantity == 0:
                    dt = time[time_idx_val]-time[time_idx_val-1]
                    dt_sum += dt
                avg_quantities_zed_x[i_quantity,:,:] += f_zed_x*dt

        plt.tight_layout()
        plt.savefig(fig_filename)

    except Exception as e:
        print(str(e)+"\n")
        traceback.print_exc()
        break

try:
    # Make movie using ffmpeg
    os.system('ffmpeg -r 15 -i ' + img_dir + '/fig_t-%03d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p ' + img_dir + '/video_quantities_x_zed'+dirname_string+'.mp4')
except:
    print("COULD NOT MAKE MOVIE!")

# Plot of time averages
avg_quantities_zed_x /= dt_sum

for i_quantity, quantity in enumerate(quantities):
    label    = labels[i_quantity]
    ax = axs_tavg[i_quantity]

    vmax = np.abs(avg_quantities_zed_x[i_quantity]).max()

    if rerun_all:
        _, _, im, x, zed, f_zed_x, zed_avg_x, vmin, vmax = diagObj.plot_quantity_x_zed(quantity=avg_quantities_zed_x[i_quantity], fig=fig_tavg, ax=ax, vmin="symm", cmap="coolwarm")
        ax.axvline(0, c='k', alpha=0.75)
        cbar = fig.colorbar(im, ax=ax)
        ax.set_title(label)

    if overplot_quantity_idx is not None:
        dl_over_B_avg = diagObj.dl_over_B_avg()
        overplot_x = np.sum(avg_quantities_zed_x[overplot_quantity_idx]*dl_over_B_avg[:,None], axis=0)
        norm = (zed[-1]/2)/np.nanmax(np.abs(overplot_x))
        rot  = transforms.Affine2D().rotate_deg(90)
        base = ax.transData
        ax.plot(x, -norm*overplot_x, c='k', ls=':', alpha=0.75, label=labels[overplot_quantity_idx], transform=rot+base)
        ax.legend(loc='upper left', fontsize=20)

    np.savetxt(dirname+"/data_zed_x_"+datanames[i_quantity]+".dat", avg_quantities_zed_x[i_quantity])
np.savetxt(dirname+"/data_zed_x_x.dat", x)
np.savetxt(dirname+"/data_zed_x_zed.dat", zed)

fig_tavg.tight_layout()
fig_tavg.savefig("fig_quantities_x_zed_tavg_"+dirname_string+".pdf")

    
