import numpy as np
from os.path import exists
import traceback
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})

import stellaDiagnostics as sD

vmin = "symm"
vmax = None

# Load data
filename_base = "/CBC"

dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000_continue"
time_min  = 3.9876E+03

dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-6.3000_continue"
time_min  = 3.4462E+03

dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-5.6000"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-5.6000"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-7.3500"
#vmax = 3e2
#vmin = vmax/1e4

#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-4.2000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.2000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.2000"

#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-8.4000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-8.4000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-8.4000"

dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.2000_restart_linear"
time_min = 2.2241E+03

#time_min  = 0

kx_mins = [   0,   0.20]
kx_maxs = [0.20,      2]

Nrows = len(kx_mins)

#dt_avg = 25
dt_avg = None

dirname_string = dirname.replace("/","_")

filename = dirname+filename_base
diagObj = sD.stellaDiagnostics(filename)

# Setup

time_max  = 1e10
#time_max  = 1700

time_idx_min = diagObj.get_time_idx(time_min)
time_idx_max = diagObj.get_time_idx(time_max)
time_idx_step = 5
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)

rerun_all = True
#rerun_all = False

# Directory with images
img_dir = "fig_" + dirname_string + "_gvmus_Z-NZ_kxs"
if dt_avg is not None:
    img_dir += "_dtavg-%i" % (dt_avg)

import os
os.system("mkdir -p " + img_dir)
if rerun_all:
    os.system('rm -rf ' + img_dir + '/*')
else:
    os.system('rm -rf ' + img_dir + '/video*')

# Plot for each timestep
for i_time_idx, time_idx_val in enumerate(time_idx_vals):
    print("Plotting figure %i/%i..." % (i_time_idx+1, len(time_idx_vals)), end="\r")

    fig_filename = img_dir+"/fig_t-%.3i.png" % (i_time_idx)

    if not rerun_all and exists(fig_filename):
        continue

    plt.close()
    fig, axs = plt.subplots(nrows=Nrows,ncols=2, figsize=(24,9*Nrows))

    if Nrows == 1:
        axs = [axs]

    for irow in range(Nrows):

        kx_min = kx_mins[irow]
        kx_max = kx_maxs[irow]

        label_kx = r"$(%.2f < |k_x| < %.2f)$" % (kx_min, kx_max)

        try:
            ax = axs[irow, 0]
            _, _, im = diagObj.plot_contour_gvmu_vpa(time_idx=time_idx_val, vmin=vmin, vmax=vmax, logarithmic=True, nozonal=True, zonal=False, fig=fig, ax=ax, kx_min=kx_min, kx_max=kx_max, dt_avg=dt_avg)
            plt.colorbar(im, ax=ax)
            ax.set_title(r"$V^{-1}\int\mathrm{d}^3 r \;g_\mathrm{NZ}^2/F_M$ " + label_kx)

            ax = axs[irow, 1]
            _, _, im = diagObj.plot_contour_gvmu_vpa(time_idx=time_idx_val, vmin=vmin, vmax=vmax, logarithmic=True, nozonal=False, zonal=True, fig=fig, ax=ax, kx_min=kx_min, kx_max=kx_max, dt_avg=dt_avg)
            plt.colorbar(im, ax=ax)
            ax.set_title(r"$V^{-1}\int\mathrm{d}^3 r \;g_\mathrm{Z}^2/F_M$ " + label_kx)

            plt.savefig(fig_filename)
        except Exception as e:
            print("COULD NOT LOAD DATA/GENERATE FIGURE!")
            print(e)
            traceback.print_exc()
            continue
            #break

# Make movie using ffmpeg
#os.system('ffmpeg -r 120 -i ' + img_dir + '/fig_t-%03d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p ' + img_dir + '/video_gvmus_t_kxs'+dirname_string+'.mp4')
os.system('ffmpeg -r 30 -i ' + img_dir + '/fig_t-%03d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p ' + img_dir + '/video_gvmus_t_kxs'+dirname_string+'.mp4')
