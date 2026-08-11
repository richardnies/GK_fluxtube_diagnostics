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

# Load data
filename_base = "/CBC"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_long_tprim-4.9000"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_long_tprim-4.9000"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_long_tprim-4.9000_continue"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_no-upwind_tprim-4.9000"

#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0_higher-vel-res/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0/run_tprim-6.3000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-0.7_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-7.5000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-6.3000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-5/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-4/run_tprim-4.9000"
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000_continue"

dirname_string = dirname.replace("/","_")

filename = dirname+filename_base
diagObj = sD.stellaDiagnostics(filename)

# Setup
#kx_idxs = [0,5,10,25,50,75]
#Nr_kx_vals = len(kx_idxs)
#
#ky_idxs = [1,5,10,20]
#Nr_ky_vals = len(ky_idxs)

time_min  = 3.9876E+03
#time_idx_min  = 200
#time_idx_min  = 750
time_max  = 1e10

time_idx_min = diagObj.get_time_idx(time_min)
time_idx_max = diagObj.get_time_idx(time_max)
time_idx_step = 1
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)

rerun_all = True
#rerun_all = False

# Directory with images
img_dir = "fig_" + dirname_string + "_gvmus_Z-NZ"

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
    fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(24,9))

    try:
        ax = axs[0]
        _, _, im = diagObj.plot_contour_gvmu_vpa(time_idx=time_idx_val, logarithmic=True, vmin="symm", nozonal=True, zonal=False, fig=fig, ax=ax)
        plt.colorbar(im, ax=ax)
        ax.set_title(r"$V^{-1}\int\mathrm{d}^3 r \;g_\mathrm{NZ}^2$")

        ax = axs[1]
        _, _, im = diagObj.plot_contour_gvmu_vpa(time_idx=time_idx_val, logarithmic=True, vmin="symm", nozonal=False, zonal=True, fig=fig, ax=ax)
        plt.colorbar(im, ax=ax)
        ax.set_title(r"$V^{-1}\int\mathrm{d}^3 r \; g_\mathrm{Z}^2$")

        plt.savefig(fig_filename)
    except Exception as e:
        print(e)
        continue
        #break

# Make movie using ffmpeg
os.system('ffmpeg -r 30 -i ' + img_dir + '/fig_t-%03d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p ' + img_dir + '/video_gvmus_t_'+dirname_string+'.mp4')
