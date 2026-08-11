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

dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_long_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_no-upwind_tprim-4.9000"

dirname_string = dirname.replace("/","_")

filename = dirname+filename_base
diagObj = sD.stellaDiagnostics(filename)

# Setup
#kx_idxs = [0,5,10,25,50,75]
#Nr_kx_vals = len(kx_idxs)
#
#ky_idxs = [1,5,10,20]
#Nr_ky_vals = len(ky_idxs)

#remove_zonal = False
remove_zonal = True

overplot_zonal = True
#overplot_zonal = False

nx_padded = None
ny_padded = None

ky_order = 0

#time_idx_min  = -50
#time_idx_max  = -1
time_min  = 0
#time_min  = 3500
time_max  = 115000
time_idx_min = diagObj.get_time_idx(time_min)
time_idx_max = diagObj.get_time_idx(time_max)
time_idx_step = 1
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)

zed_vals = [0, np.pi/2, np.pi]#, -np.pi/4, 0, np.pi/4]
#zed_vals = [None, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]

mult_zed = None
#mult_zed = "neg"

#rerun_all = True
rerun_all = False

#quantity = "dyphi-T"
quantity = "density"
quantity = "upar"
#quantity = "temperature"

# Directory with images
img_dir = "fig_" + dirname_string + "_" + quantity + "_real_space"
if ky_order != 0:
    img_dir = img_dir + "_ky-order-%i" % (ky_order)

if mult_zed is not None:
    img_dir = img_dir+"_mult_zed-" + mult_zed

if remove_zonal:
    img_dir = img_dir+"_no_zonal"
import os
os.system("mkdir -p " + img_dir)
if rerun_all:
    os.system('rm -rf ' + img_dir + '/*')
else:
    os.system('rm -rf ' + img_dir + '/video*')

# Obtain vmin-vmax from last timestep
_, _, _, vmin, vmax = diagObj.plot_quantity_x_y(quantity=quantity, time_idx=-1, remove_zonal=remove_zonal, ky_order=ky_order, nx=nx_padded, ny=ny_padded, symm=True, zed_val=0)
#vmin=vmax=None

# Obtain dxphizonal normalisation if required
if overplot_zonal:
    dxphizonal, x, y, _ = diagObj.get_quantity_x_y(quantity="phi", time_idx=-1, only_zonal=True, kx_order=1, nx=nx_padded)
    norm_dxphizonal = y[-1] / (np.abs(dxphizonal)).max()/4
    tempzonal, x, y, _ = diagObj.get_quantity_x_y(quantity="temperature", time_idx=-1, only_zonal=True, kx_order=0, nx=nx_padded)
    norm_tempzonal = y[-1] / (np.abs(tempzonal)).max()/4
    dxtempzonal, x, y, _ = diagObj.get_quantity_x_y(quantity="temperature", time_idx=-1, only_zonal=True, kx_order=1, nx=nx_padded)
    norm_dxtempzonal = y[-1] / (np.abs(dxtempzonal)).max()/4
    uparzonal, x, y, _ = diagObj.get_quantity_x_y(quantity="upar", time_idx=-1, only_zonal=True, kx_order=0, nx=nx_padded)
    norm_uparzonal = y[-1] / (np.abs(uparzonal)).max()/4

# Plot for each timestep
for i_time_idx, time_idx_val in enumerate(time_idx_vals):
    print("Plotting figure %i/%i..." % (i_time_idx+1, len(time_idx_vals)), end="\r")

    fig_filename = img_dir+"/fig_t-%.3i.png" % (i_time_idx)

    if not rerun_all and exists(fig_filename):
        continue

    plt.close()
    try:

        fig, axs = plt.subplots(ncols=len(zed_vals), figsize=(9*len(zed_vals),9))
        for i_zed, zed_val in enumerate(zed_vals):
            if len(zed_vals) == 1:
                ax = axs
            else:
                ax = axs[i_zed]

            _, _, im, vmin, vmax = diagObj.plot_quantity_x_y(quantity=quantity, time_idx=time_idx_val, remove_zonal=remove_zonal, ky_order=ky_order, nx=nx_padded, ny=ny_padded, vmin=vmin, vmax=vmax, fig=fig, ax=ax, zed_val=zed_val)
            fig.colorbar(im, ax=ax)
            ax.set_aspect('equal')
            if overplot_zonal:
                dxphizonal, x, y, _  = diagObj.get_quantity_x_y(quantity="phi",         time_idx=time_idx_val, only_zonal=True, kx_order=1, nx=nx_padded, zed_val=zed_val)
                tempzonal, x, y, _ = diagObj.get_quantity_x_y(quantity="temperature", time_idx=time_idx_val, only_zonal=True, kx_order=0, nx=nx_padded, zed_val=zed_val)
                dxtempzonal, x, y, _ = diagObj.get_quantity_x_y(quantity="temperature", time_idx=time_idx_val, only_zonal=True, kx_order=1, nx=nx_padded, zed_val=zed_val)
                uparzonal, x, y, _ = diagObj.get_quantity_x_y(quantity="upar", time_idx=time_idx_val, only_zonal=True, kx_order=0, nx=nx_padded, zed_val=zed_val)
                tprim = diagObj.ncdata.variables['tprim'][0]
                tprim_lin = (1+1)*(1.33+1.91*0.8/1.4)*(1-1.5*0.18) 

                ax.plot(x, -dxphizonal[:,0]*norm_dxphizonal, c='forestgreen')
                ax.plot(x, uparzonal[:,0]*norm_uparzonal, c='c')
                #print(np.abs(dxtempzonal).max())
#                ax.plot(x, (dxtempzonal[:,0]+0.5*(tprim_lin-tprim))*norm_dxtempzonal, c='crimson')
#                ax.plot(x, dxtempzonal[:,0]+0.5*(tprim_lin-tprim))*norm_dxtempzonal, c='crimson')
#                ax.plot(x, tempzonal[:,0]*norm_tempzonal, c='crimson')
                ax.set_ylim([y[0], y[-1]])
                ax.set_xlim([x[0], x[-1]])
            if zed_val is None:
                title = r"$\theta$ avg"
            else:
                title = r"$\theta = %.2f$" % (zed_val)
            ax.set_title(title)

#            if quantity == "density":
#                ax.set_title(r"$(\delta n_i/n_i) (a/\rho_i)$")
#            if quantity == "temperature":
#                ax.set_title(r"$(\delta T_i/T_i) (a/\rho_i)$")

        plt.tight_layout()
        plt.savefig(fig_filename)

    except Exception as e:
        print(str(e)+"\n")
        break

# Make movie using ffmpeg
#os.system('ffmpeg -r 50 -i ' + img_dir + '/fig_t-%03d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p ' + img_dir + '/video_'+quantity+'_real_space.mp4')
os.system('ffmpeg -r 20 -i ' + img_dir + '/fig_t-%03d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p ' + img_dir + '/video_'+quantity+'_real_space.mp4')
#os.system('ffmpeg -r 50 -i ' + img_dir + '/fig_t-%03d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p ' + img_dir + '/video_'+quantity+'_real_space.mp4')
