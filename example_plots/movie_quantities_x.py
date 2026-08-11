import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
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
time_max = 1e6
time_idx_step = 10
fps_rate = 30

time_idx_step = 2
time_min = 500
time_max = 2000
fps_rate = int(30*5/time_idx_step)

#time_idx_step = 1
#time_min = 2200
#time_max = 2500
#fps_rate = 80

ylim = [-1.2, 1.2]
#ylim = [-2, 2]
#ylim = [-4, 4]

# Load data
filename_base = "/CBC"

#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_no-upwind_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_long_tprim-4.9000"

#### eps = 0.045
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0/run_tprim-6.3000"

#### eps = 0.36
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0/run_tprim-6.3000"
#time_min = 1000

#### eps = 0.342
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-09_scan_qinp-1.4_shat-0.8_rmaj-2.778_rhoc-0.95_fprim-0.8_vnew-0/run_tprim-2.5000"
#time_min = 4000
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-09_scan_qinp-1.4_shat-0.8_rmaj-2.778_rhoc-0.95_fprim-0.8_vnew-0/run_tprim-2.2500"
#time_min = 1500
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-09_scan_qinp-1.4_shat-0.8_rmaj-2.778_rhoc-0.95_fprim-0.8_vnew-0/run_tprim-2.0000"
#time_min = 1000

#### shat = 0.32
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.2500/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.5000/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.7500/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-2.0000/"
#time_idx_step = 3

#### shat = 0.16
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-05-05_qinp-1.4_shat-0.16_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.7500"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-05-05_qinp-1.4_shat-0.16_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-2.0000"
#time_idx_step = 3

#### qinp = 1.0
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-2.0000/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-2.2500/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-2.5000/"
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-3.0000/"
#time_idx_step = 3

#### qinp = 2.8
#dirname = "2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000"

#### restarts
#dirname = "../2026-01-20_ES_ITG_Dimits/2026-01-20_tprim_scan_adiab_el/run_tprim-2.5000_restart_scaleZ-10_kmax-0.1"
#time_min = 1.0807E+04; time_max = time_min + 500
#time_idx_step = 3

#### vnew = 0
dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-3.8500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.2000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-5.6000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-5.9500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-8.4000"

#### vnew = 1e-4
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-3.8500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.2000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.2000_HR"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-5.6000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-5.9500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-6.3000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/run_tprim-8.4000"

#### vnew = 1e-3
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-3.8500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-4.2000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-4.9000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-5.6000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-5.9500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-6.3000"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-7.3500"
#dirname = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/run_tprim-8.4000"

#normalise_all = True
normalise_all = False

#never_normalise = True
never_normalise = False

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

dt_avg = None
#dt_avg = 10

# Quantities
#quantities = ["phi",      "RH_phi",                    "dyphi-T",     "P_RH_tot",       "P_RH_even",       "P_RH_odd",      "P_RH_coll"]
#kx_orders  = [    1,             1,                            0,              0,                 0,                0,                0]
#labels     = [r"$v_E^Z$", r"$v_E^{\mathrm{RH},Z}$",        r"$Q$", r"$P_\varphi$", r"$P_\varphi^+$", r"$P_\varphi^-$",         r"$P_C$"]

quantities = ["phi",      "RH_phi",                    "dyphi-T",  "dyphi-upar",     "dyphi2",      "P_RH_tot",        "P_RH_NL",       "P_RH_even",       "P_RH_odd",       "P_RH_coll",                   "upar",            "upar",                   "temperature"]
kx_orders  = [    1,             1,                            0,             0,            0,               0,                0,                 0,                0,                 0,                        0,                 0,                               1]
mults      = [   -1,            -1,                            1,             1,            1,               1,                1,                 1,                1,                 1,                        1,                 1,                               1]
mults_zed  = [ None,          None,                         None,          None,         None,            None,             None,              None,             None,              None,                     None,             "cos",                            None]
colors     = ["0.5",        "0.75",                 "forestgreen",        "0.25",       "brown",           "g",              "k",        "crimson",     "mediumblue",          "yellow",                 "orange",          "purple",                             "c"]
labels     = [r"$v_E^Z$", r"$v_E^{\mathrm{RH},Z}$",   r"$Q$", r"$\Pi_\parallel$", r"$v_{Ex}^2$", r"$P_\mathrm{RH}$", r"$P_\varphi$", r"$P_\varphi^+$", r"$P_\varphi^-$",  r"$P_{\mathrm{RH}}^C$", r"$u_\parallel^Z$",r"$u_\parallel^Z\cos\theta$", r"$\partial_x T^Z$"]
datanames  = ["vE",        "vE_RH",                          "Q", "Pi_parallel",       "vEx2",        "P_RH",          "P_RH_phi",       "P_phi_even",    "P_phi_odd",          "P_RH_coll",           "upar",          "upar_cos",     "gradTZ"]
plot_factors = np.ones_like(mults, dtype='float')

quantities = ["phi"                   , "RH_phi"                               ,     "dyphi-T"       ]#,          "P_RH_even",                         "P_RH_odd"]
kx_orders  = [    1                   ,     1                                  ,             0       ]#,                    0,                                  0]
mults      = [   -1                   ,    -1                                  ,             1       ]#,                    1,                                 -1]
mults_zed  = [ None                   ,  None                                  ,          None       ]#,                 None,                               None]
colors     = [ "mediumblue"           ,   "crimson"                            , "forestgreen"       ]#,           "crimson",                       "mediumblue"]
labels     = [r"$v_E^Z/\rho_* v_{Ti}$", r"$v_{E, \mathrm{RH}}^Z/\rho_* v_{Ti}$", r"$Q/Q_\mathrm{gB}$"]#, r"$r"$P_\mathrm{RH}^{\mathrm{NL},+}$", r"$-P_\mathrm{RH}^{\mathrm{NL},-}$"]

quantities = ["phi"                   ,     "dyphi-T"       ,                        "P_RH_even",                         "P_RH_odd",   "P_RH_coll"       ]
kx_orders  = [    1                   ,             0       ,                                  0,                                  0,             0       ]
mults      = [   -1                   ,             1       ,                                  1,                                  1,             1       ]
mults_zed  = [ None                   ,          None       ,                               None,                               None,          None       ]
colors     = [          "k"           , "forestgreen"       ,                            "crimson",                         "mediumblue",      "orange"       ]
labels     = [r"$v_E^Z/\rho_* v_{Ti}$", r"$Q/Q_\mathrm{gB}$", r"$P_\mathrm{RH}^{\mathrm{NL},+}$", r"$P_\mathrm{RH}^{\mathrm{NL},-}$", r"$P_\mathrm{RH}^C$"]

quantities = ["phi"                   ,     "dyphi-T"       ,                      "Pi_RH_NL",                         "Pi_RH_even",                          "Pi_RH_odd",      "dyphi-upar"]
kx_orders  = [    2                   ,             0       ,                               0,                                    0,                                    0,                 0]
mults      = [   -1                   ,             1       ,                               1,                                    1,                                    1,                 1]
mults_zed  = [ None                   ,          None       ,                            None,                                 None,                                 None,              None]
colors     = [          "0.5"         , "forestgreen"       ,                             "k",                            "crimson",                         "mediumblue",          "purple"]
labels     = [r"$\gamma_E^Z$"         , r"$Q/Q_\mathrm{gB}$",r"$\Pi_\mathrm{RH}^\mathrm{NL}$", r"$\Pi_\mathrm{RH}^{\mathrm{NL},+}$", r"$\Pi_\mathrm{RH}^{\mathrm{NL},-}$",r"$\Pi_\parallel$"]

datanames  = ["tmp", "tmp", "Pi_RH_NL", "Pi_RH_even", "Pi_RH_odd", "Pi_parallel"]
plot_factors = np.ones_like(mults, dtype='float')

# Directory with images
img_dir = "fig_" + dirname_string + "_quantities_x"

if dt_avg is not None:
    img_dir += "_dtavg-%i" % (dt_avg)

import os
os.system("mkdir -p " + img_dir)
if rerun_all:
    os.system('rm -rf ' + img_dir + '/*')
else:
    os.system('rm -rf ' + img_dir + '/video*')

# Prepare plot that will show time-averaged quantities
fig_tavg, axs_tavg = plt.subplots(nrows=len(quantities), figsize=(8, 5*len(quantities)), sharex=True)
if len(quantities)==1:
    axs_tavg = [axs_tavg]
nx = len(diagObj.ncdata.variables['kx'])
avg_quantities_x = np.zeros((len(quantities), nx))

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

        fig, ax = plt.subplots(figsize=(18, 9))
        fig = None
        ax  = None
        for i_quantity, quantity in enumerate(quantities):

            kx_order = kx_orders[i_quantity]
            label    = labels[   i_quantity]
            mult     = mults[    i_quantity]
            mult_zed = mults_zed[i_quantity]
            color    = colors[   i_quantity]

            if (i_time_idx == 0 or normalise_all) and not never_normalise:
                normalise=True
            else:
                normalise=False

            fig, ax, norm_val, x, f_Z = diagObj.plot_quantity_x(quantity=quantity, time_idx=time_idx_val, kx_order=kx_order, fig=fig, ax=ax, label=label, mult=mult, mult_zed=mult_zed, normalise=normalise, color=color, time_avg=dt_avg, plot_factor=plot_factors[i_quantity])

            if i_time_idx == 0:
                plot_factors[i_quantity] *= norm_val

            # For time-average plots
            else:
                if i_quantity == 0:
                    dt = time[time_idx_val]-time[time_idx_val-1]
                    dt_sum += dt
                avg_quantities_x[i_quantity,:] += f_Z*dt

                if overplot_time_trace:
                    axs_tavg[i_quantity].plot(x, f_Z, c='k', alpha=0.05)

        ax.set_ylim(ylim)
        ax.legend(bbox_to_anchor=(1.01, 1), fontsize=20, loc="upper left")

        plt.tight_layout()
        plt.savefig(fig_filename, dpi=150)

        plt.close(fig)

    except Exception as e:
        print(str(e)+"\n")
        traceback.print_exc()
        break

try:
    # Make movie using ffmpeg
    os.system('ffmpeg -r %i -i ' % (fps_rate) + img_dir + '/fig_t-%03d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p ' + img_dir + '/video_quantities_x_'+dirname_string+'.mp4')
except:
    print("COULD NOT MAKE MOVIE!")

if rerun_all:

    # Plot of time averages
    avg_quantities_x /= dt_sum
    
    for i_quantity, quantity in enumerate(quantities):
        label    = labels[i_quantity]
    
        ax = axs_tavg[i_quantity]
    
        ax.plot(x, avg_quantities_x[i_quantity], c='crimson', lw=2)
        ax.axhline(np.mean(avg_quantities_x[i_quantity]), c='crimson', alpha=0.5, ls='--')
    
        if overplot_quantity_idx is not None:
            norm = np.nanmax(np.abs(avg_quantities_x[i_quantity]))/np.nanmax(np.abs(avg_quantities_x[overplot_quantity_idx]))
            ax.plot(x, norm*avg_quantities_x[overplot_quantity_idx], c='forestgreen', ls=':', alpha=0.75, label=labels[overplot_quantity_idx])
            ax.legend(loc='upper left', fontsize=20)
    
        try:
            vmax = np.nanmax(np.abs(avg_quantities_x[i_quantity]))
            
            x1, x2, y1, y2 = x[0], x[-1], -vmax*1.1, vmax*1.1  # subregion of the original image
            axins = ax.inset_axes(
                [0.7, 0.7, 0.27, 0.27],
                xlim=(x1, x2), ylim=(y1, y2))#, xticklabels=[], yticklabels=[])
            axins.plot(x, avg_quantities_x[i_quantity], c='crimson', lw=2)
            axins.axhline(np.mean(avg_quantities_x[i_quantity]), c='crimson', alpha=0.5, ls='--')
            axins.grid(alpha=0.5)
    
            if overplot_quantity_idx is not None:
                norm = np.nanmax(np.abs(avg_quantities_x[i_quantity]))/np.nanmax(np.abs(avg_quantities_x[overplot_quantity_idx]))
                axins.plot(x, norm*avg_quantities_x[overplot_quantity_idx], c='forestgreen', ls=':', alpha=0.75, label=labels[overplot_quantity_idx])
    
        except Exception as e:
            print(e)
    
        ax.set_ylabel(label)
    
        np.savetxt(dirname+"/data_"+datanames[i_quantity]+".dat", avg_quantities_x[i_quantity])
    np.savetxt(dirname+"/data_x.dat", x)
    
    for ax in axs_tavg:
        ax.grid(alpha=0.5)
        ax.set_xlim([x[0], x[-1]])
    
    axs_tavg[-1].set_xlabel(r"$x/\rho_i$")
    
    fig_tavg.tight_layout()
    fig_tavg.savefig("fig_quantities_x_tavg_"+dirname_string+".pdf")
    
        
