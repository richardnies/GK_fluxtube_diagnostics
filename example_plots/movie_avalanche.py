"""Movie of a 6-row x 3-column diagnostic grid for one run, tracking an
avalanche event through time: (x,t) heat-flux/vE/upar/Q traces (column 0,
plotted once -- static across the whole movie except a moving time
marker), (x,zed) contours of phi/upar/Q/RH-power (column 1), and (x,y)
real-space snapshots of density/upar/temperature/Q/Pi (column 2).

Usage:
    python movie_avalanche.py <config.py>

<config.py> defines `dirname` (required) and optionally `filename`, `code`,
`time_min`, `time_max`, `time_idx_step`, `rerun_all`, `filename_ending`,
`ny_padded`, `nx_padded`, `plot_Pis`, `fps`, `img_dir`.
"""
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.plotting.movies import ffmpeg_frames_to_video
from stella_diagnostics.scan.config import load_scan_config

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirname",))

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
time_min = getattr(config, "time_min", 2399)
time_max = getattr(config, "time_max", 2601)
time_idx_step = getattr(config, "time_idx_step", 1)
rerun_all = getattr(config, "rerun_all", True)
filename_ending = getattr(config, "filename_ending", ".png")
ny_padded = getattr(config, "ny_padded", None)
nx_padded = getattr(config, "nx_padded", None)
plot_Pis = getattr(config, "plot_Pis", False)
fps = getattr(config, "fps", 5)

run = StellaRun(config.dirname + "/" + filename, code=code)

time_idx_min = run.get_time_idx(time_min)
time_idx_max = run.get_time_idx(time_max)
time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)

# Create figure
fig, axs = plt.subplots(ncols=3, nrows=6, figsize=(21, 42))

# Directory with images
dirname_string = config.dirname.replace("/", "_")
img_dir = getattr(config, "img_dir", None) or "fig_avalanche_" + dirname_string
img_dir = Path(img_dir)
if rerun_all and img_dir.exists():
    for f in img_dir.glob("*" + filename_ending):
        f.unlink()
img_dir.mkdir(parents=True, exist_ok=True)

####### (x,t) plots
vlines = []

# 1D plot of heat flux
_, _, qflx, time = run.get_fluxes_over_time()
qflx = qflx[(time > time_min) & (time < time_max)]
time = time[(time > time_min) & (time < time_max)]
kx = run.ncdata['kx'][:]
xmax = np.pi / (kx[1] - kx[0])
norm = qflx.max() / (xmax / 2)
for ax in axs[:, 0]:
    ax.plot(time, qflx / norm - xmax, c='forestgreen', label=r"$Q/Q_\mathrm{gB}$", lw=3, zorder=int(1e25))
axs[0, 0].legend()

# vE(x, t)
print("\n*****************")
print("Plotting vE(x, t)")
print("*****************\n")
ax = axs[0, 0]
_, _, im, _, _, _ = run.plot_quantity_x_t(quantity="phi", fig=fig, ax=ax, only_zonal=True, cmap='coolwarm', time_idx_skip=time_idx_step, kx_order=1, mult=-1, time_min=time_min, time_max=time_max, vmin="symm", vmax="last", nx=nx_padded)
plt.colorbar(im, ax=ax)
ax.set_title(r"$v_E^Z$")

# upar(x, t, theta=0)
print("\n*****************")
print("Plotting upar(x, t, theta=0)")
print("*****************\n")
ax = axs[1, 0]
_, _, im, _, _, _ = run.plot_quantity_x_t(quantity="upar", fig=fig, ax=ax, only_zonal=True, cmap='coolwarm', time_idx_skip=time_idx_step, kx_order=0, mult=1, time_min=time_min, time_max=time_max, vmin="symm", vmax="last", nx=nx_padded, zed_val=0)
plt.colorbar(im, ax=ax)
ax.set_title(r"$u_\parallel^Z (\theta=0)$")

# Q(x, t)
print("\n*****************")
print("Plotting Q(x, t)")
print("*****************\n")
ax = axs[2, 0]
_, _, im, _, _, Z = run.plot_quantity_x_t(quantity="dyphi-T", fig=fig, ax=ax, only_zonal=True, cmap='coolwarm', time_idx_skip=time_idx_step, kx_order=0, mult=1, time_min=time_min, time_max=time_max, vmin="symm", vmax="last", nx=nx_padded)
plt.colorbar(im, ax=ax)
ax.set_title(r"$\langle v_{Ex}T \rangle_{y, \theta}$")

vmax = np.abs(Z[:, -1]).max()
vmin = -vmax

# Q(x, t, theta=pi/2)
print("\n*****************")
print("Plotting Q(x, t, theta=pi/2)")
print("*****************\n")
ax = axs[3, 0]
_, _, im, _, _, _ = run.plot_quantity_x_t(quantity="dyphi-T", fig=fig, ax=ax, only_zonal=True, cmap='coolwarm', time_idx_skip=time_idx_step, kx_order=0, mult=1, time_min=time_min, time_max=time_max, vmin=vmin, vmax=vmax, zed_val=np.pi / 2, nx=nx_padded)
plt.colorbar(im, ax=ax)
ax.set_title(r"$\langle v_{Ex}T \rangle_{y} (\theta=\pi/2)$")

# Q(x, t, theta=0)
print("\n*****************")
print("Plotting Q(x, t, theta=0)")
print("*****************\n")
ax = axs[4, 0]
_, _, im, _, _, _ = run.plot_quantity_x_t(quantity="dyphi-T", fig=fig, ax=ax, only_zonal=True, cmap='coolwarm', time_idx_skip=time_idx_step, kx_order=0, mult=1, time_min=time_min, time_max=time_max, vmin=vmin, vmax=vmax, zed_val=0, nx=nx_padded)
plt.colorbar(im, ax=ax)
ax.set_title(r"$\langle v_{Ex}T \rangle_{y} (\theta=0)$")

# Q(x, t, theta=-pi/2)
print("\n*****************")
print("Plotting Q(x, t, theta=-pi/2)")
print("*****************\n")
ax = axs[5, 0]
_, _, im, _, _, _ = run.plot_quantity_x_t(quantity="dyphi-T", fig=fig, ax=ax, only_zonal=True, cmap='coolwarm', time_idx_skip=time_idx_step, kx_order=0, mult=1, time_min=time_min, time_max=time_max, vmin=vmin, vmax=vmax, zed_val=-np.pi / 2, nx=nx_padded)
plt.colorbar(im, ax=ax)
ax.set_title(r"$\langle v_{Ex}T \rangle_{y} (\theta=-\pi/2)$")

##### (x, zed) plots

cbars_xzed = []

# Obtain vEzonal normalisation
dxphizonal, x, y, _ = run.get_quantity_x_y(quantity="phi", time_idx=time_idx_max, only_zonal=True, kx_order=1, nx=nx_padded)
norm_vEzonal = np.pi / 2 / (np.abs(dxphizonal)).max()

# Obtain vmin, vmax from last time point
vmins_xzed = []
vmaxs_xzed = []

fig_tmp, ax_tmp = plt.subplots()

# vEZ(x, theta)
_, _, im, _, _, _, _, vmin, vmax = run.plot_quantity_x_zed(fig=fig_tmp, ax=ax_tmp, quantity="phi", time_idx=time_idx_vals[-1], only_zonal=True, ny=ny_padded, nx=nx_padded, vmin="symm", vmax=None, cmap='coolwarm', kx_order=1, mult_fac=-1)
vmins_xzed.append(vmin); vmaxs_xzed.append(vmax)

# uparallel(x, theta)
_, _, im, _, _, _, _, vmin, vmax = run.plot_quantity_x_zed(fig=fig_tmp, ax=ax_tmp, quantity="upar", time_idx=time_idx_vals[-1], only_zonal=True, ny=ny_padded, nx=nx_padded, vmin="symm", vmax=None, cmap='coolwarm')
vmins_xzed.append(vmin); vmaxs_xzed.append(vmax)

# Q(x, theta)
_, _, im, _, _, _, _, vmin, vmax = run.plot_quantity_x_zed(fig=fig_tmp, ax=ax_tmp, quantity="dyphi-T", time_idx=time_idx_vals[-1], only_zonal=True, ny=ny_padded, nx=nx_padded, vmin="symm", vmax=None, cmap='coolwarm')
vmins_xzed.append(vmin); vmaxs_xzed.append(vmax)

# P_RH_tot
_, _, im, _, _, _, _, vmin, vmax = run.plot_quantity_x_zed(fig=fig_tmp, ax=ax_tmp, quantity="P_RH_tot", time_idx=time_idx_vals[-1], only_zonal=True, ny=ny_padded, nx=nx_padded, vmin="symm", vmax=None, cmap='coolwarm')
vmins_xzed.append(vmin); vmaxs_xzed.append(vmax)

if plot_Pis:
    # Pi_parallel(x, theta)
    _, _, im, _, _, _, _, vmin, vmax = run.plot_quantity_x_zed(fig=fig_tmp, ax=ax_tmp, quantity="par_mom_transport", time_idx=time_idx_vals[-1], only_zonal=True, ny=ny_padded, nx=nx_padded, vmin="symm", vmax=None, cmap='coolwarm')
    vmins_xzed.append(vmin); vmaxs_xzed.append(vmax)

    # Pi_perp(x, theta)
    _, _, im, _, _, _, _, vmin, vmax = run.plot_quantity_x_zed(fig=fig_tmp, ax=ax_tmp, quantity="Reynolds", time_idx=time_idx_vals[-1], only_zonal=True, ny=ny_padded, nx=nx_padded, vmin="symm", vmax=None, cmap='coolwarm')
    vmins_xzed.append(vmin); vmaxs_xzed.append(vmax)

else:
    # P_RH_odd
    _, _, im, _, _, _, _, vmin, vmax = run.plot_quantity_x_zed(fig=fig_tmp, ax=ax_tmp, quantity="P_RH_odd", time_idx=time_idx_vals[-1], only_zonal=True, ny=ny_padded, nx=nx_padded, vmin="symm", vmax=None, cmap='coolwarm')
    vmins_xzed.append(vmin); vmaxs_xzed.append(vmax)

    # P_RH_even
    _, _, im, _, _, _, _, vmin, vmax = run.plot_quantity_x_zed(fig=fig_tmp, ax=ax_tmp, quantity="P_RH_even", time_idx=time_idx_vals[-1], only_zonal=True, ny=ny_padded, nx=nx_padded, vmin="symm", vmax=None, cmap='coolwarm')
    vmins_xzed.append(vmin); vmaxs_xzed.append(vmax)


#### (x, y) plots
zed_val = 0
zed_label = r"$0$"

cbars_xy = []

vmins_xy = []
vmaxs_xy = []

# n(x, y)
_, _, im, vmin, vmax = run.plot_quantity_x_y(quantity="density", fig=fig_tmp, ax=ax_tmp, zed_val=zed_val, time_idx=time_idx_vals[-1], nx=nx_padded, ny=ny_padded, cmap='coolwarm', suptitle=False, xy_layout=False, symm=True)
vmins_xy.append(vmin); vmaxs_xy.append(vmax)

# uparallel(x, y)
_, _, im, vmin, vmax = run.plot_quantity_x_y(quantity="upar", fig=fig_tmp, ax=ax_tmp, zed_val=zed_val, time_idx=time_idx_vals[-1], nx=nx_padded, ny=ny_padded, cmap='coolwarm', suptitle=False, xy_layout=False, symm=True)
vmins_xy.append(vmin); vmaxs_xy.append(vmax)

_, _, im, vmin, vmax = run.plot_quantity_x_y(quantity="temperature", fig=fig_tmp, ax=ax_tmp, zed_val=zed_val, time_idx=time_idx_vals[-1], nx=nx_padded, ny=ny_padded, cmap='coolwarm', suptitle=False, xy_layout=False, symm=True)
vmins_xy.append(vmin); vmaxs_xy.append(vmax)

# Q(x, y)
_, _, im, vmin, vmax = run.plot_quantity_x_y(quantity="dyphi-T", fig=fig_tmp, ax=ax_tmp, zed_val=zed_val, time_idx=time_idx_vals[-1], nx=nx_padded, ny=ny_padded, cmap='coolwarm', suptitle=False, xy_layout=False, symm=True)
vmins_xy.append(vmin); vmaxs_xy.append(vmax)

# Pi_parallel(x, y)
_, _, im, vmin, vmax = run.plot_quantity_x_y(quantity="par_mom_transport", fig=fig_tmp, ax=ax_tmp, zed_val=zed_val, time_idx=time_idx_vals[-1], nx=nx_padded, ny=ny_padded, cmap='coolwarm', suptitle=False, xy_layout=False, symm=True)
vmins_xy.append(vmin); vmaxs_xy.append(vmax)

# Pi_perp(x, y)
_, _, im, vmin, vmax = run.plot_quantity_x_y(quantity="Reynolds", fig=fig_tmp, ax=ax_tmp, zed_val=zed_val, time_idx=time_idx_vals[-1], nx=nx_padded, ny=ny_padded, cmap='coolwarm', suptitle=False, xy_layout=False, symm=True)
vmins_xy.append(vmin); vmaxs_xy.append(vmax)

plt.close(fig_tmp)


#### Go through time frames
time_all = run.get_time_array()

print("Starting to plot time frames...\n")

for i_time_idx, time_idx_val in enumerate(time_idx_vals):
    print("Plotting figure %i/%i..." % (i_time_idx + 1, len(time_idx_vals)), end="\r")

    time = time_all[time_idx_val]

    fig_filename = str(img_dir) + "/fig_t-%.3i" % (i_time_idx) + filename_ending

    if not rerun_all and Path(fig_filename).exists():
        continue

    try:
        ##### (x, t) plots

        # Remove old vlines
        for vline in vlines:
            vline.remove()

        # Create new vlines
        vlines = []
        for ax in axs[:, 0]:
            vline = ax.axvline(time, c='k', lw=2)
            vlines.append(vline)

        #### (x, zed) plots
        for cbar in cbars_xzed:
            cbar.remove()
        cbars_xzed = []

        for ax in axs[:, 1]:
            ax.clear()
            ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

        # vEZ(x, theta)
        ax = axs[0, 1]
        ax.set_title(r"$v_E^Z$")
        _, _, im, _, _, _, _, _, _ = run.plot_quantity_x_zed(fig=fig, ax=ax, quantity="phi", time_idx=time_idx_val, only_zonal=True, ny=ny_padded, nx=nx_padded, vmin=vmins_xzed[0], vmax=vmaxs_xzed[0], cmap='coolwarm', kx_order=1, mult_fac=-1)
        cbar = fig.colorbar(im, ax=ax)
        cbars_xzed.append(cbar)

        # uparallel(x, theta)
        ax = axs[1, 1]
        ax.set_title(r"$u_\parallel^Z$")
        _, _, im, _, _, _, _, _, _ = run.plot_quantity_x_zed(fig=fig, ax=ax, quantity="upar", time_idx=time_idx_val, only_zonal=True, ny=ny_padded, nx=nx_padded, vmin=vmins_xzed[1], vmax=vmaxs_xzed[1], cmap='coolwarm')
        cbar = fig.colorbar(im, ax=ax)
        cbars_xzed.append(cbar)

        # Q(x, theta)
        ax = axs[2, 1]
        ax.set_title(r"$\langle v_{Ex} T \rangle_y$")
        _, _, im, _, _, _, _, _, _ = run.plot_quantity_x_zed(fig=fig, ax=ax, quantity="dyphi-T", time_idx=time_idx_val, only_zonal=True, ny=ny_padded, nx=nx_padded, vmin=vmins_xzed[2], vmax=vmaxs_xzed[2], cmap='coolwarm')
        cbar = fig.colorbar(im, ax=ax)
        cbars_xzed.append(cbar)

        # P_RH_tot
        ax = axs[3, 1]
        ax.set_title(r"$P_\mathrm{RH}$")
        _, _, im, _, _, _, _, _, _ = run.plot_quantity_x_zed(fig=fig, ax=ax, quantity="P_RH_tot", time_idx=time_idx_val, only_zonal=True, ny=ny_padded, nx=nx_padded, vmin=vmins_xzed[3], vmax=vmaxs_xzed[3], cmap='coolwarm')
        cbar = fig.colorbar(im, ax=ax)
        cbars_xzed.append(cbar)

        if plot_Pis:
            # Pi_parallel(x, theta)
            ax = axs[4, 1]
            ax.set_title(r"$\Pi_\parallel$")
            _, _, im, _, _, _, _, _, _ = run.plot_quantity_x_zed(fig=fig, ax=ax, quantity="par_mom_transport", time_idx=time_idx_val, only_zonal=True, ny=ny_padded, nx=nx_padded, vmin=vmins_xzed[4], vmax=vmaxs_xzed[4], cmap='coolwarm')
            cbar = fig.colorbar(im, ax=ax)
            cbars_xzed.append(cbar)

            # Pi_perp(x, theta)
            ax = axs[5, 1]
            ax.set_title(r"$\Pi_\perp$")
            _, _, im, _, _, _, _, _, _ = run.plot_quantity_x_zed(fig=fig, ax=ax, quantity="Reynolds", time_idx=time_idx_val, only_zonal=True, ny=ny_padded, nx=nx_padded, vmin=vmins_xzed[5], vmax=vmaxs_xzed[5], cmap='coolwarm')
            cbar = fig.colorbar(im, ax=ax)
            cbars_xzed.append(cbar)

        else:
            # P_RH_odd
            ax = axs[4, 1]
            ax.set_title(r"$P_\mathrm{RH}^-$")
            _, _, im, _, _, _, _, _, _ = run.plot_quantity_x_zed(fig=fig, ax=ax, quantity="P_RH_odd", time_idx=time_idx_val, only_zonal=True, ny=ny_padded, nx=nx_padded, vmin=vmins_xzed[4], vmax=vmaxs_xzed[4], cmap='coolwarm')
            cbar = fig.colorbar(im, ax=ax)
            cbars_xzed.append(cbar)

            # P_RH_even
            ax = axs[5, 1]
            ax.set_title(r"$P_\mathrm{RH}^+$")
            _, _, im, _, _, _, _, _, _ = run.plot_quantity_x_zed(fig=fig, ax=ax, quantity="P_RH_even", time_idx=time_idx_val, only_zonal=True, ny=ny_padded, nx=nx_padded, vmin=vmins_xzed[5], vmax=vmaxs_xzed[5], cmap='coolwarm')
            cbar = fig.colorbar(im, ax=ax)
            cbars_xzed.append(cbar)

        # Overplot vEZ(x)
        dxphizonal, x, y, _ = run.get_quantity_x_y(quantity="phi", time_idx=time_idx_val, only_zonal=True, kx_order=1, nx=nx_padded)

        rot = transforms.Affine2D().rotate_deg(90)
        for ax in axs[:, 1]:
            base = ax.transData
            ax.plot(x, dxphizonal[:, 0] * norm_vEzonal, c='forestgreen', transform=rot + base, label=r"$v_E^Z$", lw=3)
            ax.set_xlim([-np.pi, np.pi])
        axs[0, 1].legend()

        ##### (x, y) plots
        for cbar in cbars_xy:
            cbar.remove()
        cbars_xy = []

        for ax in axs[:, 2]:
            ax.clear()

        # n(x, y)
        ax = axs[0, 2]
        ax.set_title(r"$n (\theta = $" + zed_label + r"$)$")
        _, _, im, _, _ = run.plot_quantity_x_y(quantity="density", fig=fig, ax=ax, zed_val=zed_val, time_idx=time_idx_val, nx=nx_padded, ny=ny_padded, vmin=vmins_xy[0], vmax=vmaxs_xy[0], cmap='coolwarm', suptitle=False, xy_layout=False)
        cbar = fig.colorbar(im, ax=ax)
        cbars_xy.append(cbar)

        # uparallel(x, y)
        ax = axs[1, 2]
        ax.set_title(r"$u_\parallel (\theta = $" + zed_label + r"$)$")
        _, _, im, _, _ = run.plot_quantity_x_y(quantity="upar", fig=fig, ax=ax, zed_val=zed_val, time_idx=time_idx_val, nx=nx_padded, ny=ny_padded, vmin=vmins_xy[1], vmax=vmaxs_xy[1], cmap='coolwarm', suptitle=False, xy_layout=False)
        cbar = fig.colorbar(im, ax=ax)
        cbars_xy.append(cbar)

        # temperature(x, y)
        ax = axs[2, 2]
        ax.set_title(r"$T (\theta = $" + zed_label + r"$)$")
        _, _, im, _, _ = run.plot_quantity_x_y(quantity="temperature", fig=fig, ax=ax, zed_val=zed_val, time_idx=time_idx_val, nx=nx_padded, ny=ny_padded, vmin=vmins_xy[2], vmax=vmaxs_xy[2], cmap='coolwarm', suptitle=False, xy_layout=False)
        cbar = fig.colorbar(im, ax=ax)
        cbars_xy.append(cbar)

        # Q(x, y)
        ax = axs[3, 2]
        ax.set_title(r"$ v_{Ex} T (\theta = $" + zed_label + r"$)$")
        _, _, im, _, _ = run.plot_quantity_x_y(quantity="dyphi-T", fig=fig, ax=ax, zed_val=zed_val, time_idx=time_idx_val, nx=nx_padded, ny=ny_padded, vmin=vmins_xy[3], vmax=vmaxs_xy[3], cmap='coolwarm', suptitle=False, xy_layout=False)
        cbar = fig.colorbar(im, ax=ax)
        cbars_xy.append(cbar)

        # Pi_parallel(x, y)
        ax = axs[4, 2]
        ax.set_title(r"$\Pi_\parallel (\theta = $" + zed_label + r"$)$")
        _, _, im, _, _ = run.plot_quantity_x_y(quantity="par_mom_transport", fig=fig, ax=ax, zed_val=zed_val, time_idx=time_idx_val, nx=nx_padded, ny=ny_padded, vmin=vmins_xy[4], vmax=vmaxs_xy[4], cmap='coolwarm', suptitle=False, xy_layout=False)
        cbar = fig.colorbar(im, ax=ax)
        cbars_xy.append(cbar)

        # Pi_perp(x, y)
        ax = axs[5, 2]
        ax.set_title(r"$\Pi_\perp (\theta = $" + zed_label + r"$)$")
        _, _, im, _, _ = run.plot_quantity_x_y(quantity="Reynolds", fig=fig, ax=ax, zed_val=zed_val, time_idx=time_idx_val, nx=nx_padded, ny=ny_padded, vmin=vmins_xy[5], vmax=vmaxs_xy[5], cmap='coolwarm', suptitle=False, xy_layout=False)
        cbar = fig.colorbar(im, ax=ax)
        cbars_xy.append(cbar)

        #### Finish plot
        # xlabels
        for ax in axs[:-1, :].flatten():
            ax.set_xlabel(None)

        # ylabels
        for ax in axs[:, 1:].flatten():
            ax.set_ylabel(None)

        # Title
        fig.suptitle(r"$t v_{Ti}/a = %.2f$" % (time), fontsize=50)

        # Save plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(fig_filename)

    except Exception as e:
        print("************************")
        print(e)
        traceback.print_exc()
        print("************************")
        continue

# Make movie using ffmpeg
ffmpeg_frames_to_video(img_dir, "fig_t-*" + filename_ending, img_dir / "video_x_zed.mp4", fps=fps, extra_args=["-vf", "fps=30"])
