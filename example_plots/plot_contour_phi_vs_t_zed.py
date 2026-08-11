import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})

import stellaDiagnostics as sD

akyminmax_vals = np.asarray([0.2,0.5,1,1.5,1.7])
tprim_val      = 2

base_dirs   = ["fprim-1_adb-el_zetactr-0.00_theta0-0",\
               "fprim-1_adb-el_zetactr-0.25_theta0-0",\
               "fprim-1_adb-el_zetactr-0.50_theta0-0",\
               "fprim-1_adb-el_zetactr-0.75_theta0-0"]

title_basedir = [r"$\zeta_\mathrm{ctr} = 0$",\
                 r"$\zeta_\mathrm{ctr} = \pi/4$",\
                 r"$\zeta_\mathrm{ctr} = \pi/2$",\
                 r"$\zeta_\mathrm{ctr} = 3\pi/4$"]

Ncols = len(akyminmax_vals)
Nrows = len(base_dirs)

fig, axs = plt.subplots(nrows=Nrows,ncols=Ncols, figsize=(18,20))

for i_base_dir, base_dir in enumerate(base_dirs):
    for i_ky, ky_val in enumerate(akyminmax_vals):

        filename = base_dir+"/run_akyminmax-%.4f_tprim-%.4f/precise_QA" % (ky_val, tprim_val)
        StellaObj = sD.stellaDiagnostics(filename)

        ax = axs[i_base_dir, i_ky]
        # NOTE: plot_contour_phi_zed_t does not exist on StellaRun/stellaDiagnostics
        # (pre-existing bug, predates the restructure -- see README "Known issues").
        # The closest current equivalents are plot_quantity_zed_t and
        # RunCollection.plot_contour_phi_vs_zed_theta0.
        StellaObj.plot_contour_phi_zed_t(fig, ax, normalise_phi=True)
        title = title_basedir[i_base_dir] + r", $k_y \rho_i = $ %.2f" % (ky_val)
        ax.set_title(title)

#plt.subplots_adjust(hspace=0)
fig.suptitle(r"$|\phi|(\zeta, t)$ for $a/L_T =$ %i" % (tprim_val))
plt.tight_layout()
plt.savefig("fig_contours_phi_zed_tprim_%i.png" % (tprim_val))
