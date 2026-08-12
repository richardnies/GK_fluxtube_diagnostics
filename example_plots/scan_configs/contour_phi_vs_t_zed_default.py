"""Config for plot_contour_phi_vs_t_zed.py -- run with:
    python ../plot_contour_phi_vs_t_zed.py contour_phi_vs_t_zed_default.py

NOTE: this script calls a method (plot_contour_phi_zed_t) that has never
existed on StellaRun -- see README "Known issues". Included here anyway
so the config-driven pattern is complete/consistent across all scripts;
running it will raise AttributeError, same as before this migration.
"""

akyminmax_vals = [0.2, 0.5, 1, 1.5, 1.7]
tprim_val = 2

base_dirs = [
    "fprim-1_adb-el_zetactr-0.00_theta0-0",
    "fprim-1_adb-el_zetactr-0.25_theta0-0",
    "fprim-1_adb-el_zetactr-0.50_theta0-0",
    "fprim-1_adb-el_zetactr-0.75_theta0-0",
]
base_dir_titles = [
    r"$\zeta_\mathrm{ctr} = 0$",
    r"$\zeta_\mathrm{ctr} = \pi/4$",
    r"$\zeta_\mathrm{ctr} = \pi/2$",
    r"$\zeta_\mathrm{ctr} = 3\pi/4$",
]
filename_template = "%s/run_akyminmax-%.4f_tprim-%.4f/precise_QA"
