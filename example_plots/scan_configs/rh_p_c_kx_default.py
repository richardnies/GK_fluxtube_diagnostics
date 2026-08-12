"""Config for plot_RH_P_C_kx_from_file.py -- run with:
    python ../plot_RH_P_C_kx_from_file.py rh_p_c_kx_default.py
"""

basedir = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-"
vnew_vals = [0.0001, 0.001, 0.01]
vnew_dirs = {0.0001: "0.0001", 0.001: "0.001", 0.01: "0.01"}
vnew_labels = {0.0001: r"$10^{-4}$", 0.001: r"$10^{-3}$", 0.01: r"$10^{-2}$"}
tprim_vals = [4.2, 4.9, 5.6, 5.95, 6.3, 8.4]
eps = 0.18
