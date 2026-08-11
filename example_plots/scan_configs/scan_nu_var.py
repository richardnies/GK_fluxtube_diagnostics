"""Collision-frequency scan: nu=0 vs nu=1e-3, at two temperature gradients.

Was the "## COLLISION SCAN" block in plot_flux_time.py before it was
extracted -- run with:
    python ../plot_flux_time.py scan_nu_var.py
"""

dir_cless = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
dir_cal = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/"

dirnames = [
    dir_cless + "run_tprim-4.9000",
    dir_cless + "run_tprim-6.3000",
    dir_cal + "run_tprim-4.9000",
    dir_cal + "run_tprim-6.3000",
]
labels = [
    r"CBC ($\nu=0$) $R/L_T=4.9$",
    r"$R/L_T = 6.3$",
    r"CBC ($\nu=10^{-3}$) $R/L_T=4.9$",
    r"$R/L_T = 6.3$",
]
colors = ["k", "orange", "crimson", "forestgreen"]
ylim = [3e-3, 3e2]
figname_add = "_nu-var"
