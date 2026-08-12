"""Collisionality comparison, phi(kx, omega) contours -- run with:
    python ../plot_contour_quantity_vs_kx_omega.py contour_kx_omega_coll_comparison.py
"""

dirname0 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
dirname1 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/"

dirnames = [
    dirname0 + "run_tprim-4.9000",
    dirname1 + "run_tprim-4.9000",
    dirname0 + "run_tprim-6.3000",
    dirname1 + "run_tprim-6.3000",
]
labels = [
    r"$\nu=0, R/L_T=4.9$",
    r"$\nu=10^{-4}, R/L_T=4.9$",
    r"$\nu=0, R/L_T=6.3$",
    r"$\nu=10^{-4}, R/L_T=6.3$",
]
figname_add = "_coll_comparison"

quantity = "phi"
kx_order = 1
time_min = 500
time_max = 1e5
kx_min = 0
kx_max = 1.0
omega_min = -0.1
omega_max = 2
