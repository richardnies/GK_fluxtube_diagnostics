"""Upwind/mirror-scheme numerical-parameter scan, all at R/L_T=4.9.

Was the "## UPWIND SCAN" block in plot_flux_time.py before it was
extracted -- run with:
    python ../plot_flux_time.py scan_upwind.py
"""

dir_0 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
dir_1 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-5/"

dirnames = [
    dir_0 + "run_tprim-4.9000",
    dir_0 + "run_upwind-0.005_tprim-4.9000",
    dir_0 + "run_upwind-0.002_tprim-4.9000",
    dir_0 + "run_no-upwind_tprim-4.9000",
    dir_0 + "run_drifts-implicit-T_tprim-4.9000",
    dir_0 + "run_upwind-0.002_drifts-implicit-T_tprim-4.9000",
    dir_0 + "run_upwind-0.005_drifts-implicit-T_tprim-4.9000",
    dir_0 + "run_no-upwind_drifts-implicit-T_mirror-SL-True_tprim-4.900",
    dir_1 + "run_tprim-4.9000",
    dir_1 + "run_no-upwind_tprim-4.9000",
]
labels = [
    r"Expl dr., $u=0.02$, msl T",
    r"Expl dr., $u=0.005$, msl F",
    r"Expl dr., $u=0.002$, msl F",
    r"Expl dr., $u=0$, msl F",
    r"Impl dr., $u=0.02$, msl T",
    r"Impl dr., $u=0.002$, msl F",
    r"Impl dr., $u=0.005$, msl F",
    r"Impl dr., $u=0$, msl T",
    r"$\nu=10^{-5}$",
    r"$\nu = 10^{-5}$, no upwind",
]
colors = ["k", "orange", "crimson", "forestgreen", "mediumblue", "purple", "c", "pink", "0.5", "yellow"]
ylim = [1e-4, 1e2]
figname_add = "_upwind-scan"
