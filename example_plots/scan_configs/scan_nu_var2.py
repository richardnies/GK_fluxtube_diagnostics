"""Collision-frequency scan (v2: includes a higher-velocity-resolution and
a restarted/long run), all at R/L_T=4.9.

Was the "## COLLISION SCAN 2" block in plot_flux_time.py -- the one
actually active before this script was migrated -- before it was
extracted. Run with:
    python ../plot_flux_time.py scan_nu_var2.py
"""

dir_0 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0_higher-vel-res/"
dir_1 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
dir_2 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/"
dir_3 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-4/"
dir_4 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/"

dirnames = [
    dir_0 + "run_tprim-4.9000",
    dir_1 + "run_tprim-4.9000",
    dir_1 + "run_long_tprim-4.9000",
    dir_1 + "run_long_tprim-4.9000_continue",
    dir_2 + "run_tprim-4.9000",
    dir_3 + "run_tprim-4.9000",
    dir_4 + "run_tprim-4.9000",
]
labels = [
    r"CBC ($\nu=0$ HR) $R/L_T=4.9$",
    r"CBC ($\nu=0$) $R/L_T=4.9$",
    None,
    None,
    r"$(\nu=10^{-3})$",
    r"$(\nu=3\cdot 10^{-4})$",
    r"$(\nu=10^{-4})$",
]
colors = ["k", "crimson", "crimson", "crimson", "forestgreen", "mediumblue", "purple"]
ylim = [1e-3, 1e2]
# NOTE: the original embedded "COLLISION SCAN"/"COLLISION SCAN 2" blocks in
# plot_flux_time.py both used figname_add = "_nu-var" -- harmless there
# since only one block was ever active at a time, but would collide now
# that both exist as separate configs runnable independently. Renamed here.
figname_add = "_nu-var2"
