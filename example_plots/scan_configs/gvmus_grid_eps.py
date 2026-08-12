"""Eps (inverse aspect ratio) scan grid for plot_gvmus_all_dirs.py -- run with:
    python ../plot_gvmus_all_dirs.py gvmus_grid_eps.py
"""

dirnames = [
    [
        "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0/run_tprim-4.9000",
        "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.045_fprim-2.2_vnew-0/run_tprim-6.3000",
    ],
    [
        "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000",
        "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000",
    ],
    [
        "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0/run_tprim-4.9000",
        "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.36_fprim-2.2_vnew-0/run_tprim-6.3000",
    ],
]
col_titles = [r"$R/L_T = 4.9$", r"$R/L_T = 6.3$"]
row_titles = [r"$\epsilon = 0.045$", r"$\epsilon = 0.18$", r"$\epsilon = 0.36$"]
figname = "fig_gvmus_all_dirs_zonal_dtavg-300_kxmin-0.00_kxmax-0.20_eps.pdf"
