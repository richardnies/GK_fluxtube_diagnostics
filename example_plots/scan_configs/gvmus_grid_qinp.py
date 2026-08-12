"""Qinp scan grid for plot_gvmus_all_dirs.py -- run with:
    python ../plot_gvmus_all_dirs.py gvmus_grid_qinp.py
"""

dirnames = [
    [
        "",
        "../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-2.2500/",
        "../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-2.5000/",
        "../2026-01-20_ES_ITG_Dimits/2026-02-02_scan_qinp-1.0_shat-0.8_rmaj-2.778_rhoc-0.5_fprim-0.8_vnew-0/run_tprim-3.0000/",
    ],
    [
        "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000",
        "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000",
        "",
        "",
    ],
    [
        "2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000",
        "2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-6.3000",
        "",
        "",
    ],
]
col_titles = [r"$R/L_T = 4.9$", r"$R/L_T = 6.3$", r"$R/L_T = 7.0$", r"$R/L_T = 8.4$"]
row_titles = [r"$q=1.0$", r"$q= 1.4$", r"$q=2.8$"]
figname = "fig_gvmus_all_dirs_zonal_dtavg-300_kxmin-0.00_kxmax-0.20_qinp.pdf"
