"""Shat scan grid for plot_gvmus_all_dirs.py -- run with:
    python ../plot_gvmus_all_dirs.py gvmus_grid_shat.py
"""

dirnames = [
    [
        "",
        "",
        "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-4.9000",
        "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/run_tprim-5.6000",
    ],
    [
        "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.2500/",
        "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.5000/",
        "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.7500/",
        "../2026-01-20_ES_ITG_Dimits/2026-03-07_qinp-1.4_shat-0.32_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-2.0000/",
    ],
    [
        "",
        "",
        "../2026-01-20_ES_ITG_Dimits/2026-05-05_qinp-1.4_shat-0.16_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-1.7500/",
        "../2026-01-20_ES_ITG_Dimits/2026-05-05_qinp-1.4_shat-0.16_rmaj-2.778_rhoc-0.50_fprim-0.8_vnew-0/run_tprim-2.0000/",
    ],
]
col_titles = [r"$R/L_T = 3.5$", r"$R/L_T = 4.2$", r"$R/L_T = 4.9$", r"$R/L_T = 5.6$"]
row_titles = [r"$\hat s = 0.8$", r"$\hat s = 0.32$", r"$\hat s = 0.16$"]
figname = "fig_gvmus_all_dirs_zonal_dtavg-300_kxmin-0.00_kxmax-0.20_shat.pdf"
