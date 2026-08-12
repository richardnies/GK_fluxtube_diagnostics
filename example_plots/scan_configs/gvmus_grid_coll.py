"""Collisionality scan grid for plot_gvmus_all_dirs.py (the scan_type
active before this migration) -- run with:
    python ../plot_gvmus_all_dirs.py gvmus_grid_coll.py
"""

_tprims = ["3.8500", "4.2000", "4.9000", "5.6000", "5.9500", "6.3000", "8.4000"]
_nu_dirs = [
    "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0",
    "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001",
    "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001",
]

dirnames = [[f"{nu_dir}/run_tprim-{tprim}" for tprim in _tprims] for nu_dir in _nu_dirs]

col_titles = [r"$R/L_T = 3.85$", r"$R/L_T = 4.2$", r"$R/L_T = 4.9$", r"$R/L_T = 5.6$", r"$R/L_T = 5.95$", r"$R/L_T = 6.3$", r"$R/L_T = 8.4$"]
row_titles = [r"$\nu_{ii}R/v_{Ti} = 0$", r"$\nu_{ii}R/v_{Ti} = 10^{-4}$", r"$\nu_{ii}R/v_{Ti} = 10^{-3}$"]
figname = "fig_gvmus_all_dirs_zonal_dtavg-300_kxmin-0.00_kxmax-0.20_coll.pdf"
