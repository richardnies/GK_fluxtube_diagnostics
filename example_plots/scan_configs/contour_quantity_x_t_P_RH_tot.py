"""P_RH_tot(x, t) contour grid (the settings active before this migration).
Run with:
    python ../plot_contour_quantity_vs_t_x.py contour_quantity_x_t_P_RH_tot.py
"""

dirname_base = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0"
dirname_pattern = "run*00"

quantity = "P_RH_tot"
kx_order = 0
time_min = 0
time_max = 5000
normalise = False
only_zonal = False
remove_zonal = False
y_val = None
logarithmic = False
cmap = "coolwarm"
vmin = "symm"
vmax = "last"
