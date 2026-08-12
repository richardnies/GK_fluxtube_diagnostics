"""Config for plot_phiZ_TS_qkappa2.py -- run with:
    python ../plot_phiZ_TS_qkappa2.py phiZ_TS_qkappa2_default.py
"""

dirnames = ["2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"]
tprim_vals = [3.85, 4.9, 6.4]
qinp = 1.4
aspect_ratio = 1
kxmin = 0.3
kxmax = 1e4
fac_rescale = 2 * 2.8**2  # fac_rescale_phys=True in the original
