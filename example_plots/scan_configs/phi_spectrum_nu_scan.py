"""nu=0 / 1e-4 / 1e-3 comparison, phi(k) spectra across a tprim sweep. Run with:
    python ../plot_phi_spectrum_compare.py phi_spectrum_nu_scan.py
"""

dirname_mode = "nu_scan"
tprim_vals = [3.85, 4.2, 4.9, 5.6, 5.95, 6.3, 7.35, 8.4]

dirs = {
    "dir_1": "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/",
    "dir_2": "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/",
    "dir_4": "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/",
}

time_avg = 500
load_from_file = True
