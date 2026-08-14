"""Single nu=0 run, phi(k) spectra across a tprim sweep (the mode active
before this migration). Run with:
    python ../plot_phi_spectrum_compare.py phi_spectrum_nu0_only.py
"""

dirname_mode = "nu0_only"
tprim_vals = [3.85, 4.2, 4.9, 5.6, 5.95, 6.3, 7.35, 8.4]

dirs = {
    "dir_0": "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0_higher-vel-res/",
    "dir_1": "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/",
    "dir_2": "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.001/",
    "dir_3": "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-3e-4/",
    "dir_4": "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/",
    "dir_5": "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-1e-5/",
    "dir_q_0": "2026-06-26_scan_qinp-0.7_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/",
    "dir_q_1": "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/",
    "dir_q_2": "2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/",
}

time_avg = 500
load_from_file = True
