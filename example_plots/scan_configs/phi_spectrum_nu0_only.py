"""Single nu=0 run, phi(k) spectra across a tprim sweep (the mode active
before this migration). Run with:
    python ../plot_phi_spectrum_compare.py phi_spectrum_nu0_only.py
"""

tprim_vals = [3.85, 4.2, 4.9, 5.6, 5.95, 6.3, 7.35, 8.4]

_dir_1 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"

dirnames = [[_dir_1 + "run_tprim-%.4f" % t] for t in tprim_vals]

time_avg = 500
load_from_file = True
