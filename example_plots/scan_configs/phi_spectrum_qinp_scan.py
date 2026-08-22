"""q=0.7/1.4/2.8 comparison, phi(k) spectra across a tprim sweep. Run with:
    python ../plot_phi_spectrum_compare.py phi_spectrum_qinp_scan.py
"""

tprim_vals = [3.85, 4.2, 4.9, 5.6, 5.95, 6.3, 7.35, 8.4]

_dir_q_0 = "2026-06-26_scan_qinp-0.7_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
_dir_q_1 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
_dir_q_2 = "2026-06-26_scan_qinp-2.8_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"

dirnames = [[_dir_q_0 + "run_tprim-%.4f" % t, _dir_q_1 + "run_tprim-%.4f" % t, _dir_q_2 + "run_tprim-%.4f" % t] for t in tprim_vals]
labels = [[r"$q=0.7$", r"$q=1.4$", r"$q=2.8$"] for t in tprim_vals]

time_avg = 500
load_from_file = True
