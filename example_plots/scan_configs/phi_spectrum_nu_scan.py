"""nu=0 / 1e-4 / 1e-3 comparison, phi(k) spectra across a tprim sweep. Run with:
    python ../plot_phi_spectrum_compare.py phi_spectrum_nu_scan.py
"""

tprim_vals = [3.85, 4.2, 4.9, 5.6, 5.95, 6.3, 7.35, 8.4]

_base = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-"
_dir_1 = _base + "0/"
_dir_2 = _base + "0.001/"
_dir_4 = _base + "0.0001/"

dirnames = [[_dir_1 + "run_tprim-%.4f" % t, _dir_4 + "run_tprim-%.4f" % t, _dir_2 + "run_tprim-%.4f" % t] for t in tprim_vals]
labels = [[r"CBC ($\nu=0$)", r"CBC ($\nu=10^{-4}$)", r"CBC ($\nu=10^{-3}$)"] for t in tprim_vals]

time_avg = 500
load_from_file = True
