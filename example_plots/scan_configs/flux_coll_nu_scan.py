"""Collisionality scan for plot_flux_coll.py (the scan active before this
migration). Run with:
    python ../plot_flux_coll.py flux_coll_nu_scan.py
"""

_base = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-"

_dirs_nu = [
    _base + "0/",
    _base + "1e-6/",
    _base + "1e-5/",
    _base + "3e-5/",
    _base + "0.0001/",
    _base + "3e-4/",
    _base + "0.001/",
    _base + "3e-3/",
    _base + "0.01/",
]
tprim_vals = [3.85, 4.2, 4.9, 5.6, 5.95, 6.3, 7.35, 8.4]

dirnames = [[d + "run_tprim-%.4f" % t for d in _dirs_nu] for t in tprim_vals]
time_avg = 200
