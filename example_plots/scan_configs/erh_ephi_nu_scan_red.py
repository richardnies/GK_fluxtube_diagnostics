"""Reduced collisionality scan for plot_ERH_Ephi.py (the scan active before
this migration). Run with:
    python ../plot_ERH_Ephi.py erh_ephi_nu_scan_red.py
"""
import numpy as np

from stella_diagnostics.physics.gradients import get_aLT_lin_analytic

_base = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-"

base_dirs = [_base + "0", _base + "0.0001", _base + "0.001"]
base_labels = [r"$\nu^*=0$", r"$\nu^*=10^{-4}$", r"$\nu^*=10^{-3}$"]
aLT_lin_vals = get_aLT_lin_analytic(
    rhoc=np.array([0.18, 0.18, 0.18]),
    q=np.array([1.4, 1.4, 1.4]),
    shat=np.array([0.8, 0.8, 0.8]),
)
time_avg = 800
