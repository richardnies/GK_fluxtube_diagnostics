"""Config for plot_quantities_over_zed.py -- run with:
    python ../plot_quantities_over_zed.py quantities_over_zed_default.py
"""

dirname = "run_akyminmax-1.0000_nfield_periods-100.0000/precise_QA"
kwargs = {
    "plot_phi": True,
    "plot_B": True,
    "plot_Gamma0": True,
    "plot_omega_s_k": True,
    "norm_factor_omega_s_k": 1,
}
ylim = [-1, 2.5]
