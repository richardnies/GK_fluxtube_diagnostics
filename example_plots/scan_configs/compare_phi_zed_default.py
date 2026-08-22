"""Config for plot_compare_phi_zed.py -- run with:
    python ../plot_compare_phi_zed.py compare_phi_zed_default.py
"""

akyminmax_vals = [0.1]
nfield_periods_vals = [200, 250, 300, 400]
filename = "precise_QA"

dirnames = ["run_akyminmax-%.4f_nfield_periods-%.4f" % (aky, nfp) for aky in akyminmax_vals for nfp in nfield_periods_vals]
labels = [r"nfp = %.1f" % nfp for aky in akyminmax_vals for nfp in nfield_periods_vals]
