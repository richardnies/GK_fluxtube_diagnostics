"""Config for plot_compare_growth_rates.py -- run with:
    python ../plot_compare_growth_rates.py growth_rates_default.py
"""

akyminmax_vals = [0.1]
nfield_periods_vals = [20, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800]
filename = "precise_QA"

dirnames = [["run_akyminmax-%.4f_nfield_periods-%.4f" % (aky, nfp) for aky in akyminmax_vals] for nfp in nfield_periods_vals]
series_labels = [r"$Nfp=%i$" % nfp for nfp in nfield_periods_vals]
