"""Config for plot_compare_growth_rates.py, pointed at this directory's
two runs (run_tprim-4.2000, run_tprim-6.7000). Run with:
    cd stella_minimal_scan
    python3 ../example_plots/plot_compare_growth_rates.py scan_config_growth_rates.py

Kept separate from scan_config_series.py even though both are "one series
per outer dirnames entry": that file treats both runs as ONE series
(a single vnew=0.0001 physical regime containing two R/L_T points), which
is what ERH_Ephi.py/param_scan_Dimits.py/contour_phi_vs_t_zed.py want.
plot_compare_growth_rates.py instead calls RunCollection.plot_omega_ky
once per series and expects every run *within* a series to share the same
ky/omega structure (it's meant for an akyminmax convergence sweep at fixed
R/L_T) -- combining both differently-configured R/L_T runs into one such
call crashes with a ValueError building a ragged omega-vs-time array. This
minimal scan has no real akyminmax sweep, so each R/L_T run instead gets
its own single-run series here, matching how the original driver's
akyminmax_vals=[""] (a single dummy value) forced the same one-run-per-
series shape.
"""

filename = "example"
figname_add = ""

# --- the scan itself: single source of truth ---
tprim_vals = [4.2, 6.7]
dirnames = [["run_tprim-%.4f" % t] for t in tprim_vals]  # one series per tprim, one run each
series_labels = [r"$R/L_T=%.1f$" % t for t in tprim_vals]
