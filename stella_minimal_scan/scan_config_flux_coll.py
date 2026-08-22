"""Config for plot_flux_coll.py, pointed at this directory's two runs
(run_tprim-4.2000, run_tprim-6.7000), both at the single vnew=0.0001
collisionality this minimal scan has. Run with:
    cd stella_minimal_scan
    python3 ../example_plots/plot_flux_coll.py scan_config_flux_coll.py

Kept separate from scan_config_series.py: plot_qflx_vs_nu_scan's
`dirnames` is nested the other way round -- dirnames[i_tprim] = flat list
of collisionality-swept run directories (one series per R/L_T, x-axis =
nu_ii, both tprim and nu_ii read directly from each run's own netCDF
output) -- versus scan_config_series.py's dirnames[i_series] = flat list
of R/L_T-swept run directories (one series per physical regime, x-axis =
R/L_T). A real scan with more than one collisionality value would have
more than one directory per inner list here; this minimal scan only has
one (vnew=0.0001), so each series is a single-run list.
"""

filename = "example"
time_avg = 20
figname_add = ""

# --- the scan itself: single source of truth ---
tprim_vals = [4.2, 6.7]
dirnames = [["run_tprim-%.4f" % t] for t in tprim_vals]  # one series per tprim, one run each (single vnew=0.0001 point)
