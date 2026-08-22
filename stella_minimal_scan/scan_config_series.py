"""Shared config for the three multi-run diagnostics that plot one series
(or grid row) per outer `dirnames` entry, treating this directory's two
runs (run_tprim-4.2000, run_tprim-6.7000) as a single series. Run from
inside this directory:
    cd stella_minimal_scan
    python3 ../example_plots/plot_ERH_Ephi.py scan_config_series.py
    python3 ../example_plots/plot_param_scan_Dimits.py scan_config_series.py
    python3 ../example_plots/plot_contour_phi_vs_t_zed.py scan_config_series.py

(compare_growth_rates.py uses scan_config_growth_rates.py instead, even
though it's also a "one series per outer dirnames entry" script -- its
actual scan axis here is tprim itself (one series per R/L_T, each a
SINGLE run), the transpose of this file's "one series containing both
R/L_T runs" -- see that file's own docstring for why combining both runs
into one series crashes it.)

`dirnames` is a nested list, dirnames[i_series] = flat list of run
directories in that series; for this minimal scan there's only one series
(both runs at the single vnew=0.0001 collisionality this scan has), so
`dirnames` has one outer entry containing both run directories -- the
same shape plot_contour_phi_vs_t_zed.py's grid convention
(`dirnames[row][col]`) needs too, since a 1-row grid IS a 1-series list.
tprim is read directly from each run's own netCDF output by every one of
these functions, not supplied separately.

Kept separate from scan_config_grid.py (gvmus_all_dirs.py/
mean_quantities_x.py/mean_quantities_x_zed.py, which need this same
nested-dirnames shape and value) only because mean_quantities_x_zed.py's
own `labels` field means something different there (per-quantity panel
labels, not per-run labels) -- merging would silently collide under that
name. Kept separate from scan_config_flux_coll.py (flux_coll.py) because
its nested dirnames is transposed the other way round (series=tprim,
member=nu) -- see that file's own docstring.
"""

filename = "example"

time_min = 10
time_max = 250
time_avg = 20

# --- the scan itself: single source of truth for every block below ---
tprim_vals = [4.2, 6.7]
dirnames = [["run_tprim-%.4f" % t for t in tprim_vals]]  # one series (vnew=0.0001) containing both runs

# --- ERH_Ephi.py, param_scan_Dimits.py ---
labels = ["vnew-0.0001"]
aLT_lin_vals = [0.0]
xlim = None

# --- contour_phi_vs_t_zed.py (fails here -- see its own module docstring:
# plot_contour_phi_zed_t is a confirmed pre-existing bug, predates this
# migration) ---
row_titles = ["vnew-0.0001"]
col_titles = ["tprim-%.1f" % t for t in tprim_vals]
