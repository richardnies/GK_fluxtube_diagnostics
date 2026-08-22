"""Shared config for the three multi-run diagnostics that lay runs out on
a row/col grid (`dirnames[row][col]`, `row_titles`, `col_titles`). Run
from inside this directory:
    cd stella_minimal_scan
    python3 ../example_plots/plot_gvmus_all_dirs.py scan_config_grid.py
    python3 ../example_plots/plot_mean_quantities_x.py scan_config_grid.py
    python3 ../example_plots/plot_mean_quantities_x_zed.py scan_config_grid.py

Kept separate from scan_config.py (rather than merged) because
gvmus_all_dirs.py/mean_quantities_x.py/mean_quantities_x_zed.py all read
`config.dirnames` expecting a NESTED 2D list (`dirnames[row][col]`),
while flux_time.py/geometry_compare_flux_tubes.py/
plot_contour_quantity_vs_kx_omega.py (in scan_config.py) read the same
field name expecting a flat list.

Also kept separate from scan_config_series.py (ERH_Ephi.py/
param_scan_Dimits.py/compare_growth_rates.py/contour_phi_vs_t_zed.py,
which share the exact same nested-dirnames shape and value used here)
specifically because mean_quantities_x_zed.py's own `labels` field means
something different -- a fixed list of per-quantity panel labels, not
per-run labels -- which would silently collide under the same name if
every nested-dirnames script were merged into one file.

`time_avg` is read by plot_mean_quantities_x.py (forwarded to
get_quantities_x_tavg) but not by plot_mean_quantities_x_zed.py (which
only uses time_idx_step-based framing, so this field is simply unread
there, not an error). A pre-existing get_quantity_zed_x_y bug used to
break plot_mean_quantities_x.py's RH composite quantities
(Pi_RH_even/Pi_RH_odd/Pi_RH_NL, part of its default quantities_plot
list) whenever time_avg was engaged -- fixed; safe to share here now.

As in scan_config.py, the scan itself is declared once in `tprim_vals`
right after the time window, and `dirnames`/`col_titles` are built from
it -- adding a third run means editing `tprim_vals` (and `dirnames`'s
row layout) in one place.
"""

filename = "example"

time_min = 10
time_max = 50
time_avg = 20

# --- the scan itself: single source of truth for the grid below ---
tprim_vals = [4.2, 6.7]
dirnames = [["run_tprim-%.4f" % t for t in tprim_vals]]  # one row (nu=0.0001), one column per tprim_vals entry
row_titles = ["vnew-0.0001"]
col_titles = ["tprim-%.1f" % t for t in tprim_vals]

kx_min = None
kx_max = None
quantities_plot = "P_RH"  # mean_quantities_x.py only -- edit to "Q", "P_RH_scatter", "P_RH", "Pi_RH", or "Z_profiles"
time_idx_step = 5
