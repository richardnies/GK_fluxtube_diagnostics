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
field name expecting a flat list -- a real, pre-existing structural
incompatibility under one shared field name, not a naming choice this
consolidation can paper over without changing those scripts' own config
contract (out of scope here, see the naming-inconsistency-glossary note
in stella_diagnostics/__init__.py).

`time_avg` is read by plot_mean_quantities_x.py (forwarded to
get_quantities_x_tavg) but not by plot_mean_quantities_x_zed.py (which
only uses time_idx_step-based framing, so this field is simply unread
there, not an error). A pre-existing get_quantity_zed_x_y bug used to
break plot_mean_quantities_x.py's RH composite quantities
(Pi_RH_even/Pi_RH_odd/Pi_RH_NL, part of its default quantities_plot
list) whenever time_avg was engaged -- fixed; safe to share here now.
"""

filename = "example"

time_min = 10
time_max = 50
time_avg = 20

dirnames = [["run_tprim-4.2000", "run_tprim-6.7000"]]
row_titles = ["vnew-0.0001"]
col_titles = ["tprim-4.2", "tprim-6.7"]

kx_min = None
kx_max = None
quantities_plot = "P_RH"  # mean_quantities_x.py only -- edit to "Q", "P_RH_scatter", "P_RH", "Pi_RH", or "Z_profiles"
time_idx_step = 5
