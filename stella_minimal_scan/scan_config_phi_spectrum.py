"""Config for plot_phi_spectrum_compare.py, pointed at this directory's
higher-resolution run_tprim-4.2000 (see run_tprim-4.2000/run_config.py's
docstring for why it's higher-resolution than its sibling run --
specifically so this diagnostic has a real multi-mode kx/ky grid to plot
a spectrum over; run_tprim-6.7000 is left low-resolution and not used
here). Run with:
    cd stella_minimal_scan
    python3 ../example_plots/plot_phi_spectrum_compare.py scan_config_phi_spectrum.py

Kept separate from scan_config.py/scan_config_grid.py/scan_config_series.py
(not merged) since this script's `dirnames` is a list of per-output-figure
groups (dirnames[i_group] = flat list of run directories in that figure),
a third nesting meaning distinct from either of those files' conventions.
"""

filename = "example"
code = "stella"

# Trailing window matching run_tprim-4.2000/run_config.py's own
# time_avg (the settled quasi-steady plateau, t=39..52).
time_avg = 13

# --- the scan itself: single source of truth, as in scan_config.py ---
# Deliberately just the higher-resolution run (run_tprim-6.7000 excluded,
# see module docstring): one figure group containing that one run. tprim
# is read directly from its own netCDF output, not supplied separately.
dirnames = [["run_tprim-4.2000"]]
