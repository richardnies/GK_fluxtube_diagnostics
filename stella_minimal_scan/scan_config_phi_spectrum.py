"""Config for plot_phi_spectrum_compare.py, pointed at this directory's
higher-resolution run_tprim-4.2000 (see run_tprim-4.2000/run_config.py's
docstring for why it's higher-resolution than its sibling run --
specifically so this diagnostic has a real multi-mode kx/ky grid to plot
a spectrum over; run_tprim-6.7000 is left low-resolution and not used
here). Run with:
    cd stella_minimal_scan
    python3 ../example_plots/plot_phi_spectrum_compare.py scan_config_phi_spectrum.py

Kept separate from scan_config.py/scan_config_grid.py (not merged) since
this script's `dirs`+`dirname_mode` config shape is structurally
different from either -- see stella_diagnostics/__init__.py's
naming-inconsistency-glossary note on why cross-script config-shape
mismatches aren't unified under one field name.
"""

dirname_mode = "nu0_only"
tprim_vals = [4.2]
dirs = {"dir_1": ""}
filename = "example"
code = "stella"

# Trailing window matching run_tprim-4.2000/run_config.py's own
# time_avg (the settled quasi-steady plateau, t=39..52).
time_avg = 13
