"""Config for plot_phi_spectrum_compare.py -- run with:
    python ../example_plots/plot_phi_spectrum_compare.py scan_phi_spectrum_config.py

NOTE: fails here with IndexError inside the pre-existing (unmodified)
stella_diagnostics/scan/spectrum_scan.py::plot_phi_k_spectrum -- this
run's ky grid only has 2 points, too coarse for that function's
scale_kmin/rescale logic (k[1]-k[0] on an array that ends up with fewer
elements than expected after slicing). A data-resolution limitation of
this minimal example, not a config-driven-migration bug: the same
function, called the same way, would need a run with a less coarse
k-grid to actually exercise this code path.
"""
dirname_mode = "nu0_only"
tprim_vals = [4.2, 6.7]
dirs = {"dir_1": "./"}
filename = "example"
time_avg = None
load_from_file = False
