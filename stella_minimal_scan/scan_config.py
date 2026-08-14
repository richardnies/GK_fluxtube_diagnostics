"""Shared config for every multi-run (scan-comparison) diagnostic pointed
at this directory's two runs (run_tprim-4.2000, run_tprim-6.7000). Run
any multi-run driver from inside this directory, e.g.:
    cd stella_minimal_scan
    python3 ../example_plots/plot_flux_coll.py scan_config.py
    python3 ../example_plots/plot_ERH_Ephi.py scan_config.py
    python3 ../example_plots/plot_flux_time.py scan_config.py
    python3 ../example_plots/plot_geometry_compare_flux_tubes.py scan_config.py
    python3 ../example_plots/plot_contour_quantity_vs_kx_omega.py scan_config.py
    python3 ../example_plots/plot_contour_quantity_vs_t_x.py scan_config.py
    python3 ../example_plots/plot_param_scan_Dimits.py scan_config.py

(gvmus_all_dirs.py, mean_quantities_x.py, mean_quantities_x_zed.py use
scan_config_grid.py instead -- see that file's docstring for why.)

One file instead of one config per script (see run_config.py's docstring
for the run_tprim-4.2000/6.7000 single-run equivalent of this rationale),
so every multi-run diagnostic here shares the same time_min/time_max/
time_avg -- see the naming-inconsistency-glossary note in
stella_diagnostics/__init__.py.

Each script names its own dirname-list shape differently (a pre-existing
API difference between scripts, not something this consolidation
unifies -- see stella_diagnostics/__init__.py): `base_dirs` (ERH_Ephi,
param_scan_Dimits), `dirs_nu`/`vals_nu`/`tprim_vals` (flux_coll),
`dirnames` as a flat list (flux_time, geometry_compare, kx_omega), or
`dirname_base`+`dirname_pattern` glob matching (contour_quantity_vs_t_x).
All those variants are defined below so `load_scan_config` (which
tolerates fields a given script doesn't read) can serve every script from
this one file.

`time_avg` (20 here) is shared only by the scripts already confirmed safe
for it (ERH_Ephi, flux_coll, param_scan_Dimits -- see run_config.py's
docstring for the pre-existing get_quantity_zed_x_y bug this avoids).

time_min/time_max (10/60) previously differed per script -- some used the
same post-transient window as everything else, others (kx_omega,
contour_quantity_vs_t_x) defaulted to an unbounded full-run range
(0..1e10/1e5) since a contour plot's whole point is showing time
evolution. Consolidated to the same window as every other diagnostic
here per the "same parameter for all" decision -- both contour scripts
still show plenty of evolution within 10..60 (this run's full range is
0..67.66) and now show the same physically-relevant saturated window as
every other plot pointed at this scan.
"""

filename = "example"

time_min = 10
time_max = 250
time_avg = 20

# --- ERH_Ephi.py, param_scan_Dimits.py ---
base_dirs = ["."]
base_labels = ["vnew-0.0001"]
aLT_lin_vals = [0.0]
xlim = None

# --- flux_coll.py ---
dirs_nu = ["."]
vals_nu = [0.0001]
tprim_vals = [4.2, 6.7]

# --- flux_time.py, geometry_compare_flux_tubes.py, kx_omega.py (flat dirnames) ---
dirnames = [
    "run_tprim-4.2000",
    "run_tprim-6.7000",
]
labels = [r"$R/L_T=4.2$", r"$R/L_T=6.7$"]
colors = ["mediumblue", "crimson"]
ylim = [1e-25, 1e3]
figname_add = "_minimal_scan"
quantity = "phi"
kwargs = {}

# --- contour_quantity_vs_t_x.py (glob-based dirname matching) ---
dirname_base = "."
dirname_pattern = "run_tprim*"
kx_order = 0
only_zonal = True
