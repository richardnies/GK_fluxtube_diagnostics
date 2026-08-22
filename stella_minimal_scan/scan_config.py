"""Shared config for every multi-run (scan-comparison) diagnostic that
takes a flat list of run directories, pointed at this directory's two
runs (run_tprim-4.2000, run_tprim-6.7000). Run any of these drivers from
inside this directory, e.g.:
    cd stella_minimal_scan
    python3 ../example_plots/plot_flux_time.py scan_config.py
    python3 ../example_plots/plot_geometry_compare_flux_tubes.py scan_config.py
    python3 ../example_plots/plot_contour_quantity_vs_kx_omega.py scan_config.py
    python3 ../example_plots/plot_contour_quantity_vs_t_x.py scan_config.py
    python3 ../example_plots/plot_zonal_upar_vE_kx.py scan_config.py
    python3 ../example_plots/plot_compare_phi_zed.py scan_config.py

(gvmus_all_dirs.py/mean_quantities_x.py/mean_quantities_x_zed.py use
scan_config_grid.py instead -- their dirnames is nested `dirnames[row][col]`,
not a flat list, and mean_quantities_x_zed.py's own `labels` field means
something different -- per-quantity panel labels, not per-run labels --
which would collide under the same name if merged here. ERH_Ephi.py/
param_scan_Dimits.py/compare_growth_rates.py/contour_phi_vs_t_zed.py use
scan_config_series.py (nested dirnames, one series/group). flux_coll.py
uses scan_config_flux_coll.py (nested the other way round: series=tprim,
member=nu). See each of those files' own docstring for why they're kept
separate rather than merged here.)

One file instead of one config per script (see run_config.py's docstring
for the run_tprim-4.2000/6.7000 single-run equivalent of this rationale),
so every diagnostic here shares the same time_min/time_max/time_avg -- see
the naming-inconsistency-glossary note in stella_diagnostics/__init__.py.

Every script below takes directories the same way: `dirnames`, a flat list
of run directories, declared once as the single source of truth right
after the time window (along with `labels`/`colors`, one entry per
dirname). tprim itself is never supplied here -- every consuming function
reads it directly from each run's own netCDF output.

`time_avg` (20 here) is shared only by the scripts already confirmed safe
for it (see run_config.py's docstring for the pre-existing
get_quantity_zed_x_y bug this avoids).

time_min/time_max (10/60) previously differed per script -- some used the
same post-transient window as everything else, others (kx_omega,
contour_quantity_vs_t_x) defaulted to an unbounded full-run range
(0..1e10/1e5) since a contour plot's whole point is showing time
evolution. Consolidated to the same window as every other diagnostic
here per the "same parameter for all" decision -- both contour scripts
still show plenty of evolution within 10..250 (this run's full range is
0..67.66) and now show the same physically-relevant saturated window as
every other plot pointed at this scan.
"""

filename = "example"

time_min = 10
time_max = 250
time_avg = 20

# --- the scan itself: single source of truth for every block below ---
tprim_vals = [4.2, 6.7]
dirnames = ["run_tprim-%.4f" % t for t in tprim_vals]
labels = [r"$R/L_T=%.1f$" % t for t in tprim_vals]
colors = ["mediumblue", "crimson"]

# --- flux_time.py, geometry_compare_flux_tubes.py, kx_omega.py ---
ylim = [1e-25, 1e3]
figname_add = "_minimal_scan"
quantity = "phi"
kwargs = {}

# --- contour_quantity_vs_t_x.py ---
kx_order = 0
only_zonal = True

# --- plot_zonal_upar_vE_kx.py ---
kx_min = 0.0
kx_max = 0.5

# --- compare_phi_zed.py (fails here -- see its own module docstring: this
# driver hardcodes zed_times_nfield_periods=True, which needs a VMEC
# ".vmec.geo" file; these runs only have Miller geometry) ---
