"""Shared config for every single-run diagnostic pointed at this run
directory. Run any single-run driver from the BASE directory
(stella_minimal_scan/, NOT from inside this run directory), e.g.:
    cd stella_minimal_scan
    python3 ../example_plots/plot_zonal_shear_diagnostic.py run_tprim-6.7000/run_config.py
    python3 ../example_plots/movie_quantities_x.py run_tprim-6.7000/run_config.py
    python3 ../example_plots/plot_RH_phi_E_P_t_all_kx.py run_tprim-6.7000/run_config.py
    ... (plot_fluxes.py, plot_correlation_func.py, plot_quantities_over_zed.py,
    movie_gvmus_t.py, movie_gvmus_Z-NZ.py, movie_gvmus_Z-NZ_kxs.py,
    movie_gzvs_Z-NZ.py, movie_quantities_x_zed.py, movie_quantity_real_space.py)

See run_tprim-4.2000/run_config.py's docstring for why this is one shared
file instead of one config per script and why it's run from the base
directory rather than from inside this run directory (so output never
ends up mixed in with this run's own data files). time_avg matches the
sibling run's value for comparability across the two, not because this
run's own dynamics were re-examined here.
"""

dirname = "run_tprim-6.7000"
filename = "example"
code = "stella"

time_min = 10
time_max = 60
time_avg = 20
figname_add = ""
kx_max = 0.3

# Only movie_gvmus_Z-NZ_kxs.py needs these (required for it specifically).
kx_mins = [0]
kx_maxs = [0.2]
