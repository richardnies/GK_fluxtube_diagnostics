"""Shared config for every single-run diagnostic pointed at this run
directory. Run any single-run driver from the BASE directory
(stella_minimal_scan/, NOT from inside this run directory), e.g.:
    cd stella_minimal_scan
    python3 ../example_plots/plot_zonal_shear_diagnostic.py run_tprim-4.2000/run_config.py
    python3 ../example_plots/movie_quantities_x.py run_tprim-4.2000/run_config.py
    python3 ../example_plots/plot_RH_phi_E_P_t_all_kx.py run_tprim-4.2000/run_config.py
    ... (plot_fluxes.py, plot_correlation_func.py, plot_quantities_over_zed.py,
    movie_gvmus_t.py, movie_gvmus_Z-NZ.py, movie_gvmus_Z-NZ_kxs.py,
    movie_gzvs_Z-NZ.py, movie_quantities_x_zed.py, movie_quantity_real_space.py)

See run_tprim-6.7000/run_config.py's docstring for the sibling-run
equivalent and for why this is one shared file instead of one config per
script. This run's data directory has been replaced multiple times over
the course of development; time_min/time_max below are re-derived from
this run's own flux trace (run.get_fluxes_over_time()) each time, not
copied from the sibling run, since the transient/saturation timing
differs between replacements.

Latest replacement (higher resolution: nx=8,ny=8,nzed=8,nvgrid=8,nmu=8,
vs. the sibling's lower-resolution nx=4,ny=4,nzed=4,nvgrid=4,nmu=4 --
kept intentionally asymmetric so higher-resolution-only diagnostics like
plot_phi_spectrum_compare.py have a real multi-mode kx/ky grid to plot a
spectrum over) has a somewhat shorter run (t=0..52 vs. the sibling's
t=0..62): heat flux spikes negative around t=18-30 (a transient, not
a steady-state feature), then decays and settles into a slowly-decaying
quasi-steady plateau from about t=39 onward. time_min/time_max/time_avg
below bracket that plateau.
"""

dirname = "run_tprim-4.2000"
filename = "example"
code = "stella"

time_min = 39
time_max = 52
time_avg = 13
kx_max = 0.3

# Only movie_gvmus_Z-NZ_kxs.py needs these (required for it specifically).
kx_mins = [0]
kx_maxs = [0.2]

# Only plot_quantities_over_zed.py needs this (forwarded as **kwargs to
# plot_quantities_over_zed). kx_idx_phi/ky_idx_phi=1 picks the first
# nonzero (kx,ky) mode for phi -- at the (0,0) zonal mode phi/omega_s_k
# are identically 0 or 0/0 here, which silently plots nothing. Gamma0/B
# are geometry-only and always evaluated at (kx_idx=0, ky_idx=0)
# regardless of this setting (get_Gamma0/get_omega_s_k don't expose an
# index override through this driver) -- harmless for Gamma0 (=1 at
# kperp=0, a meaningful baseline) but means plot_omega_s_k would still
# come out all-NaN even with this override, so it's left off here.
kwargs = dict(plot_phi=True, plot_B=True, plot_Gamma0=True, kx_idx_phi=1, ky_idx_phi=1)
