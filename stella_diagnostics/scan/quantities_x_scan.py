"""Time-averaged (x) / (zed, x) profiles of real-space quantities, computed
by accumulating a per-time-idx quantity over an explicit frame-index range.

Extracted from example_plots/movie_quantities_x.py, plot_mean_quantities_x.py,
movie_quantities_x_zed.py, and plot_mean_quantities_x_zed.py. The originals
computed this accumulation inline inside the movie-frame-rendering loop and
saved the result to ``data_<name>.dat`` files that a second script
(plot_mean_quantities_x*.py) would separately np.loadtxt back in -- both a
hand-rolled disk cache (now redundant with @cached) and a source of two
confirmed bugs this module's structure removes rather than patches:

- movie_quantities_x.py's active ``datanames`` list contained two entries
  named "tmp" (a leftover placeholder), so the second computed quantity's
  time average silently overwrote the first's ``data_tmp.dat``. With a
  dict-keyed result and an explicit uniqueness assertion, this collision is
  now impossible instead of silently wrong.
- Both movie scripts only accumulated dt_sum/the running average inside the
  same loop that rendered (or skipped, if already on disk) each frame PNG,
  so a rerun with ``rerun_all=False`` after frames already existed produced
  either a stale tavg figure (movie_quantities_x.py never rewrote it) or a
  ``dt_sum == 0`` divide-by-zero (movie_quantities_x_zed.py). With
  @cached, the tavg computation is decoupled from which frame PNGs already
  exist and always runs to completion once per distinct set of parameters.
"""

import numpy as np

from stella_diagnostics.io.cache import cached
from stella_diagnostics.plotting.zed_plots import get_quantity_x_zed
from stella_diagnostics.quantities.realspace import get_quantity_x

# The full set of (quantity, kx_order, mult, mult_zed) -> dataname mappings
# historically computed by movie_quantities_x.py, reconstructed from the
# union of its several mutually-exclusive (hand-commented) `quantities`
# blocks (confirmed via git history) -- since get_quantities_x_tavg is
# @cached, there's no more need to hand-run the script multiple times with
# a different block uncommented to populate every data_<name>.dat file that
# plot_mean_quantities_x.py's various quantities_plot modes read; one call
# with this full list computes (and caches) all of them at once.
FULL_QUANTITIES_X = dict(
    quantities=["phi", "RH_phi", "dyphi-T", "dyphi-upar", "dyphi2", "P_RH_tot", "P_RH_NL", "P_RH_even", "P_RH_odd", "P_RH_coll", "upar", "upar", "temperature", "Pi_RH_NL", "Pi_RH_even", "Pi_RH_odd"],
    kx_orders=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    mults=[-1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    mults_zed=[None, None, None, None, None, None, None, None, None, None, None, "cos", None, None, None, None],
    datanames=["vE", "vE_RH", "Q", "Pi_parallel", "vEx2", "P_RH", "P_RH_phi", "P_phi_even", "P_phi_odd", "P_RH_coll", "upar", "upar_cos", "gradTZ", "Pi_RH_NL", "Pi_RH_even", "Pi_RH_odd"],
)

# Same, for movie_quantities_x_zed.py's (single, currently-active) quantities
# block.
FULL_QUANTITIES_X_ZED = dict(
    quantities=["phi", "RH_phi", "dyphi-upar", "dyphi-T", "dyphi2", "P_RH_tot", "P_RH_NL", "P_RH_even", "P_RH_odd", "P_RH_coll", "upar", "temperature"],
    kx_orders=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    mults=[-1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    datanames=["vE", "vE_RH", "Pi_parallel", "Q", "vEx2", "P_RH", "P_RH_phi", "P_phi_even", "P_phi_odd", "P_RH_coll", "upar", "gradTZ"],
)


@cached(version=2)
def get_quantities_x_tavg(run, quantities, kx_orders, mults, mults_zed, datanames, time_min=500, time_max=1e6, time_idx_step=10, time_avg=None) -> dict:
    """dt-weighted time average, over the frame index range
    [time_idx_min, time_idx_max, time_idx_step), of get_quantity_x's 1D
    (x) output for each (quantity, kx_order, mult, mult_zed) tuple.

    Matches movie_quantities_x.py's accumulation exactly: the first frame
    (i_time_idx==0) is evaluated but excluded from the average (it only
    ever set that script's per-frame plot-normalisation, which has no
    effect on the accumulated data and so isn't reproduced here -- see
    quantities/realspace.py's get_quantity_x docstring), and dt is the raw
    time difference between the two most recent *sampled* time_idx values
    (not divided by time_idx_step).

    time_avg (renamed from dt_avg): per-frame smoothing width forwarded to
    get_quantity_x's own time_avg -- a window CENTERED on each frame's own
    time (quantities/realspace.py::get_quantity_zed_x_y), not a trailing
    window. This is a genuinely different convention from the
    trailing-window time_avg used by
    stella_diagnostics.scan.zonal_flow_scan/rh_flux_scan and
    physics.velocity_space.plot_contour_gvmu_vpa -- kept distinct
    (not silently unified) since "smoothed around this exact frame" and
    "trailing up to this frame" are different, both legitimate, choices;
    only the field *name* is now shared for config-vocabulary consistency.
    """
    assert len(set(datanames)) == len(datanames), "datanames must be unique: %r" % (datanames,)

    time_idx_min = run.get_time_idx(time_min)
    time_idx_max = run.get_time_idx(time_max)
    time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)
    time = run.ncdata.variables["t"][:]

    nx = len(run.ncdata.variables["kx"])
    avg_quantities_x = np.zeros((len(quantities), nx))

    dt_sum = 0
    x = None
    for i_time_idx, time_idx_val in enumerate(time_idx_vals):
        for i_quantity, quantity in enumerate(quantities):
            kx_order = kx_orders[i_quantity]
            mult = mults[i_quantity]
            mult_zed = mults_zed[i_quantity]

            _, x, f_Z, _ = get_quantity_x(run, quantity=quantity, time_idx=time_idx_val, kx_order=kx_order, mult=mult, mult_zed=mult_zed, normalise=False, time_avg=time_avg)

            if i_time_idx > 0:
                if i_quantity == 0:
                    dt = time[time_idx_val] - time[time_idx_val - 1]
                    dt_sum += dt
                avg_quantities_x[i_quantity, :] += f_Z * dt

    avg_quantities_x /= dt_sum

    result = {"x": x}
    for i, name in enumerate(datanames):
        result[name] = avg_quantities_x[i]
    return result


@cached(version=1)
def get_quantities_x_zed_tavg(run, quantities, kx_orders, mults, datanames, time_min=500, time_max=1e6, time_idx_step=2) -> dict:
    """Like get_quantities_x_tavg, but for the (zed, x) profiles from
    get_quantity_x_zed (only_zonal=True), matching
    movie_quantities_x_zed.py's accumulation exactly."""
    assert len(set(datanames)) == len(datanames), "datanames must be unique: %r" % (datanames,)

    time_idx_min = run.get_time_idx(time_min)
    time_idx_max = run.get_time_idx(time_max)
    time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_step)
    time = run.ncdata.variables["t"][:]

    nx = len(run.ncdata.variables["kx"])
    nzed = len(run.ncdata.variables["zed"])
    avg_quantities_zed_x = np.zeros((len(quantities), nzed, nx))

    dt_sum = 0
    x = zed = None
    for i_time_idx, time_idx_val in enumerate(time_idx_vals):
        for i_quantity, quantity in enumerate(quantities):
            kx_order = kx_orders[i_quantity]
            mult = mults[i_quantity]

            f_zed_x, x, zed, _ = get_quantity_x_zed(run, quantity=quantity, time_idx=time_idx_val, kx_order=kx_order, mult_fac=mult, only_zonal=True)

            if i_time_idx > 0:
                if i_quantity == 0:
                    dt = time[time_idx_val] - time[time_idx_val - 1]
                    dt_sum += dt
                avg_quantities_zed_x[i_quantity, :, :] += f_zed_x * dt

    avg_quantities_zed_x /= dt_sum

    result = {"x": x, "zed": zed}
    for i, name in enumerate(datanames):
        result[name] = avg_quantities_zed_x[i]
    return result
