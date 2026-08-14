"""Time-averaged kx*rhoi outer-scale comparison across runs, plus the
scaling-vs-tprim figure built from those averages.

Extracted from example_plots/plot_compare_kx_rhoi.py's inline windowed
time-average -- the per-run computation (run.read_avg_kx_rhoi() already
exists on StellaRun; the windowing/averaging around it did not) is now a
small cached function here.
"""

import numpy as np

from stella_diagnostics.io.cache import cached


@cached(version=3)
def get_time_avg_kx_rhoi(run, time_avg=350, take_last=False, time_val_avg=None):
    """(time, kx_rhoi_O, kx_rhoi_O_avg) for one run.

    kx_rhoi_O_avg is either the last value (take_last=True), or a
    time_avg-WIDE window average: trailing (the last time_avg time units
    of the run) when time_val_avg is None, or centered on time_val_avg
    otherwise -- nan if the run doesn't span a full time_avg-wide window.
    Same convention as stella_diagnostics.scan.zonal_flow_scan/rh_flux_scan.

    NOTE: before this fix, time_avg here meant an ABSOLUTE THRESHOLD
    (average everything with t > time_avg), not a window width -- a
    different, inconsistent convention from every other time-averaging
    function in this codebase. This is a deliberate behavior change (not
    preserved as historical numeric output) per an explicit decision to
    unify all of them onto one convention; there is no real, checked-in
    output in this repo to compare against (this function's real configs
    point at directories outside this repo).

    Also fixes a separate, pre-existing bug found while verifying the
    above: run.read_avg_kx_rhoi() returns (kx_rhoi_O, time), but this
    function was unpacking it as `time, kx_rhoi_O = ...` (swapped) --
    confirmed via git history to predate this fix. A plain variable-order
    bug, not a physics/formula choice, so corrected rather than preserved.
    """
    kx_rhoi_O, time = run.read_avg_kx_rhoi()

    if take_last:
        return time, kx_rhoi_O, kx_rhoi_O[-1]

    if time_val_avg is None:
        mask = time > time[-1] - time_avg
    else:
        mask = (time > time_val_avg - time_avg / 2) & (time < time_val_avg + time_avg / 2)

    if time[-1] - time[0] < time_avg or not np.any(mask):
        kx_rhoi_O_avg = np.nan
    else:
        kx_rhoi_O_avg = np.average(kx_rhoi_O[mask])

    return time, kx_rhoi_O, kx_rhoi_O_avg
