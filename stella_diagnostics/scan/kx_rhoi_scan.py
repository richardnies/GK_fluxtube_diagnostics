"""Time-averaged kx*rhoi outer-scale comparison across runs, plus the
scaling-vs-tprim figure built from those averages.

Extracted from example_plots/plot_compare_kx_rhoi.py's inline windowed
time-average -- the per-run computation (run.read_avg_kx_rhoi() already
exists on StellaRun; the windowing/averaging around it did not) is now a
small cached function here.
"""

import numpy as np

from stella_diagnostics.io.cache import cached


@cached(version=1)
def get_time_avg_kx_rhoi(run, time_avg=350, take_last=False):
    """(time, kx_rhoi_O, kx_rhoi_O_avg) for one run.

    kx_rhoi_O_avg is either the last value (take_last=True) or the average
    over t > time_avg (nan if the run hasn't reached time_avg).
    """
    time, kx_rhoi_O = run.read_avg_kx_rhoi()

    if take_last:
        kx_rhoi_O_avg = kx_rhoi_O[-1]
    elif time[-1] > time_avg:
        kx_rhoi_O_avg = np.average(kx_rhoi_O[time > time_avg])
    else:
        kx_rhoi_O_avg = np.nan

    return time, kx_rhoi_O, kx_rhoi_O_avg
