"""Collisional Rosenbluth-Hinton P_RH(kx), normalized by nu_ii*E_RH.

Extracted from example_plots/plot_RH_P_C_kx_from_file.py -- originally read
precomputed "data_ERH_mean_kx.dat"/"data_P_RH_coll_even_mean_kx.dat"/
"data_P_RH_coll_odd_mean_kx.dat" files that were never generated anywhere
in this codebase (an external, undocumented input). Those quantities are
already computed (and `@cached`) by
stella_diagnostics.scan.rh_per_kx_scan.get_RH_per_kx_means, which this now
calls directly instead -- no more standalone .dat files to keep in sync.
"""

import numpy as np

from stella_diagnostics.scan.rh_per_kx_scan import get_RH_per_kx_means


def get_P_RH_coll_normalized_vs_kx(run, vnew, eps=0.18, zero_threshold=1e-14, **kwargs):
    """(kx, P_RH_coll_mean_kx_norm) for one run, or (None, None) if the
    collisional P_RH is negligible everywhere (matches the original
    script's skip-if-below-threshold behavior). **kwargs forwarded to
    get_RH_per_kx_means (time_min/time_max/kx_max/passing_trapped/
    fphi/fapar/fbpar/fcoll).

    eps (inverse aspect ratio r/R) defaults to the Cyclone Base Case's own
    value (0.18 -- see scan.flux_energy_scan._get_flux_energy_vs_time's
    matching epsilon default), not this particular run's actual geometry
    (available as run.aspect_ratio if a run-specific ratio is needed).

    zero_threshold (absolute, not relative to this run's own P_RH scale):
    a floating-point-zero guard, not a physically-motivated cutoff -- it
    exists to catch fcoll=0 runs, where P_RH_coll is identically zero up
    to roundoff, not to judge whether a nonzero collisional signal is
    "physically negligible" (which would need a threshold relative to
    e.g. E_RH_mean_kx, since gyroBohm-normalized P_RH's absolute scale
    varies a lot run to run). Override it if a run's genuine collisional
    signal is ever small enough to fall below the default.
    """
    means = get_RH_per_kx_means(run, **kwargs)
    P_RH_coll_mean_kx = means["P_RH_coll_even_mean_kx"] + means["P_RH_coll_odd_mean_kx"]
    if np.sum(np.abs(P_RH_coll_mean_kx)) < zero_threshold:
        return None, None

    P_RH_coll_mean_kx_norm = P_RH_coll_mean_kx / (vnew * means["E_RH_mean_kx"]) * eps**2
    return means["kx_all"], P_RH_coll_mean_kx_norm
