"""(1-Gamma0)|phi_Z(kx)|^2 spectrum integration, for the E_zonal-vs-q*kappa^2
scaling comparison.

Extracted from example_plots/plot_phiZ_TS_qkappa2.py -- reads precomputed
"<prefix>_kx_zonal.dat" / "<prefix>_kx_zonal_stddev.dat" files (an external
cache this script only ever consumed, never generated) and integrates them
over a kx window. Not wrapped in stella_diagnostics.io.cache since it
doesn't take a StellaRun (there's no netCDF read here at all, just two flat
files) and the integration itself is cheap.
"""

import numpy as np


def get_Ezonal_from_kx_zonal_file(filename_prefix, kxmin=0.3, kxmax=1e4, fac_rescale=1.0):
    """(Ezonal, Ezonal_stddev): the (1-Gamma0)|phi_Z(kx)|^2 spectrum in
    '<filename_prefix>_kx_zonal.dat' (and its _stddev sibling), integrated
    over kx in [kxmin, kxmax] and scaled by fac_rescale.
    """
    phi2k, kx = np.loadtxt(filename_prefix + "_kx_zonal.dat")
    phi2k_stddev, kx = np.loadtxt(filename_prefix + "_kx_zonal_stddev.dat")

    idx_min = np.argmin(np.abs(kx - kxmin))
    idx_max = np.argmin(np.abs(kx - kxmax))

    Ezonal = np.trapz(y=phi2k[idx_min:idx_max], x=kx[idx_min:idx_max]) * fac_rescale
    Ezonal_stddev = np.trapz(y=phi2k_stddev[idx_min:idx_max], x=kx[idx_min:idx_max]) * fac_rescale

    return Ezonal, Ezonal_stddev
