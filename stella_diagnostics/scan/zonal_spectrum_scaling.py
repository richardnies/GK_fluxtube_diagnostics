"""(1-Gamma0)|phi_Z(kx)|^2 spectrum integration, for the E_zonal-vs-q*kappa^2
scaling comparison.

Used by example_plots/plot_phiZ_TS_qkappa2.py. Originally read precomputed
"<prefix>_kx_zonal.dat"/"_kx_zonal_stddev.dat" files -- these are the same
zonal (kx>0, ky=0) phi^2(kx) spectrum that
stella_diagnostics.scan.spectrum_scan.get_phi_k_spectrum(plot_kx=True,
only_zonal=True) computes, so this now calls that directly instead of
reading hand-rolled files (which nothing in this codebase ever wrote under
that exact name -- a pre-existing naming mismatch with
plot_phi_k_spectrum's own "_Ephi_kx_zonal.dat" convention, not a real
separate external dependency).
"""

import numpy as np

from stella_diagnostics.scan.spectrum_scan import get_phi_k_spectrum


def get_Ezonal(run, kxmin=0.3, kxmax=1e4, fac_rescale=1.0, time_idx=-1, time_avg=None):
    """(Ezonal, Ezonal_stddev): the (1-Gamma0)|phi_Z(kx)|^2 spectrum for
    one run, integrated over kx in [kxmin, kxmax] and scaled by
    fac_rescale.
    """
    kx, phi2_k, phi2_k_stddev = get_phi_k_spectrum(run, plot_kx=True, only_zonal=True, time_idx=time_idx, time_avg=time_avg)

    idx_min = np.argmin(np.abs(kx - kxmin))
    idx_max = np.argmin(np.abs(kx - kxmax))

    Ezonal = np.trapz(y=phi2_k[idx_min:idx_max], x=kx[idx_min:idx_max]) * fac_rescale
    Ezonal_stddev = np.trapz(y=phi2_k_stddev[idx_min:idx_max], x=kx[idx_min:idx_max]) * fac_rescale

    return Ezonal, Ezonal_stddev
