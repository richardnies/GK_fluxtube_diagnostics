"""Backward-compatible shim: re-exports stella_diagnostics under the
legacy top-level names so existing scripts that do
``import stellaDiagnostics as sD; sD.stellaDiagnostics(filename)``
keep working unmodified. New code should import from stella_diagnostics
directly (see stella_diagnostics.io.run.StellaRun).

NOTE: stellaDiagnostics is now an alias for StellaRun, so
``type(obj).__name__`` reports "StellaRun", not "stellaDiagnostics".
Flag this if any downstream code introspects the class name instead of
using isinstance()/duck typing.
"""
from stella_diagnostics.io.run import StellaRun as stellaDiagnostics

# Module-level helper functions that historically lived in this file,
# for any scripts that imported them directly (e.g. `sD.get_fft_k(...)`).
from stella_diagnostics.spectral.fft import (
    get_fft_real_space,
    get_fft_k,
    plot_Wigner_t_omega,
    get_Wigner_x_kx,
)
from stella_diagnostics.spectral.omega import (
    extract_growth_rate,
    get_avg_stddev_timetrace,
    get_convergence_quantity,
    plot_convergence_quantity,
    Laplace_transform,
    estimate_omega_gamma_signal,
)
from stella_diagnostics.physics.fluxes import get_true_flux_norm
from stella_diagnostics.physics.correlations import (
    get_correlation_func_1D,
    get_correlation_func_2D,
)
from stella_diagnostics.spectral.stats import get_statistics
from stella_diagnostics.plotting.zed_plots import plot_y_over_zed
from stella_diagnostics.scan.spectrum_scan import plot_qflx_tprim_qinp_dir

__all__ = [
    "stellaDiagnostics",
    "get_fft_real_space",
    "get_fft_k",
    "plot_Wigner_t_omega",
    "get_Wigner_x_kx",
    "extract_growth_rate",
    "get_avg_stddev_timetrace",
    "get_convergence_quantity",
    "plot_convergence_quantity",
    "Laplace_transform",
    "estimate_omega_gamma_signal",
    "get_true_flux_norm",
    "get_correlation_func_1D",
    "get_correlation_func_2D",
    "get_statistics",
    "plot_y_over_zed",
    "plot_qflx_tprim_qinp_dir",
]
