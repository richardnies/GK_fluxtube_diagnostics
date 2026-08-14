"""Diagnostics/plotting package for STELLA (and GX/GS2) gyrokinetic simulation output.

The main entry points are :class:`stella_diagnostics.io.run.StellaRun` (a
single simulation output) and :class:`stella_diagnostics.scan.run_collection.RunCollection`
(a set of runs compared side by side). The legacy top-level modules
``stellaDiagnostics`` and ``loadStellaScan`` re-export these two classes
under their old names for backward compatibility.

Naming inconsistencies inherited from the original codebase (kept as-is
to preserve the existing public API rather than renamed, per the
decision to restructure/dedupe without changing any public method or
keyword-argument name):

- ``timeavg``/``timemax`` (e.g. ``StellaRun.plot_flux_over_time``) is a
  third, still-unreconciled spelling of "time-averaging window",
  distinct from the two below.
- growth-rate/omega convergence checks (``read_data_omega_k``,
  ``read_omega_t``, ``plot_omega_ky``, ``plot_omega_kx``,
  ``plot_contour_gamma_kx_ky``, ``load_omegas``) now use the same
  ``time_avg``/``time_val_avg`` field names as ordinary
  quantity-time-averaging (renamed from ``delta_t_avg``/``t_val``), but
  NOT quite the same math: ``time_val_avg=X`` here means a TRAILING
  window ``(X - time_avg, X]`` (matching the definition of
  ``read_data_omega_k``'s own ``timestep`` argument, which it's meant to
  generalize), not the CENTERED window the rest of the package uses for
  the same field. This is a genuinely different analysis (finding when a
  linear growth rate has converged) from ordinary quantity-time-averaging,
  so the field names are now aligned for consistency but the underlying
  computation was deliberately left as-is. ``spectrum_scan.plot_Q_k_spectrum``
  is unrelated dead code (not called by any driver) that still spells this
  ``delta_t_avg`` -- left alone rather than renamed for a function nothing
  uses.
- ``read_data_omega_k``/``read_omega_t`` correctly handle all three
  ``.omega`` ascii file layouts stella can write (7-column, or either of
  two differently-meaning 5-column variants -- see
  ``spectral.omega._read_omega_ascii_file``'s docstring, confirmed against
  the actual STELLA Fortran source); requesting ``om_avg=True``/``False``
  for data the file doesn't contain raises a clear ``ValueError`` rather
  than silently reading the wrong column or crashing on the reshape.
- "exclude the zonal (ky=0) component" is spelled four different ways:
  ``remove_zonal``, ``only_zonal``, ``keep_only_zonal``, and
  ``zonal``/``nozonal``.
- the radial magnetic/curvature drift is spelled ``gbdrift0``,
  ``cvdrift0``, ``vdriftx``, or ``vMx`` depending on which function/plot
  label you're looking at.

Common abbreviations: RH = Rosenbluth-Hinton (residual zonal-flow
test), FLR = finite-Larmor-radius (gyroaveraging), EZ = zonal energy.

Ordinary quantity-time-averaging (as opposed to the omega-convergence
family above) now uses one shared convention across the whole package:
``time_min``/``time_max`` for a plain range (no averaging), and
``time_avg``/``time_val_avg`` for a windowed average --
``time_val_avg=None`` (default) means a TRAILING window of width
``time_avg`` ending at the run's last sample (or, for per-frame/per-call
functions that take an explicit ``time_idx``/``time_val`` instead of a
``time_val_avg``, ending at that call's own reference time);
``time_val_avg=<X>`` means a window of width ``time_avg`` CENTERED on
``X``. See ``stella_diagnostics.scan.zonal_flow_scan`` for the reference
implementation. ``quantities_x_scan.get_quantities_x_tavg``'s per-frame
``time_avg`` is the one deliberate exception, kept centered (not
trailing) since it smooths a value around one specific already-explicit
frame rather than summarizing a whole run -- see its own docstring.
``kx_max``/``kx_min`` were NOT unified: they mean at least four different
things across the package (a plot-axis limit, a velocity-space kx-band
filter, a per-kx cutoff, and a long-wavelength low-pass filter) that
compute genuinely different quantities, so merging the name wouldn't
have merged the computation.
"""
