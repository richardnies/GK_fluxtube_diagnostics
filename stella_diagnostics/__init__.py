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

- "time-averaging window" is spelled three different ways across
  sibling functions: ``timeavg``/``timemax``, ``time_avg``/``time_max``,
  and ``delta_t_avg``/``t_val``.
- "exclude the zonal (ky=0) component" is spelled four different ways:
  ``remove_zonal``, ``only_zonal``, ``keep_only_zonal``, and
  ``zonal``/``nozonal``.
- the radial magnetic/curvature drift is spelled ``gbdrift0``,
  ``cvdrift0``, ``vdriftx``, or ``vMx`` depending on which function/plot
  label you're looking at.

Common abbreviations: RH = Rosenbluth-Hinton (residual zonal-flow
test), FLR = finite-Larmor-radius (gyroaveraging), EZ = zonal energy.
"""
