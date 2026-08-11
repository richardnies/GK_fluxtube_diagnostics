# stella_diagnostics

Post-processing and plotting diagnostics for [STELLA](https://github.com/stellaGK/stella)
(and GX/GS2) gyrokinetic simulation output.

## Install

```
pip install -e .
```

This makes both the new package (`import stella_diagnostics`) and the
legacy top-level modules (`import stellaDiagnostics`, `import
loadStellaScan`, `import setupStellaScan`) importable from any working
directory — no `sys.path.append` needed.

Dependencies: `numpy`, `netCDF4`, `matplotlib`, `scipy`, `seaborn`,
`tftb`. `tftb` (the Time-Frequency Toolbox) is only used by the two
Wigner-Ville functions in `stella_diagnostics.spectral.fft`, but is
imported unconditionally at that module's top, so it's a hard
dependency for now even if you never call those two functions.

## Package layout

```
stella_diagnostics/
    io/          StellaRun (single-run handle) + per-code path/variable resolution
    grid.py      shared kx/ky/zed/time readers, nearest_index, dl_over_B_avg, FLR
    quantities/  quantity-name -> data dispatch (k-space and real-space) + LaTeX labels
    physics/     fluxes, Rosenbluth-Hinton (RH), zonal energy, correlations, velocity space
    spectral/    FFTs, omega/growth-rate extraction, time-trace statistics
    plotting/    plot_* functions, grouped by independent variable (zed / k-space / real-space / flux-time)
    scan/        RunCollection (cross-run comparisons) + omega/spectrum scan plots
```

`stellaDiagnostics.py` and `loadStellaScan.py` at the repo root are
backward-compatibility shims: they just re-export
`stella_diagnostics.io.run.StellaRun` and
`stella_diagnostics.scan.run_collection.RunCollection` under their old
names, so any existing script doing
`import stellaDiagnostics as sD; sD.stellaDiagnostics(filename)` keeps
working unmodified. New code should import from `stella_diagnostics`
directly.

`setupStellaScan.py` is a separate, unrelated tool (VMEC field-line
geometry setup for generating stella input decks) — it shares no code
with the diagnostics package above.

## Usage

A `StellaRun` (aka `stellaDiagnostics`) wraps a single simulation
output. `filename_base` is a path *without* extension — the class
appends `.out.nc`/`.nc`, `.in`, `.omega`, `.vmec.geo`/`.geometry`,
`.fluxes` etc. itself depending on `code`:

```python
from stella_diagnostics.io.run import StellaRun

run = StellaRun("run_akyminmax-1.0000_nfield_periods-30.0000/precise_QA")
axs = run.plot_flux_over_time()
```

A `RunCollection` (aka `loadStellaScan`) wraps a list of runs for
side-by-side comparison:

```python
from stella_diagnostics.scan.run_collection import RunCollection

scan = RunCollection(
    ["run_akyminmax-0.1000_nfield_periods-100.0000/precise_QA",
     "run_akyminmax-0.1000_nfield_periods-200.0000/precise_QA"],
    labels=["Nfp=100", "Nfp=200"],
)
scan.plot_phi_vs_zed(zed_times_nfield_periods=True)
```

`code="stella"` (default) / `"GX"` / `"GS2"` selects which simulation
code produced the output, since file naming and netCDF variable
layout differ between them.

See `example_plots/` for further worked examples (growth-rate
convergence scans, flux-tube geometry comparisons, correlation
functions, k-spectra, ...).

### Run-directory naming convention

Nothing in the code enforces a specific directory-naming scheme, but
the run directories the example scripts expect follow patterns like
these (inherited from how the runs were originally generated):

```
run_akyminmax-<aky>_nfield_periods-<nfp>/precise_QA[_NL]
run_tprim-<tprim>_zeta_center-<zctr>/precise_QA[_NL]
<base_dir>/run_akyminmax-<aky>_tprim-<tprim>/precise_QA
```

where `<base_dir>` is often itself parameterized, e.g.
`fprim-1_adb-el_zetactr-<zctr>_theta0-0`. `precise_QA`/`precise_QA_NL`
is the `filename_base` passed to `StellaRun`.

## Testing

```
pytest tests/
```

43 tests, all data-free (no real STELLA output needed): import checks
for every module, an `inspect`-based diff proving the public API
surface on `StellaRun`/`RunCollection` is unchanged from before the
restructure, an AST-based check that every method call in
`example_plots/*.py` resolves on the real classes, and smoke tests
against a synthetic in-memory netCDF dataset (construction, the core
grid readers, a couple of real analysis code paths end-to-end).

### Manual verification required

The test suite above does **not** validate physics/numbers, and covers
only a small slice of the ~40-branch quantity dispatch (fabricating a
netCDF dataset large enough to exercise all of it wasn't worth doing
blind). Before relying on this for real work, run all 8
`example_plots/*.py` scripts against real STELLA run directories and
confirm the 7 currently-working ones (see "Known issues" below)
produce figures matching what the pre-restructure code produced.

## Known issues

These predate the restructure and were deliberately left as-is (flagged
with `# NOTE`/`# TODO` comments in the code) rather than silently fixed,
since fixing them would be a behavior change:

- **`example_plots/plot_contour_phi_vs_t_zed.py`** calls
  `StellaObj.plot_contour_phi_zed_t(...)`, a method that has never
  existed on `stellaDiagnostics`/`StellaRun`. Closest current
  equivalents: `plot_quantity_zed_t` or
  `RunCollection.plot_contour_phi_vs_zed_theta0`.
- **`stella_diagnostics/plotting/zed_plots.py`**, inside
  `plot_phi_vs_zed`: a malformed comment swallowed a
  `def plot_phi2_vs_t_zed(...):` line, so what would have been a
  separate method is unreachable dead code appended after a `return`.
  See the `# NOTE` at that spot.
- **`stella_diagnostics/quantities/realspace.py`**,
  `get_quantity_zed_x_y`: independently re-implements the same
  ~40-branch quantity dispatch as
  `quantities/registry.get_quantity_zed_kx_ky` instead of calling it
  and FFT-transforming the result (an FFT helper for exactly this
  purpose already exists in `spectral/fft.py` but isn't used here).
  Not merged, since doing so would change a live numerical code path.
  See the `# TODO` at the top of that file.
