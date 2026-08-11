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

## Caching and movies

`stella_diagnostics.io.cache` and `stella_diagnostics.plotting.movies`
are standalone package features for the common "post-processing is
slow, and I end up hand-caching intermediate data to a `.dat` file in
one script and re-loading it in another" situation. No script is
specially "the generator" or "the reader" of cached data -- any call
to a cached function is a cache hit if the parameters (and the run's
source files) haven't changed since it last ran, and a transparent
recompute (which also refreshes the cache) if they have.

```python
from stella_diagnostics.io.cache import cached
from stella_diagnostics.io.run import StellaRun

@cached(version=1)
def get_time_avg_quantity(run, quantity, time_avg, species_idx=0):
    ...  # however expensive; return an ndarray, or a tuple of
         # ndarrays/scalars

run = StellaRun("run_tprim-4.2000/precise_QA")
data = get_time_avg_quantity(run, "phi", time_avg=50)      # computes, caches
data = get_time_avg_quantity(run, "phi", time_avg=50)      # instant, from cache
data = get_time_avg_quantity(run, "phi", time_avg=100)     # different params -> recomputes
```

The cache key is derived from the function's bound arguments (minus
`run`), so widening a time-averaging window or changing which quantity
you're asking for automatically triggers a recompute -- there's no
manual cache file to delete. The cache is also invalidated
automatically if the run's underlying `.out.nc`/`.fluxes`/`.omega`
files get a newer modification time than the cache (e.g. the
simulation was restarted or extended). Cache files are written as
sibling files next to `filename_base`
(`<filename_base>__cache_<name>_<hash>.npz`), matching the existing
`.out.nc`/`.omega`/`.fluxes` convention.

`get_cached(run, name, compute_fn, params=..., version=..., force=...)`
is the lower-level function `@cached` wraps, for cases where you don't
want to decorate a whole free function. `force=True` bypasses the
cache unconditionally; `clear_cache(run, name=None)` deletes one cache
entry or all of a run's cache entries; the environment variable
`STELLA_DIAGNOSTICS_NO_CACHE=1` disables caching globally (useful when
debugging).

`stella_diagnostics.plotting.movies.render_movie(img_dir,
frame_indices, frame_fn, ...)` replaces the mkdir/skip-if-exists-frame/
ffmpeg-subprocess boilerplate duplicated across the `movie_*.py`
example scripts: pass it an output directory, a sequence of frame
indices, and a callback `frame_fn(i, idx) -> Figure`, and it renders
each frame (skipping ones that already exist unless `rerun_all=True`)
and encodes the result with `ffmpeg`. Raises a clear
`FFmpegNotFoundError` (frames are still written) if `ffmpeg` isn't
installed.

`stella_diagnostics.plotting.mpl_helpers.set_default_style()` replaces
the `plt.rcParams.update({...})` block duplicated at the top of most
`example_plots/*.py` scripts.

None of `example_plots/*.py` has been migrated to use these yet -- they're
available for new scripts, and for migrating existing ones incrementally
whenever convenient.

## Testing

```
pytest tests/
```

89 tests, all data-free (no real STELLA output needed): import checks
for every module, an `inspect`-based diff proving the public API
surface on `StellaRun`/`RunCollection` is unchanged from before the
restructure, an AST-based check that every method call in
`example_plots/*.py` resolves on the real classes (including
instance attributes like `.ncdata` set in `__init__`, not just
methods), caching-layer and movie-rendering tests (`ffmpeg` calls
mocked, no real binary needed), and smoke tests against a synthetic
in-memory netCDF dataset (construction, the core grid readers, a
couple of real
analysis code paths end-to-end).

### Verified against real run data

Beyond the data-free suite, ~90 `StellaRun` methods were exercised
directly against two real stella outputs (a linear-scan run and a
smaller multi-species run) to catch anything the synthetic fixture
couldn't, and 15 representative plots were actually rendered and
visually inspected (not just checked for exceptions) -- flux/time
traces, flux-tube geometry, k-space/real-space/velocity-space
contours, RH diagnostics, poloidal-ring plots. Everything that worked
before the restructure still works; every failure (including one
plot that ran without error but rendered blank) traced back to a
pre-existing issue in the original code, unrelated to the restructure
(confirmed by re-reading the identical logic in the pre-restructure
baseline commit). See "Known issues" below for the newly-confirmed
ones.

This was still only two runs (one linear-scan, one small multi-mode/
multi-species), not the full ~40-branch quantity dispatch or every
plot function, and does not check that plotted *numbers* are
physically correct, only that the code paths run. Before relying on
this for real work, run the `example_plots/*.py` scripts you actually
use against your own run directories and sanity-check the output.

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

The following were newly found while verifying against real run data
(see "Verified against real run data" above); all confirmed present,
unchanged, in the pre-restructure baseline, so they predate the
restructure too:

- **`get_energies_over_time`** and **`get_moments2_over_time`**
  (`stella_diagnostics/physics/fluxes.py`): for
  `code="stella"` these just `print("To be implemented.")` and fall
  through to `return delfs2, hs2, phis2, time` / `return phi2, ...`
  without ever assigning those names, so both raise
  `UnboundLocalError` on any stella run. Only the `code="GS2"` branch
  is implemented.
- **`plot_RH_phi_I`** (`stella_diagnostics/physics/rosenbluth_hinton.py`):
  reads `len(idxs_kx)` before the `if idxs_kx is None: idxs_kx =
  np.arange(...)` line that would give it a value, so calling it with
  the documented default (`idxs_kx=None`) always raises `TypeError`.
- **`plot_quantity_zonal`** (`stella_diagnostics/plotting/kspace_plots.py`):
  builds axis labels as `r"$\partial_x $" + label`, which raises
  `TypeError` whenever the default `label=None` is used (i.e. every
  call that doesn't explicitly pass a `label`).
- **`plot_net_radial_drift`** (`stella_diagnostics/plotting/flux_plots.py`):
  calls `self.evaluate_net_radial_drift(zed_b=zed_b)`, but
  `evaluate_net_radial_drift`'s only parameter is `B_bounce` — it has
  never accepted `zed_b`. This makes `plot_net_radial_drift`
  unconditionally broken; `evaluate_net_radial_drift` itself works
  fine when called directly.
- **`read_omega_t`/`read_data_omega_k`** (`stella_diagnostics/spectral/omega.py`):
  assumes the `.omega` file has exactly 7 columns per row (`[time ky
  kx Re(om) Im(om) Re(om_avg) Im(om_avg)]`); some stella versions
  write a 5-column `.omega` file instead, which makes the hardcoded
  `.reshape(-1, dim_ky, dim_kx, 7)` fail outright. Separately, even
  with a matching 7-column file, this pair of functions only works
  for single-`(kx,ky)`-point linear runs — `read_omega_t` assigns the
  per-timestep result into scalar array slots
  (`omega_r[i] = self.read_data_omega_k(...)`), which raises
  `ValueError` as soon as a run has more than one `(kx,ky)` mode.
- **`plot_phi_vs_zed`** (`stella_diagnostics/plotting/zed_plots.py`):
  doesn't expose `kx_idx`/`ky_idx`, so it always plots the `(kx=0,
  ky=0)` mode -- which stella always sets identically to zero -- and
  with the also-default `normalise_phi=True`, `read_phi_vs_zed`
  divides by `max(phi)=0`, giving an all-NaN/masked array. Net effect:
  `run.plot_phi_vs_zed()` called with no arguments silently produces a
  blank plot on every stella run. Confirmed by actually rendering the
  output, not just checking for exceptions. Use
  `run.read_phi_vs_zed(kx_idx=..., ky_idx=...)` directly, or
  `plot_quantities_over_zed(plot_phi=True, kx_idx_phi=...,
  ky_idx_phi=...)`, with a non-trivial `(kx, ky)`.
- **Stella-version variable-name drift**: several read paths look up
  netCDF variable names that some stella versions no longer write
  under those names, e.g. `qflx_kxky` (now `qflux_vs_kxkys` in newer
  output), `gvmus`/`gzvs` (now `g2_vs_vpamus`/`g2_vs_zvpas`), and
  `Wenergy_g`. Affected: `plot_flux_spectra`,
  `plot_flux_spectra_kx_ky`, `read_flux_spectra`, `get_n_T_vpa_mu`,
  `get_gvpa_gmu`, `get_Evpa_Emu`, `read_g_vs_zed`,
  `get_Wenergy_t_zed_kx_ky`. If your stella build is recent, expect
  `KeyError`/`IndexError` from these specific functions even though
  the rest of the package works fine against the same run.
