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

## Scan comparisons: config-driven, not copy-pasted

If you compare runs across many scan directories, the pattern of
"duplicate the whole plotting script into each directory, hand-edit the
list of directories/labels inside it, and manually re-apply any fix to
every copy whenever the analysis code changes" doesn't scale. The fix:
pull the scan definition (which directories, labels, colors, axis limits)
out as a small, standalone, data-only Python config file, and keep the
actual analysis/plotting code in exactly one place --
`stella_diagnostics` -- called by a single reusable driver script that's
never copied, only pointed at a different config.

`example_plots/plot_flux_time.py` is a worked example of this: it used to
contain the full Qflx(t)/E_phi(t)/E_upar(t) analysis logic *and* several
entire hardcoded scan definitions back to back (only the last one before
the plotting loop was ever active -- the rest were dead code kept around
as copy-paste templates for the next comparison). Now it's a ~20-line
driver:

```
python plot_flux_time.py scan_configs/scan_nu_var.py
```

`example_plots/scan_configs/scan_nu_var.py`, `scan_upwind.py`, and
`scan_nu_var2.py` are the three comparisons that used to be those
hardcoded blocks, each now a small file defining just `dirnames`
(required) and optionally `labels`, `colors`, `filename`, `code`,
`Q_div`, `skip_phi2`, `plot_ratio`, `ylim`, `figname_add`. Adding a new
comparison means writing a new file like these -- not copying
`plot_flux_time.py` -- and any improvement to the underlying analysis in
`stella_diagnostics.scan.flux_energy_scan` (which is `@cached`, so
re-plotting the same comparison after the first run is fast) applies to
every config automatically, current or future.

Two building blocks in `stella_diagnostics.scan.config` support this for
any future driver+config pair, not just this one:

```python
from stella_diagnostics.scan.config import load_scan_config, discover_runs

# Dynamically load a scan config .py file, with clear errors for missing
# required fields:
config = load_scan_config("scan_configs/scan_nu_var.py")

# For *regular* single-parameter scans, skip hand-typing directory names
# entirely: glob a base directory and extract the scan value from each
# subdirectory's name.
runs = discover_runs(
    "2026-06-26_scan_qinp-1.4_.../",
    pattern="run_tprim-*",
    param_regex=r"tprim-([0-9.eE+-]+)",
)
# -> [(dirname, tprim_value), ...], sorted by tprim_value
```

A config file can call `discover_runs` internally to build `dirnames`/
`labels` programmatically for a regular scan, or just hardcode a curated
list for an irregular/curated comparison (like `scan_upwind.py`, which
mixes several different numerical settings rather than one swept
parameter) -- both produce the same kind of config module, so the driver
doesn't need to know which one a given config used.

### Running the converted scripts

19 of the 30 scripts in `example_plots/` have been converted to this
pattern (see "Known issues" below for the 11 not yet converted). Every
converted script is run the same way, from inside `example_plots/`:

```
python <script>.py scan_configs/<config>.py
```

`scan_configs/*.py` are example configs, one (or a few) per driver,
showing real values that used to be hardcoded in the script itself --
copy one and edit it to point at your own runs rather than editing the
driver. Any *.py file with the right fields works; the ones checked in
are just worked examples, not a fixed list.

| Script | Example config(s) | What it plots |
|---|---|---|
| `plot_fluxes.py` | `fluxes_default.py` | Flux(t), one run |
| `plot_correlation_func.py` | `correlation_func_default.py` | Parallel correlation function, one run |
| `plot_quantities_over_zed.py` | `quantities_over_zed_default.py` | Multiple quantities vs zed, one run |
| `plot_geometry_compare_flux_tubes.py` | `geometry_compare_default.py` | Flux-tube geometry comparison |
| `plot_compare_growth_rates.py` | `growth_rates_default.py` | omega(ky) convergence vs akyminmax x nfield_periods |
| `plot_compare_phi_zed.py` | `compare_phi_zed_default.py` | phi(zed) vs akyminmax x nfield_periods |
| `plot_contour_phi_vs_t_zed.py` | `contour_phi_vs_t_zed_default.py` | \|phi\|(zeta, t) grid (**broken**, see "Known issues") |
| `plot_gvmus_all_dirs.py` | `gvmus_grid_{shat,qinp,eps,coll}.py` | g(vpa, mu) contour grid across a 2D run sweep |
| `plot_phi_spectrum_compare.py` | `phi_spectrum_{nu0_only,nu_scan,qinp_scan}.py` | phi(k) spectra (ky, kx nonzonal, kx zonal) across a tprim sweep |
| `plot_contour_quantity_vs_kx_omega.py` | `contour_kx_omega_coll_comparison.py` | quantity(kx, omega) contour comparison |
| `plot_compare_kx_rhoi.py` | `compare_kx_rhoi_default.py` | kx*rhoi(t) + scaling-vs-tprim comparison |
| `plot_phiZ_TS_qkappa2.py` | `phiZ_TS_qkappa2_default.py` | E_zonal vs q*kappa^2 scaling |
| `plot_RH_P_C_kx_from_file.py` | `rh_p_c_kx_default.py` | Collisional P_RH(kx)/(nu*E_RH) vs vnew x tprim |
| `plot_contour_quantity_vs_t_x.py` | `contour_quantity_x_t_P_RH_tot.py` | quantity(x, t) contour grid + Qflx/phi2 overlay |
| `plot_flux_time.py` | `scan_nu_var.py`, `scan_upwind.py`, `scan_nu_var2.py` | Qflx(t)/E_phi(t)/E_upar(t) comparison |
| `movie_gvmus_t.py` | `movie_gvmus_t_default.py` | Movie: g(vpa, mu) vs time, one run |
| `movie_gvmus_Z-NZ.py` | `movie_gvmus_ZNZ_default.py` | Movie: g_NZ/g_Z(vpa, mu) side by side vs time |
| `movie_gzvs_Z-NZ.py` | `movie_gzvs_ZNZ_default.py` | Movie: g_NZ/g_Z(zed, vpa) side by side vs time |
| `movie_gvmus_Z-NZ_kxs.py` | `movie_gvmus_ZNZ_kxs_default.py` | Movie: g_NZ/g_Z(vpa, mu), one row per kx band, vs time |

Each converted script has a module docstring listing its config's exact
fields (required and optional) -- read the top of the script, or an
existing config in `scan_configs/`, before writing a new one.

### Not yet converted

11 scripts remain in their original hardcoded-path, standalone form --
each contains substantial original analysis logic (not just a call into
one existing package function), so converting them means extracting real
physics code into `stella_diagnostics`, not just moving data into a
config file. Left for a later pass rather than done blind, since none of
it could be verified against real multi-run data here:

- `plot_flux_coll.py`, `plot_ERH_Ephi.py` -- E_phi/Gamma0 calculations
  that overlap with `stella_diagnostics.scan.flux_energy_scan` and need
  reconciling, not just moving.
- `plot_RH_phi_E_P_t_all_kx.py` -- per-kx caching/aggregation/summary
  figures; has a real bug where using its `.dat` cache silently skips
  generating a whole summary figure.
- `plot_mean_quantities_x.py` + `movie_quantities_x.py`, and
  `plot_mean_quantities_x_zed.py` + `movie_quantities_x_zed.py` -- two
  more ad hoc `.dat`-cache read/write script pairs (like the
  `flux_energy_scan.py` example was). One has a **divide-by-zero bug**
  (dividing an accumulator by a sum that's 0 when no new frames render).
- `plot_param_scan_Dimits.py` + `get_Dimits.py` -- a `.json`-cache pair;
  `get_Dimits.py` alone has ~500 lines of dense, original growth-rate/RH
  analysis with no package equivalent yet.
- `plot_zonal_distribution.py` -- a fully self-contained analysis
  subsystem (stella input-namelist parsing, stella's internal
  `kxkyz_lo` domain-decomposition layout, restart-file reading) that
  doesn't use `StellaRun` at all.
- `movie_quantity_real_space.py` -- needs a small new package helper for
  its pre-loop zonal-profile normalization before `render_movie` can be
  wired in cleanly.

## Testing

```
pytest tests/
```

109 tests, all data-free (no real STELLA output needed): import checks
for every module, an `inspect`-based diff proving the public API
surface on `StellaRun`/`RunCollection` is unchanged from before the
restructure, an AST-based check that every method call in
`example_plots/*.py` resolves on the real classes (including
instance attributes like `.ncdata` set in `__init__`, not just
methods), caching-layer and movie-rendering tests (`ffmpeg` calls
mocked, no real binary needed), scan-config-loading/directory-discovery
tests, a multi-run flux/energy-comparison test (using a synthetic
3-run scan directory) verifying the underlying computation is
actually cached, and smoke tests against a synthetic
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
- **`plot_parallel_correlation_function`** (`stella_diagnostics/physics/correlations.py`)
  vs. **`example_plots/plot_correlation_func.py`**: the function returns
  5 values (`fig, ax, im, avg_delta_chi, k`), but the script has always
  unpacked only 3 (`fig, ax, im = ...`), so it raises `ValueError` on
  every run. Confirmed present in the very first commit of this repo --
  predates this restructure entirely. Preserved as-is, flagged with a
  `# NOTE`, per this project's flag-not-fix convention; found while
  verifying the config-driven scripts against a real multi-run scan.
