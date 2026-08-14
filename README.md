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
    io/          StellaRun (single-run handle) + per-code path/variable resolution;
                 cache.py (transparent disk cache, see "Caching and movies");
                 restart.py (stella restart-file/namelist I/O, no StellaRun involved)
    grid.py      shared kx/ky/zed/time readers, nearest_index, dl_over_B_avg, FLR
    quantities/  quantity-name -> data dispatch (k-space and real-space) + LaTeX labels
    physics/     fluxes, Rosenbluth-Hinton (RH), zonal energy, correlations, velocity space,
                 gradients (shared analytic critical-gradient formula),
                 zonal_distribution (restart-file velocity-space moments/free energy)
    spectral/    FFTs, omega/growth-rate extraction, time-trace statistics
    plotting/    plot_* functions, grouped by independent variable (zed / k-space / real-space / flux-time);
                 movies.py (render_movie frame-loop + ffmpeg encoding helper);
                 zonal_distribution_plots.py (restart-file-based plotting/orchestration)
    scan/        RunCollection (cross-run comparisons) + omega/spectrum scan plots;
                 config.py (scan-config loading + directory discovery);
                 flux_energy_scan.py, rh_flux_scan.py, rh_per_kx_scan.py,
                 quantities_x_scan.py, zonal_flow_scan.py (multi-run comparison
                 plots + their underlying @cached per-run computations)
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
simulation was restarted or extended). Cache files live in a hidden
`.stella_diagnostics_cache/` directory next to `filename_base`, one
per run directory (shared by every `filename_base` in it) -- keeps a
normal `ls` of a run directory free of cache clutter, the same way
`.git`/`__pycache__` stay out of the way elsewhere.

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

### One config per run/scan directory, not per script

`example_plots/scan_configs/*.py` above is one config file per *script*
(several scripts, each with its own config, all pointed at the same
runs) -- convenient when each config represents an independent,
one-off comparison you own. But if you regularly run *several different
diagnostics* against the *same* run directory or the *same* multi-run
scan, prefer one config file per **directory** instead: a `run_config.py`
sitting in a single run's own directory (for every single-run driver:
`plot_fluxes.py`, `movie_gvmus_t.py`, `plot_zonal_shear_diagnostic.py`,
etc.), or a `scan_config.py` sitting in a scan's base directory (for
multi-run drivers like `plot_flux_coll.py`, `plot_ERH_Ephi.py`,
`plot_param_scan_Dimits.py`). `load_scan_config` tolerates fields a given
script doesn't read, so one file can serve every driver pointed at that
directory. Run every driver from the **base directory** (the scan's
top-level directory, one level up from any individual run), not from
inside the run directory itself -- every driver saves its output
(figures, movie frames/videos) relative to the current working
directory, so a figure never ends up mixed in with a run's own
`.out.nc`/`.fluxes`/`.omega` files:

```
python ../example_plots/plot_zonal_shear_diagnostic.py run_tprim-4.2000/run_config.py
python ../example_plots/movie_quantities_x.py run_tprim-4.2000/run_config.py
```

The payoff: every diagnostic pointed at the same directory automatically
agrees on `time_min`/`time_max`/`kx_max` (and, where safe, `time_avg` --
see the vocabulary below) instead of each having its own independently-
guessed value in its own file, and you're not maintaining N nearly-
identical config files that drift out of sync. `stella_minimal_scan/` is
a full worked example: `run_tprim-4.2000/run_config.py` and
`run_tprim-6.7000/run_config.py` (one per run directory, each pointed at
by `dirname` relative to `stella_minimal_scan/` -- not `"."`, since the
driver is invoked from there, not from inside the run directory), plus
`scan_config.py` and `scan_config_grid.py` at the top level (two, not
one, because `plot_gvmus_all_dirs.py`/`plot_mean_quantities_x.py`/
`plot_mean_quantities_x_zed.py` expect `dirnames` as a nested
`dirnames[row][col]` grid while the other multi-run drivers expect a flat
list under the same field name -- a real, pre-existing structural
incompatibility between those scripts' config contracts, not something
one shared file can paper over; see `scan_config_grid.py`'s docstring).
Read either `run_config.py`'s docstring for the full field rationale,
including which fields are deliberately left out.

#### Canonical time-averaging vocabulary

Every quantity-time-averaging field across the package (previously
`dt_avg`, `delta_t_avg`, and `time_avg` inconsistently meaning different
things in different scripts -- some of it genuine sloppiness, since fixed)
now means the same thing everywhere:

- `time_min`/`time_max` -- a plain time range, no averaging.
- `time_avg` (a window *width*) + `time_val_avg` (a window *center*,
  default `None`) -- a windowed average. `time_val_avg=None` means a
  **trailing** window of width `time_avg` ending at the run's last time
  sample (or, for per-frame functions that already take an explicit
  `time_idx`/`time_val`, ending at that call's own reference time
  instead); `time_val_avg=<X>` means a window of width `time_avg`
  **centered** on `X`. See `stella_diagnostics.scan.zonal_flow_scan` for
  the reference implementation, and the module docstring of
  `stella_diagnostics/__init__.py` for the couple of deliberate
  exceptions (`quantities_x_scan.get_quantities_x_tavg`'s per-frame
  `time_avg`, kept centered on purpose, and the growth-rate/omega
  convergence family below).

  The omega/growth-rate-convergence family (`read_data_omega_k`,
  `read_omega_t`, `plot_omega_ky`, `plot_omega_kx`,
  `plot_contour_gamma_kx_ky`, `load_omegas`) was renamed from
  `delta_t_avg`/`t_val` to the same `time_avg`/`time_val_avg` field names
  for consistency, but the math underneath is genuinely different, not
  just renamed: `time_val_avg=<X>` here means a **trailing** window
  `(X - time_avg, X]`, not centered on `X`. This is a deliberate,
  documented exception (see `stella_diagnostics/__init__.py`), left
  as-is since finding when a linear growth rate has converged is a
  different analysis from ordinary quantity-time-averaging.
  `spectrum_scan.plot_Q_k_spectrum` still spells this `delta_t_avg` -- it
  is unrelated dead code (not called by any driver) and was left alone.

`kx_max`/`kx_min` were deliberately **not** unified the same way: they
mean at least four different things across the package (a plot-axis
limit, a velocity-space kx-band filter, a per-kx cutoff, and a
long-wavelength low-pass filter) that compute genuinely different
quantities, so merging the name wouldn't have merged the computation --
see the same `__init__.py` docstring.

### Running the converted scripts

All 35 scripts in `example_plots/` have been converted to this pattern.
Every converted script is run the same way, from inside `example_plots/`:

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
| `plot_flux_coll.py` | `flux_coll_nu_scan.py` | Qflx/gammaE/vE_RH/upar vs collisionality x tprim |
| `plot_ERH_Ephi.py` | `erh_ephi_nu_scan_red.py` | E_RH/E_phi/chihat/gammaE vs tprim across base dirs |
| `plot_RH_phi_E_P_t_all_kx.py` | `rh_phi_e_p_all_kx_default.py` | Per-kx RH phi_I/E_RH/P_RH figures + "summed over kx"/"vs kx" summaries |
| `movie_quantities_x.py` | `run_config.py`* | Movie: several real-space (x) quantities overlaid, + time-averaged summary |
| `plot_mean_quantities_x.py` | `scan_config_grid.py`* | Grid of time-averaged (x) quantity comparisons (5 display modes) |
| `movie_quantities_x_zed.py` | `run_config.py`* | Movie: several (zed, x) quantity contours, + time-averaged summary |
| `plot_mean_quantities_x_zed.py` | `scan_config_grid.py`* | Grid of time-averaged (zed, x) quantity contours, one figure per quantity |
| `movie_quantity_real_space.py` | `run_config.py`* | Movie: one quantity in (x, y) at several zed slices, with zonal overlay |
| `plot_param_scan_Dimits.py` | `scan_config.py`* | 15-panel R/L_T scan: heat flux, ExB shear, RH power transfer vs tprim |
| `plot_zonal_shear_diagnostic.py` | `run_config.py`* | Rich single-run diagnostic page (heat flux/growth rate/shear/RH power) |
| `plot_zonal_distribution.py` | *(no config file -- edit the CONFIG block at the bottom, see below)* | g(vpa, mu) zonal-mode heatmaps read directly from restart files |

\* Configs marked with `*` live in `stella_minimal_scan/` itself (one
`run_config.py` per run directory, `scan_config.py`/`scan_config_grid.py`
at the top level) rather than in `example_plots/scan_configs/`, per the
"one config per run/scan directory" convention above -- every other
single-run/multi-run driver in this table (`plot_fluxes.py`,
`movie_gvmus_t.py`, `plot_flux_coll.py`, `plot_ERH_Ephi.py`,
`plot_geometry_compare_flux_tubes.py`, `plot_contour_quantity_vs_kx_omega.py`,
`plot_contour_quantity_vs_t_x.py`, `plot_RH_phi_E_P_t_all_kx.py`, etc.) is
also verified working against these same `stella_minimal_scan/` configs,
not just the ones asterisked here -- see that directory for the full
worked example.

Each converted script has a module docstring listing its config's exact
fields (required and optional) -- read the top of the script, or an
existing config in `scan_configs/`, before writing a new one.

`plot_zonal_distribution.py` is the one exception to the config-file
pattern above: it reads stella *restart* files directly (no `StellaRun`
involved at all), so it keeps its original CONFIG-block-at-the-bottom
style -- edit `BASEDIR`/`THETA`/`KX`/etc. in the script and run it with no
arguments. The underlying namelist parsing, `kxkyz_lo` domain-decomposition
layout, and restart-file reading live in `stella_diagnostics.io.restart`;
the velocity-space moment/free-energy physics in
`stella_diagnostics.physics.zonal_distribution`; the plotting/orchestration
in `stella_diagnostics.plotting.zonal_distribution_plots`.

The heavy computation behind the other ten scripts above lives in
`stella_diagnostics.scan.rh_flux_scan`, `rh_per_kx_scan`,
`quantities_x_scan`, and `zonal_flow_scan` -- each is `@cached` (see
"Caching and movies" above), decomposed into several independently-callable
functions rather than one big one, so a run missing some of the needed
data still contributes whatever panels it can rather than failing the
whole comparison. This also structurally fixed three bugs that existed in
the original standalone scripts: `plot_RH_phi_E_P_t_all_kx.py`'s old
`.dat`-cache-hit path silently skipped a whole summary figure;
`movie_quantities_x.py`'s active `datanames` list had two entries both
named `"tmp"`, so the second quantity's time average silently clobbered
the first's; and `movie_quantities_x_zed.py`'s tavg loop divided by a
`dt_sum` that stayed 0 whenever every frame PNG already existed on disk.
All three are now impossible by construction rather than patched, since
the cached computation no longer depends on which frame images happen to
be on disk.

## Testing

```
pytest tests/
```

117 tests, all data-free (no real STELLA output needed): import checks
for every module, an `inspect`-based diff proving the public API
surface on `StellaRun`/`RunCollection` is unchanged from before the
restructure, an AST-based check that every method call in
`example_plots/*.py` resolves on the real classes (including
instance attributes like `.ncdata` set in `__init__`, not just
methods), caching-layer and movie-rendering tests (`ffmpeg` calls
mocked, no real binary needed), scan-config-loading/directory-discovery
tests, a multi-run flux/energy-comparison test (using a synthetic
3-run scan directory) verifying the underlying computation is
actually cached, an `rh_per_kx_scan` test against a synthetic run with
fabricated `RH_inertia`/`RH_phi_I`/`RH_fluxes_phi_*` variables, and smoke
tests against a synthetic
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
- **`read_data_omega_k`/`read_omega_t`** (`stella_diagnostics/spectral/omega.py`,
  **fixed**): used to assume the `.omega` file always has exactly 7
  columns per row (`[time ky kx Re(om) Im(om) Re(omavg) Im(omavg)]`),
  crashing outright on the hardcoded `.reshape(-1, dim_ky, dim_kx, 7)`
  against stella versions that write a 5-column file instead. Checked
  directly against the real STELLA Fortran source
  (`diagnostics_omega.f90`'s `open_omega_ascii_file`/
  `write_omega_to_ascii_file`, controlled by the independent
  `write_omega_vs_kxky`/`write_omega_avg_vs_kxky` namelist flags) and a
  real 5-column example file: there are actually **two** different
  5-column layouts, not one -- `[time ky kx Re(omavg) Im(omavg)]` (only
  the time-averaged frequency) or `[time ky kx Re(om) Im(om)]` (only the
  instantaneous frequency), depending on which flag was set. Both have
  identical shape but different content, distinguishable only by header
  text (`"omavg"` vs. `"frequency"`/`"growth rate"`).
  `spectral.omega._read_omega_ascii_file` now detects all three layouts
  at load time; `read_data_omega_k` raises a clear `ValueError` (instead
  of silently reading the wrong column) if `om_avg=True`/`False` is
  requested for data the file doesn't actually contain, and
  `read_omega_t` picks whichever of raw/averaged the file has. Verified
  against the real 7-column `stella_minimal_scan` data, the real 5-column
  example file (values match exactly), and synthetic multi-mode fixtures
  for all three layouts (`tests/test_omega_format.py`).

  (`read_omega_t`'s separate multi-`(kx,ky)`-mode bug -- it used to
  raise `ValueError` for any run with more than one `(kx,ky)` mode,
  both from miscounting the number of distinct timesteps and from
  assigning a whole `(dim_ky, dim_kx)`-shaped array into a scalar slot
  -- is also fixed; `read_omega_t` now returns `(Nr_timesteps,)`-shaped
  `omega_r`/`omega_i` for a single-mode run same as before, or
  `(Nr_timesteps, dim_ky, dim_kx)`-shaped for a multi-mode run, verified
  against a real multi-mode run. `read_data_omega_k` itself,
  `load_omegas`, `plot_omega_ky`, `plot_omega_kx`, and
  `plot_contour_gamma_kx_ky` were already working correctly for
  multi-mode runs before this fix -- only `read_omega_t`, which none of
  those call, had this particular bug.)
- **Stella-version variable-name drift**: `get_Wenergy_t_zed_kx_ky`
  (`stella_diagnostics/physics/zonal_energy.py`) looks up the netCDF
  variable `Wenergy_g`, which recent stella builds don't write under
  that name -- confirmed against a real run (no `Wenergy_g`-like
  variable present at all). Unlike the other variable-name-drift cases
  below, no modern replacement variable name could be identified with
  confidence (nothing shape/name-matches closely enough to be sure), so
  this one is left broken rather than guessed at -- not exercised by
  any `example_plots/*.py` driver, only reachable via a direct
  `run.get_Wenergy_t_zed_kx_ky(...)` call.

  Other, similar drift was found and fixed this session, each verified
  against a real run written by a recent stella build: `qflx_kxky` (now
  `qflux_vs_kxkyzs` -- not `qflux_vs_kxkys`, a same-prefix but
  differently-shaped, zed-integrated sibling variable) in
  `read_flux_spectra` (and transitively `plot_flux_spectra`/
  `plot_flux_spectra_kx_ky`, which call it); `gzvs` (now `g2_vs_zvpas`)
  in `read_g_vs_zed`; and `gvmus`/`gvmus_Z`/`gvmus_NZ` (now
  `g2_vs_vpamus`/`g2nozonal_vs_vpamus`, with no direct zonal-only
  replacement -- derived as `g2_vs_vpamus - g2nozonal_vs_vpamus`,
  matching the pattern `plot_contour_gvmu_vpa` already used) in
  `get_gvpa_gmu`, `get_Evpa_Emu`, `get_n_T_vpa_mu`, and
  `plot_contour_gvmu_vpa`'s own `plot_diff` branch. All of these now
  try the older name first and fall back to the modern one.

  A related case, found and **fixed** while verifying against a higher
  -resolution real run: `quantities/registry.py`'s `get_quantity_zed_kx_ky`
  "pressure" branch read `run.ncdata.variables['pressure']` directly with
  no fallback, raising `KeyError` on stella builds that only write
  `pressure_perp` (not a combined `pressure`) -- confirmed against a real
  run. `quantities/realspace.py`'s independently-maintained copy of this
  same dispatch (`get_quantity_zed_x_y`, see the ~40-branch duplication
  note above) already had a `density`+`temperature`+`pressure_perp`
  reconstruction fallback for exactly this case; `registry.py`'s branch
  now has the same fallback, so both copies of the dispatch handle it
  consistently.

### Cleanup pass (code duplication, dead code, misleading names)

A systematic audit (not tied to any specific bug report) found and fixed
several more issues, verified against real run data throughout:

- **Two more `NameError` bugs**, both fixed: `physics/correlations.py`'s
  `plot_parallel_correlation_function` referenced an undefined `phi_sum`
  (the line above it computed `quantity_sum`) in its `zeta_max=True`
  branch; `quantities/registry.py`'s `"(1-Gamma0)phi"` quantity branch
  referenced an undefined `kperp2` (the line above it assigned
  `kperp2_zed_kx_ky`). Both crashed on every call that reached them.
- **`example_plots/plot_fluxes.py`**: hardcoded `axs[2].set_yscale("log")`
  on the heat-flux panel silently hid a real negative transient (Q dips
  to about -20 before recovering) -- same log-scale-hides-negative-values
  bug class as `rh_flux_scan.py`/`plot_correlation_func.py`, fixed the
  same way (`symlog`).
- **The `kxmin_filter`/`kxmax_filter`/`kymin_filter`/`kymax_filter`
  parameter family** (`quantities/realspace.py`, `physics/zonal_energy.py`,
  `physics/velocity_space.py`, `plotting/kspace_plots.py`,
  `plotting/realspace_plots.py`, `plotting/zed_plots.py`, and their
  `io/run.py` delegates) was renamed to
  `kx_lowpass_cutoff`/`kx_highpass_cutoff`/`ky_lowpass_cutoff`/
  `ky_highpass_cutoff`: the old names were backwards from their actual
  behavior -- `kxmin_filter` zeroed out `|kx| > kxmin_filter` (an upper/
  low-pass cutoff, not a "min"), and `kxmax_filter` zeroed out
  `|kx| < kxmax_filter` (a lower/high-pass cutoff, not a "max"). Two
  outlier default values in `zed_plots.py` (`1000`/`0` instead of the
  `inf`/`-1` used everywhere else for "disabled") were normalized to
  match at the same time.
- Extracted several duplicated computations into shared helpers (all
  verified bit-identical against real run data before/after, or by
  direct end-to-end re-run): `physics/zonal_energy.py`'s periodic
  centered-difference-along-zed (was written out 3 times) into
  `_periodic_zed_derivative`; `physics/correlations.py`'s O(N^2)
  periodic-shift correlation (written out once per x/y axis) into
  `_periodic_shift_correlation`; the "ensure omegas loaded" guard
  duplicated across `scan/omega_scan.py`'s three functions into
  `_ensure_omegas_loaded`; the resample-to-uniform-time step duplicated
  3x in `spectral/omega.py` into `_resample_uniform_time` (their
  subsequent FFT/filter steps have real differences -- FFT axis, omega
  sign convention, strict vs. non-strict filter bounds -- and were left
  un-merged); the vmin/vmax-sentinel-resolution step duplicated across
  3 pcolormesh plots in `zed_plots.py`/`realspace_plots.py` into
  `plotting/mpl_helpers.resolve_vmin_vmax`.
- Removed dead/commented-out scratch code across `scan/spectrum_scan.py`,
  `physics/correlations.py` (a never-finished, broken `"interpolate"`
  branch), `plotting/{realspace,kspace}_plots.py` (including a full
  duplicate of a live `ky_idx` dispatch block in `kspace_plots.py`),
  `spectral/{stats,omega}.py`, and `quantities/registry.py`.
- Renamed a few misleading names: `physics/fluxes.py`'s `flux_norm()`
  no longer shadows its own name with a same-named local variable;
  `physics/correlations.py`'s `plot_parallel_correlation_function` no
  longer reuses one variable (`quantity`) for both a string selector and
  the array it selects; `scan/run_collection.py`'s `dataObj`/
  `list_dataObj` (a pre-restructure holdover naming a `StellaRun`
  instance as if it might be a raw netCDF handle) is now `single_run`/
  `list_runs`, matching every other file's `run` convention.

Found but deliberately **not** fixed (flagged instead, since the fix
isn't obvious or the code path is unused by any driver):

- **`scan/omega_scan.py`'s `plot_omega_kx`**, `ky_idx="max"` branch:
  `idx_ky_max = np.argmax(run.omega_i[i])` doesn't depend on the loop
  variable `j` (the kx index) at all, so it's recomputed identically on
  every iteration and used as if it were a valid ky index despite being
  a flat index into the flattened `(ky,kx)` plane. `plot_omega_ky`'s
  equivalent `kx_idx="max"` branch does this correctly (`run.omega_i[i,j]`,
  properly indexed by both loop variables). Not exercised by any
  `example_plots/*.py` driver (the default is `ky_idx=0`).
- `run_collection.py`'s `load_omegas`/several `plot_contour_*`
  cross-run functions still raise `ValueError` (`np.array` on a ragged
  list) when combining runs whose `(kx,ky)` grids have different shapes
  -- confirmed while testing `RunCollection` against this session's two
  differently-resolved `stella_minimal_scan` runs together. Works fine
  for same-shape runs (the normal case); pre-existing, unrelated to any
  change made this session.
