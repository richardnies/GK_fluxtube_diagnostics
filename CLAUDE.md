# Adapting a new plotting script

The user periodically drops a new raw script into `example_plots/` written
in the old, pre-restructure style: `sys.path.append('/home/rnies/stella_diagnostics'); import stellaDiagnostics as sD`,
hardcoded dirnames/time ranges inline, `plt.rcParams.update({...})` at the
top, no config file. Convert it to match the ~35 already-converted scripts
in `example_plots/`. Checklist, roughly in order:

1. **Check whether the physics already exists.** Grep `stella_diagnostics/`
   (`StellaRun`'s methods in `io/run.py`, `scan/*.py`, `physics/*.py`) for
   the netCDF variables / method calls the raw script uses. If everything
   it needs is already a `StellaRun`/`RunCollection` method, this is pure
   wiring, no new physics -- see `plot_dEZ_kx.py`, `movie_avalanche.py`,
   `movie_quantity_3d_torus.py` for examples. If it computes something
   genuinely new (reads netCDF variables directly that nothing else in the
   package reads), extract that computation into
   `stella_diagnostics/physics/<module>.py` or `scan/<module>.py` as a
   small, `@cached`-decorated function taking `run` as its first argument,
   then add a matching delegate method on `StellaRun` in `io/run.py` --
   see `physics/correlations.py::get_perp_correlation_function` + its
   `io/run.py` delegate, or `physics/energy_transfer.py`, as templates.
   Only extract genuinely reusable computation; keep plotting/orchestration
   in the driver script itself.

2. **Swap imports.** Replace
   ```python
   dir_stella_diagnostics = '/home/rnies/stella_diagnostics'
   sys.path.append(dir_stella_diagnostics)
   import stellaDiagnostics as sD
   ```
   with
   ```python
   from stella_diagnostics.io.run import StellaRun
   from stella_diagnostics.scan.run_collection import RunCollection  # if multi-run
   ```
   and `sD.stellaDiagnostics(filename)` -> `StellaRun(filename, code=code)`.

3. **Swap style setup.** Replace the `plt.rcParams.update({...})` block
   with `from stella_diagnostics.plotting.mpl_helpers import
   set_default_style; set_default_style()`.

4. **Convert to config-driven.** Replace hardcoded dirname/time_min/
   time_max/etc with:
   ```python
   from stella_diagnostics.scan.config import load_scan_config
   if len(sys.argv) != 2:
       sys.exit(f"usage: python {sys.argv[0]} <config.py>")
   config = load_scan_config(sys.argv[1], required=(...))
   ```
   then `getattr(config, "field", default)` for everything, using the
   *current* hardcoded values as defaults. For single-run scripts, split
   `dirname`/`filename` (not one combined path):
   `StellaRun(config.dirname + "/" + getattr(config, "filename", "CBC"), code=...)`.
   For multi-run scripts, match whatever shape the underlying package
   function already expects (flat `dirnames` list vs nested
   `dirnames[row][col]` vs a `RunCollection` filenames_base list) -- don't
   invent a new shape or force two genuinely different shapes to share one
   field name (see `scan_config.py` vs `scan_config_grid.py` in
   `stella_minimal_scan/` for why those two stayed separate files).

5. **Canonical field vocabulary** -- reuse these names, don't invent
   synonyms (see `stella_diagnostics/__init__.py`'s
   naming-inconsistency-glossary docstring for the full rationale):
   - `time_min`/`time_max`: plain time range.
   - `time_avg` (window width) + `time_val_avg` (window center, default
     `None` = trailing window from the run's last sample): windowed
     average.
   - `dirname`/`filename`/`code`: single-run identity.
   - `kx_max`/`kx_min`: do **not** assume these mean the same thing across
     scripts -- check what the underlying function actually does with
     them before reusing the name for something new.

6. **Output location.** Every driver saves output relative to cwd, never
   by prefixing `config.dirname` into a figure/movie path -- that mixes
   output into a run's own `.out.nc`/`.fluxes`/`.omega` directory. Scripts
   are run from the *base* directory (one level up from any individual
   run), e.g. `python ../example_plots/foo.py run_tprim-4.2000/run_config.py`
   from inside `stella_minimal_scan/`.

7. **No `.dat` file caching.** If the raw script does its own
   `np.savetxt`/`np.loadtxt` hand-rolled caching, remove it -- wrap the
   underlying computation in `@cached` (`stella_diagnostics.io.cache`)
   instead, per point 1. `@cached` functions transparently skip
   recomputation when nothing relevant changed (keyed on params + the
   run's own source-file mtimes); a converted script should have no
   manual cache files to manage.

8. **Movies.** Use `stella_diagnostics.plotting.movies.render_movie`
   (fresh-figure-per-frame case) or call `ffmpeg_frames_to_video` directly
   if the script reuses one persistent figure across frames (see
   `movie_avalanche.py`, which redraws only part of a shared figure each
   frame for performance) -- never hand-roll
   `os.system("mkdir -p ...")`/`os.system("ffmpeg ...")`.

9. **Axis limits.** Don't hardcode `ax.set_ylim(...)`/`ax.set_xlim(...)`
   to a fixed numeric window. Default to `None` (let matplotlib
   autoscale) and expose the limit as an optional config field for the
   user to override. A fixed positive-only window on a log-scale axis
   silently produces a blank plot the moment the data goes negative --
   this happened twice in the same afternoon (`rh_flux_scan.py`,
   `plot_correlation_func.py`) before this rule existed.

10. **Module docstring.** Every converted script gets a docstring at the
    top: usage (`python <script>.py <config.py>`) and every config field
    it reads, required vs optional -- match the style of any
    already-converted script.

11. **Verify against real data.** `stella_minimal_scan/` has two real runs
    (`run_tprim-4.2000`, `run_tprim-6.7000`), each with its own
    `run_config.py` -- run the converted script against them if its data
    requirements fit (most single-run diagnostics do). If the script
    needs data `stella_minimal_scan` doesn't have (e.g. a
    `PiNZ_Kx`-style diagnostic needing a specific compile-time stella
    output flag, or a multi-run `tprim_scan`/`qinp_scan` directory
    layout), verify structurally instead: a synthetic netCDF fixture
    (see `tests/conftest.py`'s pattern) with the right variable
    names/shapes, checked for finite/sane output; or a symlinked
    directory tree matching the config's expected naming (see how
    `plot_correlation_func_perp.py`/`plot_correlation_func.py` were
    checked via a symlinked `run_tprim_val-4.2000` directory pointing at
    the real `run_tprim-4.2000`).

12. **Check for blank plots, not just exceptions.** "Runs without
    crashing" isn't sufficient -- render the actual figure (pymupdf) and
    look at it, or at minimum print the data values being plotted and
    sanity-check they fall inside whatever axis range will be shown.

13. **Update `README.md`'s script table** once a script is converted and
    verified (see the "Running the converted scripts" section).

14. **Flag, don't fix, unrelated pre-existing bugs** noticed along the
    way -- add a `# NOTE` comment explaining the bug precisely, and
    mention it to the user, rather than silently changing behavior
    outside the current task's scope.

Reference examples from the scripts converted so far: `plot_dEZ_kx.py`,
`movie_quantity_3d_torus.py`, `movie_avalanche.py` (pure wiring, no new
physics); `plot_correlation_func_perp.py` + `plot_correlation_func.py` +
`stella_diagnostics/physics/correlations.py::get_perp_correlation_function`
(new physics extraction, verified via a symlinked real run);
`plot_energyflux_Pi_Kx_Ky_NEW.py` +
`stella_diagnostics/physics/energy_transfer.py` (new physics extraction,
unverifiable against real data here -- structural/synthetic verification
only).
