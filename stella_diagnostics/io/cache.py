"""Transparent disk cache for expensive per-run quantity computations.

Replaces the ad hoc "np.savetxt in one script, np.loadtxt in another, no
invalidation" pattern that had independently appeared four times across the
codebase (example_plots/movie_quantities_x_zed.py + plot_mean_quantities_x_zed.py,
and twice more inside stella_diagnostics/scan/spectrum_scan.py) with one
mechanism: no script is specially "the generator" or "the reader" of cached
data. Any call to get_cached()/@cached transparently returns a cached result
if the call's params and the run's source files haven't changed since it was
written, and transparently recomputes (and re-caches) if they have -- so e.g.
widening a time-averaging window just works, with no manual cache deletion.

Cache files are sibling files next to `run.filename_base`, following the
same convention as the `.out.nc`/`.omega`/`.fluxes` files StellaRun already
derives from it.
"""

import functools
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np

CACHE_SCHEMA_VERSION = 1

_SOURCE_FILE_ATTRS = ("netcdf_file", "fluxes_file", "omega_file")


def _canonicalize(obj):
    """Make `obj` JSON-serializable and order-independent (dict keys sorted,
    numpy scalars/arrays converted to plain python)."""
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _canonicalize(obj.tolist())
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def cache_key(name, params, version=0):
    """16-hex-char sha1 digest of (name, version, schema, canonicalized params)."""
    payload = json.dumps(
        {
            "name": name,
            "version": version,
            "schema": CACHE_SCHEMA_VERSION,
            "params": _canonicalize(params or {}),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _cache_path(run, name, key):
    return run.filename_base + "__cache_" + name + "_" + key + ".npz"


def _source_fingerprint(run):
    """Max mtime (ns) over whichever of the run's source files exist. None
    if none of them exist (e.g. a fabricated/synthetic run with no sibling
    files) -- callers must treat that as "no freshness information available"
    without crashing."""
    mtimes = []
    for attr in _SOURCE_FILE_ATTRS:
        path = getattr(run, attr, None)
        if path and os.path.exists(path):
            mtimes.append(os.stat(path).st_mtime_ns)
    return max(mtimes) if mtimes else None


def _is_cache_disabled():
    return os.environ.get("STELLA_DIAGNOSTICS_NO_CACHE", "") not in ("", "0")


def _pack_result(result):
    """-> (return_shape, scalar_mask, arrays) where arrays is a dict of
    array-name -> ndarray suitable for np.savez_compressed."""
    if isinstance(result, tuple):
        scalar_mask = [np.isscalar(x) or isinstance(x, (int, float, complex, bool)) for x in result]
        arrays = {f"out_{i}": np.asarray(x) for i, x in enumerate(result)}
        return "tuple", scalar_mask, arrays
    return "single", [False], {"out": np.asarray(result)}


def _unpack_result(npz, return_shape, scalar_mask):
    if return_shape == "single":
        arr = npz["out"]
        return arr.item() if scalar_mask[0] else arr
    out = []
    for i, is_scalar in enumerate(scalar_mask):
        arr = npz[f"out_{i}"]
        out.append(arr.item() if is_scalar else arr)
    return tuple(out)


def clear_cache(run, name=None):
    """Delete cache file(s) for `run`. If `name` is given, only entries for
    that quantity name are removed; otherwise every cache entry for this run
    is removed. Returns the number of files deleted."""
    parent = Path(run.filename_base).parent
    base = Path(run.filename_base).name
    marker = "__cache_" + (name + "_" if name else "")
    n = 0
    for p in parent.glob(base + marker + "*.npz"):
        p.unlink()
        n += 1
    return n


def get_cached(run, name, compute_fn, params=None, version=0, force=False):
    """Return compute_fn()'s result, using a cached copy on disk if one
    exists for this (name, version, params) and the run's source files
    haven't changed since it was written.

    compute_fn: zero-argument callable. Its return value may be a single
      ndarray, or a tuple mixing ndarrays and python scalars.
    params: dict of the call-site values that determine the numeric result
      (NOT cosmetic plot kwargs like color/label). Part of the cache key --
      different params means a different cache entry, so e.g. changing a
      time-averaging window automatically triggers a recompute instead of
      silently returning a stale result.
    version: bump manually when compute_fn's numerics change in a way that
      must invalidate old caches regardless of params.
    force: bypass any cached copy, always recompute and overwrite. Also
      forced on globally by env var STELLA_DIAGNOSTICS_NO_CACHE=1.
    """
    params = params or {}
    key = cache_key(name, params, version)
    path = _cache_path(run, name, key)
    fingerprint = _source_fingerprint(run)

    if not force and not _is_cache_disabled() and os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as npz:
                meta = json.loads(str(npz["__meta__"][0]))
                fresh = (
                    meta.get("schema") == CACHE_SCHEMA_VERSION
                    and meta.get("version") == version
                    and meta.get("params") == _canonicalize(params)
                    and (
                        fingerprint is None
                        or meta.get("source_fingerprint") is None
                        or meta["source_fingerprint"] >= fingerprint
                    )
                )
                if fresh:
                    return _unpack_result(npz, meta["return_shape"], meta["scalar_mask"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass  # corrupt/partial cache file -- fall through and recompute

    result = compute_fn()

    return_shape, scalar_mask, arrays = _pack_result(result)
    meta = {
        "name": name,
        "version": version,
        "schema": CACHE_SCHEMA_VERSION,
        "params": _canonicalize(params),
        "source_fingerprint": fingerprint,
        "created": time.time(),
        "return_shape": return_shape,
        "scalar_mask": scalar_mask,
    }
    # np.savez_compressed appends ".npz" to string filenames that don't
    # already end with it, so write through an explicit file object to
    # keep the exact temp-file name we chose for the atomic rename below.
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as fh:
        np.savez_compressed(fh, __meta__=np.array([json.dumps(meta)]), **arrays)
    os.replace(tmp_path, path)

    return result


def cached(name=None, version=0, param_names=None):
    """Decorator for a free function f(run, *args, **kwargs) that wraps it
    with get_cached(). `run` (the first argument) is excluded from the cache
    key automatically; param_names optionally restricts which of the
    remaining bound arguments participate in the key (default: all of
    them). Positional and keyword calls that bind to the same logical
    arguments produce the same cache key, via inspect.signature binding.
    """

    def decorator(f):
        cache_name = name or f.__name__
        sig = inspect.signature(f)

        @functools.wraps(f)
        def wrapper(run, *args, **kwargs):
            bound = sig.bind(run, *args, **kwargs)
            bound.apply_defaults()
            call_args = dict(bound.arguments)
            run_param_name = next(iter(sig.parameters))
            call_args.pop(run_param_name, None)
            if param_names is not None:
                call_args = {k: v for k, v in call_args.items() if k in param_names}

            def compute_fn():
                return f(run, *args, **kwargs)

            return get_cached(run, cache_name, compute_fn, params=call_args, version=version)

        return wrapper

    return decorator
