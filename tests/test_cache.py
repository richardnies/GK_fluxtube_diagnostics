"""Data-free tests for stella_diagnostics.io.cache: cache hit/miss on
params/version/source-file changes, force bypass, clear_cache, the @cached
decorator, and round-tripping of single/tuple/scalar-mixed return values."""
import os
import time

import numpy as np
import pytest

from stella_diagnostics.io import cache


def _counting_compute(calls, value):
    def compute():
        calls["n"] += 1
        return value

    return compute


def test_cache_hit_on_identical_params(synthetic_stella_run):
    run = synthetic_stella_run
    calls = {"n": 0}
    compute = _counting_compute(calls, np.array([1.0, 2.0, 3.0]))

    r1 = cache.get_cached(run, "q", compute, params={"a": 1})
    r2 = cache.get_cached(run, "q", compute, params={"a": 1})

    assert calls["n"] == 1
    np.testing.assert_array_equal(r1, r2)


def test_cache_miss_on_different_params(synthetic_stella_run):
    run = synthetic_stella_run
    calls = {"n": 0}
    compute = _counting_compute(calls, np.array([1.0]))

    cache.get_cached(run, "q", compute, params={"time_avg": 10})
    cache.get_cached(run, "q", compute, params={"time_avg": 10})
    assert calls["n"] == 1

    cache.get_cached(run, "q", compute, params={"time_avg": 20})
    assert calls["n"] == 2


def test_cache_miss_on_different_version(synthetic_stella_run):
    run = synthetic_stella_run
    calls = {"n": 0}
    compute = _counting_compute(calls, np.array([1.0]))

    cache.get_cached(run, "q", compute, params={"a": 1}, version=0)
    cache.get_cached(run, "q", compute, params={"a": 1}, version=1)
    assert calls["n"] == 2


def test_force_always_recomputes(synthetic_stella_run):
    run = synthetic_stella_run
    calls = {"n": 0}
    compute = _counting_compute(calls, np.array([1.0]))

    cache.get_cached(run, "q", compute, params={"a": 1})
    cache.get_cached(run, "q", compute, params={"a": 1}, force=True)
    assert calls["n"] == 2


def test_source_file_mtime_invalidates_cache(synthetic_stella_run):
    run = synthetic_stella_run
    calls = {"n": 0}
    compute = _counting_compute(calls, np.array([1.0]))

    cache.get_cached(run, "q", compute, params={"a": 1})
    assert calls["n"] == 1

    # Simulate the simulation being restarted/extended: bump the source
    # netCDF file's mtime without changing params.
    time.sleep(0.01)
    os.utime(run.netcdf_file, None)

    cache.get_cached(run, "q", compute, params={"a": 1})
    assert calls["n"] == 2


def test_clear_cache_removes_file_and_forces_recompute(synthetic_stella_run):
    run = synthetic_stella_run
    calls = {"n": 0}
    compute = _counting_compute(calls, np.array([1.0]))

    cache.get_cached(run, "q", compute, params={"a": 1})
    n_removed = cache.clear_cache(run, name="q")
    assert n_removed == 1

    cache.get_cached(run, "q", compute, params={"a": 1})
    assert calls["n"] == 2


def test_clear_cache_all_names(synthetic_stella_run):
    run = synthetic_stella_run
    compute = _counting_compute({"n": 0}, np.array([1.0]))

    cache.get_cached(run, "q1", compute, params={})
    cache.get_cached(run, "q2", compute, params={})
    assert cache.clear_cache(run) == 2


def test_env_var_disables_cache(monkeypatch, synthetic_stella_run):
    run = synthetic_stella_run
    calls = {"n": 0}
    compute = _counting_compute(calls, np.array([1.0]))

    cache.get_cached(run, "q", compute, params={"a": 1})
    monkeypatch.setenv("STELLA_DIAGNOSTICS_NO_CACHE", "1")
    cache.get_cached(run, "q", compute, params={"a": 1})
    assert calls["n"] == 2


def test_round_trip_single_array(synthetic_stella_run):
    run = synthetic_stella_run
    value = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = cache.get_cached(run, "q", lambda: value, params={})
    cached_result = cache.get_cached(run, "q", lambda: value, params={})
    np.testing.assert_array_equal(result, value)
    np.testing.assert_array_equal(cached_result, value)


def test_round_trip_tuple_arrays_and_scalars(synthetic_stella_run):
    run = synthetic_stella_run
    value = (np.array([1.0, 2.0, 3.0]), 4.5, True, 7)

    cache.get_cached(run, "q", lambda: value, params={})
    out = cache.get_cached(run, "q", lambda: value, params={})

    arr, f, b, i = out
    np.testing.assert_array_equal(arr, value[0])
    assert isinstance(f, float) and f == 4.5
    assert isinstance(b, (bool, np.bool_)) and b == True  # noqa: E712
    assert isinstance(i, int) and i == 7


def test_round_trip_dict_arrays_and_scalars(synthetic_stella_run):
    run = synthetic_stella_run
    value = {"arr": np.array([1.0, 2.0, 3.0]), "flt": 4.5, "flag": True, "count": 7}

    cache.get_cached(run, "q", lambda: value, params={})
    out = cache.get_cached(run, "q", lambda: value, params={})

    assert set(out.keys()) == {"arr", "flt", "flag", "count"}
    np.testing.assert_array_equal(out["arr"], value["arr"])
    assert isinstance(out["flt"], float) and out["flt"] == 4.5
    assert isinstance(out["flag"], (bool, np.bool_)) and out["flag"] == True  # noqa: E712
    assert isinstance(out["count"], int) and out["count"] == 7


def test_dict_cache_hit_no_recompute(synthetic_stella_run):
    run = synthetic_stella_run
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"a": np.array([1.0]), "b": 2.0}

    cache.get_cached(run, "q", compute, params={"x": 1})
    cache.get_cached(run, "q", compute, params={"x": 1})
    assert calls["n"] == 1

    cache.get_cached(run, "q", compute, params={"x": 2})
    assert calls["n"] == 2


def test_cached_decorator_matches_direct_call(synthetic_stella_run):
    run = synthetic_stella_run
    calls = {"n": 0}

    @cache.cached(version=1)
    def get_q(run, x, y=2):
        calls["n"] += 1
        return np.array([x, y])

    a = get_q(run, 1, y=2)
    b = get_q(run, 1, 2)  # positional vs keyword, same logical call
    assert calls["n"] == 1
    np.testing.assert_array_equal(a, b)

    c = get_q(run, 1, y=3)
    assert calls["n"] == 2
    np.testing.assert_array_equal(c, np.array([1, 3]))


def test_cached_decorator_preserves_signature(synthetic_stella_run):
    import inspect

    @cache.cached()
    def get_q(run, x, y=2):
        return x + y

    sig = inspect.signature(get_q)
    assert list(sig.parameters) == ["run", "x", "y"]


def test_param_names_restricts_cache_key(synthetic_stella_run):
    run = synthetic_stella_run
    calls = {"n": 0}

    @cache.cached(param_names=["quantity"])
    def get_q(run, quantity, color=None):
        calls["n"] += 1
        return np.array([1.0])

    get_q(run, "phi", color="red")
    get_q(run, "phi", color="blue")  # cosmetic-only difference -> cache hit
    assert calls["n"] == 1

    get_q(run, "density", color="red")
    assert calls["n"] == 2


def test_corrupt_cache_file_falls_back_to_recompute(synthetic_stella_run):
    run = synthetic_stella_run
    calls = {"n": 0}
    compute = _counting_compute(calls, np.array([1.0]))

    cache.get_cached(run, "q", compute, params={"a": 1})
    key = cache.cache_key("q", {"a": 1}, 0)
    path = cache._cache_path(run, "q", key)
    with open(path, "wb") as f:
        f.write(b"not a valid npz file")

    cache.get_cached(run, "q", compute, params={"a": 1})
    assert calls["n"] == 2
