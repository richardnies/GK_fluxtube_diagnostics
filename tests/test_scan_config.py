"""Data-free tests for stella_diagnostics.scan.config: dynamic loading of
scan-definition config files, and glob+regex-based run directory discovery."""
import os

import pytest

from stella_diagnostics.scan.config import discover_runs, load_scan_config


def _write_config(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return path


def test_load_scan_config_happy_path(tmp_path):
    path = _write_config(
        tmp_path,
        "cfg.py",
        'dirnames = ["a", "b"]\nlabels = ["A", "B"]\nylim = [1e-3, 1e2]\n',
    )
    config = load_scan_config(path)
    assert config.dirnames == ["a", "b"]
    assert config.labels == ["A", "B"]
    assert config.ylim == [1e-3, 1e2]


def test_load_scan_config_optional_field_absent(tmp_path):
    path = _write_config(tmp_path, "cfg.py", 'dirnames = ["a"]\n')
    config = load_scan_config(path)
    assert getattr(config, "labels", None) is None


def test_load_scan_config_missing_required_field_raises(tmp_path):
    path = _write_config(tmp_path, "cfg.py", 'labels = ["A"]\n')
    with pytest.raises(ValueError, match="dirnames"):
        load_scan_config(path)


def test_load_scan_config_custom_required(tmp_path):
    path = _write_config(tmp_path, "cfg.py", 'dirnames = ["a"]\n')
    with pytest.raises(ValueError, match="labels"):
        load_scan_config(path, required=("dirnames", "labels"))


def test_load_scan_config_two_configs_dont_collide(tmp_path):
    # Each config is dynamically imported as its own module -- make sure
    # loading two different config files with the same top-level names
    # doesn't leak state between them (e.g. via sys.modules caching).
    path_a = _write_config(tmp_path, "cfg_a.py", 'dirnames = ["a"]\n')
    path_b = _write_config(tmp_path, "cfg_b.py", 'dirnames = ["b"]\n')
    config_a = load_scan_config(path_a)
    config_b = load_scan_config(path_b)
    assert config_a.dirnames == ["a"]
    assert config_b.dirnames == ["b"]


def _make_run_dirs(tmp_path, names):
    base = tmp_path / "scan"
    base.mkdir()
    for name in names:
        (base / name).mkdir()
    return base


def test_discover_runs_plain_list(tmp_path):
    base = _make_run_dirs(tmp_path, ["run_a", "run_b", "not_a_run"])
    result = discover_runs(base, pattern="run_*")
    assert [os.path.basename(p) for p in result] == ["run_a", "run_b"]


def test_discover_runs_with_param_regex_sorted_by_value(tmp_path):
    base = _make_run_dirs(
        tmp_path, ["run_tprim-6.3000", "run_tprim-4.2000", "run_tprim-4.9000"]
    )
    result = discover_runs(base, pattern="run_*", param_regex=r"tprim-([0-9.eE+-]+)")
    values = [v for _, v in result]
    assert values == sorted(values)
    assert values == pytest.approx([4.2, 4.9, 6.3])


def test_discover_runs_param_regex_no_sort(tmp_path):
    base = _make_run_dirs(tmp_path, ["run_tprim-6.3000", "run_tprim-4.2000"])
    result = discover_runs(base, pattern="run_*", param_regex=r"tprim-([0-9.eE+-]+)", sort=False)
    # alphabetical glob order, not sorted by value
    assert [os.path.basename(p) for p, _ in result] == sorted(
        ["run_tprim-6.3000", "run_tprim-4.2000"]
    )


def test_discover_runs_exclude_string(tmp_path):
    base = _make_run_dirs(tmp_path, ["run_tprim-4.2000", "run_tprim-4.2000_NL"])
    result = discover_runs(base, pattern="run_*", exclude="_NL")
    assert [os.path.basename(p) for p in result] == ["run_tprim-4.2000"]


def test_discover_runs_exclude_list(tmp_path):
    base = _make_run_dirs(tmp_path, ["run_a", "run_b", "run_c"])
    result = discover_runs(base, pattern="run_*", exclude=["_a", "_c"])
    assert [os.path.basename(p) for p in result] == ["run_b"]


def test_discover_runs_non_matching_directory_raises_clear_error(tmp_path):
    base = _make_run_dirs(tmp_path, ["run_tprim-4.2000", "run_other"])
    with pytest.raises(ValueError, match="run_other"):
        discover_runs(base, pattern="run_*", param_regex=r"tprim-([0-9.eE+-]+)")


def test_discover_runs_empty_base_dir(tmp_path):
    base = tmp_path / "empty_scan"
    base.mkdir()
    assert discover_runs(base, pattern="run_*") == []
