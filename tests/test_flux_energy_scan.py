"""Data-free tests for stella_diagnostics.scan.flux_energy_scan: the
extracted plot_flux_time.py analysis logic, exercised end-to-end against a
synthetic multi-run scan directory (see tests/conftest.py's
synthetic_scan_dirs fixture), plus verification that the per-run computation
is actually cached."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stella_diagnostics.scan.flux_energy_scan as flux_energy_scan
from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.scan.flux_energy_scan import (
    _get_flux_energy_vs_time,
    plot_qflx_and_energy_vs_time,
)


def test_plot_qflx_and_energy_vs_time_all_runs_plotted(synthetic_scan_dirs):
    fig, ax = plot_qflx_and_energy_vs_time(
        synthetic_scan_dirs, labels=["a", "b", "c"], filename="CBC"
    )
    # 3 lines per run (1 species each in the synthetic fixture): Qflx, E_phi, E_upar
    assert len(ax.get_lines()) == 3 * len(synthetic_scan_dirs)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    # run label "a" is a prefix (species name is always appended, e.g. "a Species 0")
    assert any(l.startswith("a") for l in legend_labels)
    assert any("E_{\\varphi}" in l for l in legend_labels)
    plt.close(fig)


def test_plot_qflx_and_energy_vs_time_skip_phi2(synthetic_scan_dirs):
    fig, ax = plot_qflx_and_energy_vs_time(
        synthetic_scan_dirs, labels=["a", "b", "c"], filename="CBC", skip_phi2=True
    )
    assert len(ax.get_lines()) == len(synthetic_scan_dirs)
    plt.close(fig)


def test_plot_qflx_and_energy_vs_time_plot_ratio(synthetic_scan_dirs):
    fig, ax = plot_qflx_and_energy_vs_time(
        synthetic_scan_dirs, labels=["a", "b", "c"], filename="CBC", plot_ratio=True
    )
    # 2 lines per run: Qflx, ratio (no separate E_upar line)
    assert len(ax.get_lines()) == 2 * len(synthetic_scan_dirs)
    plt.close(fig)


def test_plot_qflx_and_energy_vs_time_no_labels(synthetic_scan_dirs):
    fig, ax = plot_qflx_and_energy_vs_time(synthetic_scan_dirs, filename="CBC")
    assert len(ax.get_lines()) == 3 * len(synthetic_scan_dirs)
    plt.close(fig)


def test_plot_qflx_and_energy_vs_time_skips_failed_run(synthetic_scan_dirs, capsys):
    dirnames = synthetic_scan_dirs + ["/nonexistent/path"]
    fig, ax = plot_qflx_and_energy_vs_time(dirnames, filename="CBC")
    # the bad dir is skipped, not raised
    assert len(ax.get_lines()) == 3 * len(synthetic_scan_dirs)
    plt.close(fig)


def test_get_flux_energy_vs_time_shapes(synthetic_scan_dirs):
    run = StellaRun(synthetic_scan_dirs[0] + "/CBC", code="stella")
    time, qflx, e_phi, e_upar = _get_flux_energy_vs_time(run, Q_div=10)
    assert time.shape == qflx.shape == e_phi.shape == e_upar.shape


def test_get_flux_energy_vs_time_skip_phi2_returns_none(synthetic_scan_dirs):
    run = StellaRun(synthetic_scan_dirs[0] + "/CBC", code="stella")
    time, qflx, e_phi, e_upar = _get_flux_energy_vs_time(run, skip_phi2=True)
    assert e_phi is None and e_upar is None


def test_get_flux_energy_vs_time_is_cached(synthetic_scan_dirs, monkeypatch):
    run = StellaRun(synthetic_scan_dirs[0] + "/CBC", code="stella")

    calls = {"n": 0}
    real_iv = flux_energy_scan.specialfunc.iv

    def counting_iv(*args, **kwargs):
        calls["n"] += 1
        return real_iv(*args, **kwargs)

    monkeypatch.setattr(flux_energy_scan.specialfunc, "iv", counting_iv)

    from stella_diagnostics.io.cache import clear_cache

    clear_cache(run)

    r1 = _get_flux_energy_vs_time(run, Q_div=10)
    n_after_first = calls["n"]
    assert n_after_first > 0

    r2 = _get_flux_energy_vs_time(run, Q_div=10)
    assert calls["n"] == n_after_first  # cache hit, no new iv() calls
    np.testing.assert_array_equal(r1[1], r2[1])

    r3 = _get_flux_energy_vs_time(run, Q_div=20)  # different param -> recompute
    assert calls["n"] > n_after_first
    assert not np.allclose(r1[1], r3[1])  # different Q_div -> different qflx
