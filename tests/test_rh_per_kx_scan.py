import numpy as np

from stella_diagnostics.scan.rh_per_kx_scan import (
    get_RH_per_kx_means,
    plot_RH_per_kx_summary_vs_kx,
    plot_RH_per_kx_summary_vs_time,
)


def test_get_RH_per_kx_means_shapes(synthetic_stella_run_with_rh):
    run = synthetic_stella_run_with_rh
    kx_all = run.ncdata["kx"][:]

    means = get_RH_per_kx_means(run, time_min=0, time_max=1e10, kx_max=1e4)

    assert set(means.keys()) == {
        "kx_all",
        "t",
        "E_RH_mean_kx",
        "P_RH_num_mean_kx",
        "P_RH_even_mean_kx",
        "P_RH_odd_mean_kx",
        "P_RH_phi_even_mean_kx",
        "P_RH_phi_odd_mean_kx",
        "P_RH_apar_even_mean_kx",
        "P_RH_apar_odd_mean_kx",
        "P_RH_bpar_even_mean_kx",
        "P_RH_bpar_odd_mean_kx",
        "P_RH_coll_even_mean_kx",
        "P_RH_coll_odd_mean_kx",
        "E_RH_t_sumkx",
        "P_RH_phi_even_t_sumkx",
        "P_RH_phi_odd_t_sumkx",
        "P_RH_apar_even_t_sumkx",
        "P_RH_apar_odd_t_sumkx",
        "P_RH_bpar_even_t_sumkx",
        "P_RH_bpar_odd_t_sumkx",
        "P_RH_coll_t_sumkx",
        "has_hyper",
    }
    assert means["kx_all"].shape == kx_all.shape
    for key in ("E_RH_mean_kx", "P_RH_even_mean_kx", "P_RH_odd_mean_kx"):
        assert means[key].shape == kx_all.shape
    assert means["t"].shape == means["E_RH_t_sumkx"].shape
    assert means["has_hyper"] is False

    # kx<=0 entries are never touched by the per-kx loop -> stay zero.
    assert np.all(means["E_RH_mean_kx"][kx_all <= 0] == 0)
    # at least one kx>0 entry got a real (nonzero) contribution.
    assert np.any(means["E_RH_mean_kx"][kx_all > 0] != 0)


def test_get_RH_per_kx_means_cache_hit_returns_same_result(synthetic_stella_run_with_rh):
    run = synthetic_stella_run_with_rh
    kwargs = dict(time_min=0, time_max=1e10, kx_max=1e4)

    first = get_RH_per_kx_means(run, **kwargs)
    second = get_RH_per_kx_means(run, **kwargs)

    for key in first:
        if key == "has_hyper":
            assert first[key] == second[key]
        else:
            np.testing.assert_array_equal(first[key], second[key])


def test_get_RH_per_kx_means_with_D_hyper(synthetic_stella_run_with_rh):
    run = synthetic_stella_run_with_rh
    means = get_RH_per_kx_means(run, time_min=0, time_max=1e10, kx_max=1e4, D_hyper=1e-3)

    assert means["has_hyper"] is True
    assert "P_RH_hyper_mean_kx" in means
    assert "P_RH_hyper_t_sumkx" in means


def test_summary_plots_run_without_error(synthetic_stella_run_with_rh):
    run = synthetic_stella_run_with_rh

    fig, axs = plot_RH_per_kx_summary_vs_time(run, time_min=0, time_max=1e10, kx_max=1e4)
    assert len(axs) == 2

    # E_RH_mean_kx is zero at the (unvisited) kx<=0 entries, so the
    # 1/E_RH_mean_kx normalisation in plot_RH_per_kx_summary_vs_kx divides
    # by zero there -- pre-existing original behavior, not something this
    # test should mask by changing the fixture's kx grid.
    with np.errstate(divide="ignore", invalid="ignore"):
        fig, axs = plot_RH_per_kx_summary_vs_kx(run, time_min=0, time_max=1e10, kx_max=1e4)
    assert len(axs) == 3
