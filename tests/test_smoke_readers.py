"""Exercises real (non-mocked) code paths against a synthetic netCDF
dataset built in conftest.py -- catches array-shape/indexing/variable-
name bugs introduced by the restructure, without needing real STELLA
output. Does not validate the physics/numbers themselves."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_construction(synthetic_stella_run):
    run = synthetic_stella_run
    assert run.code == "stella"
    assert run.safety_factor == 1.4


def test_grid_readers(synthetic_stella_run):
    run = synthetic_stella_run
    kx, ky, zed = run.get_kx_ky_zed()
    assert len(kx) == 5
    assert len(ky) == 4
    assert len(zed) == 9

    time = run.get_time_array()
    assert len(time) == 6

    idx = run.get_time_idx(time[2])
    assert idx == 2

    weight = run.dl_over_B_avg()
    assert weight.shape == zed.shape


def test_read_avg_kx_rhoi(synthetic_stella_run):
    run = synthetic_stella_run
    kx_rhoi_o, time = run.read_avg_kx_rhoi()
    assert len(kx_rhoi_o) == len(time) == 6


def test_flux_norm_and_fluxes_over_time(synthetic_stella_run):
    run = synthetic_stella_run
    norm = run.flux_norm()
    assert norm > 0

    pflx, vflx, qflx, time = run.get_fluxes_over_time()
    assert len(qflx) == len(time)


def test_plot_flux_over_time(synthetic_stella_run):
    run = synthetic_stella_run
    axs = run.plot_flux_over_time()
    assert axs is not None
    plt.close("all")


def test_run_collection_wraps_run(synthetic_stella_run, tmp_path):
    from stella_diagnostics.scan.run_collection import RunCollection

    # RunCollection takes filenames, not StellaRun objects -- reconstruct
    # from the same filename_base the fixture already set up.
    scan = RunCollection([synthetic_stella_run.filename_base], labels=["run"])
    assert len(scan.list_dataObj) == 1

    ax = scan.plot_phi_vs_zed()
    assert ax is not None
    plt.close("all")
