"""Verifies stella_diagnostics/spectral/omega.py's handling of the three
possible .omega ascii file layouts stella can write (confirmed directly
against the real STELLA Fortran source, diagnostics_omega.f90's
open_omega_ascii_file/write_omega_to_ascii_file):

  - 7 columns (write_omega_vs_kxky and write_omega_avg_vs_kxky both true):
    [time ky kx Re(om) Im(om) Re(omavg) Im(omavg)]
  - 5 columns, avg-only (write_omega_avg_vs_kxky only):
    [time ky kx Re(omavg) Im(omavg)]
  - 5 columns, instantaneous-only (write_omega_vs_kxky only):
    [time ky kx Re(om) Im(om)]

The two 5-column layouts have identical shape but different content, and
are told apart only by header text -- these tests cover both.
"""
import numpy as np
import pytest

from stella_diagnostics.spectral.omega import _read_omega_ascii_file, read_data_omega_k


def _write_omega_file(path, kx, ky, times, ncols, header):
    with open(path, "w") as f:
        f.write(header + "\n")
        for t in times:
            for iky, ky_val in enumerate(ky):
                for ikx, kx_val in enumerate(kx):
                    row = [t, ky_val, kx_val]
                    # Re/Im "raw" value -- a simple deterministic function of
                    # (t, ky, kx) so tests can check exact values back out.
                    raw_r, raw_i = t * 0.1 + ky_val, t * 0.1 + kx_val
                    if ncols == 7:
                        avg_r, avg_i = raw_r + 100, raw_i + 100
                        row += [raw_r, raw_i, avg_r, avg_i]
                    elif ncols == 5 and "omavg" in header.lower():
                        avg_r, avg_i = raw_r + 100, raw_i + 100
                        row += [avg_r, avg_i]
                    else:
                        row += [raw_r, raw_i]
                    f.write(" ".join("%.8e" % v for v in row) + "\n")


HEADER_7COL   = "#time ky kx Re[om] Im[om] Re[omavg] Im[omavg]"
HEADER_5AVG   = "#time ky kx Re[omavg] Im[omavg]"
HEADER_5RAW   = "#time ky kx frequency growth rate"


@pytest.mark.parametrize("ncols,header", [(7, HEADER_7COL), (5, HEADER_5AVG), (5, HEADER_5RAW)])
def test_format_detection(synthetic_stella_run, ncols, header):
    run = synthetic_stella_run
    kx, ky, _ = run.get_kx_ky_zed()
    _write_omega_file(run.omega_file, kx, ky, times=[10.0, 20.0], ncols=ncols, header=header)

    omega_data, has_raw, has_avg = _read_omega_ascii_file(run)
    assert omega_data.shape == (2, len(ky), len(kx), ncols)

    if ncols == 7:
        assert has_raw and has_avg
    elif "omavg" in header.lower():
        assert has_avg and not has_raw
    else:
        assert has_raw and not has_avg


def test_7col_om_avg_selects_avg_columns(synthetic_stella_run):
    run = synthetic_stella_run
    kx, ky, _ = run.get_kx_ky_zed()
    _write_omega_file(run.omega_file, kx, ky, times=[10.0, 20.0], ncols=7, header=HEADER_7COL)

    time, ky_out, kx_out, omega_r, omega_i = read_data_omega_k(run, timestep=-1, om_avg=True, check_convergence=False)
    assert np.allclose(omega_r, 20.0 * 0.1 + ky_out + 100)

    time, ky_out, kx_out, omega_r, omega_i = read_data_omega_k(run, timestep=-1, om_avg=False, check_convergence=False)
    assert np.allclose(omega_r, 20.0 * 0.1 + ky_out)


def test_5col_avg_only_rejects_om_avg_false(synthetic_stella_run):
    run = synthetic_stella_run
    kx, ky, _ = run.get_kx_ky_zed()
    _write_omega_file(run.omega_file, kx, ky, times=[10.0, 20.0], ncols=5, header=HEADER_5AVG)

    time, ky_out, kx_out, omega_r, omega_i = read_data_omega_k(run, timestep=-1, om_avg=True, check_convergence=False)
    assert np.allclose(omega_r, 20.0 * 0.1 + ky_out + 100)

    with pytest.raises(ValueError, match="instantaneous"):
        read_data_omega_k(run, timestep=-1, om_avg=False, check_convergence=False)


def test_5col_raw_only_rejects_om_avg_true(synthetic_stella_run):
    run = synthetic_stella_run
    kx, ky, _ = run.get_kx_ky_zed()
    _write_omega_file(run.omega_file, kx, ky, times=[10.0, 20.0], ncols=5, header=HEADER_5RAW)

    time, ky_out, kx_out, omega_r, omega_i = read_data_omega_k(run, timestep=-1, om_avg=False, check_convergence=False)
    assert np.allclose(omega_r, 20.0 * 0.1 + ky_out)

    with pytest.raises(ValueError, match="time-averaged"):
        read_data_omega_k(run, timestep=-1, om_avg=True, check_convergence=False)


def test_read_omega_t_multimode_all_formats(synthetic_stella_run):
    from stella_diagnostics.spectral.omega import read_omega_t

    run = synthetic_stella_run
    kx, ky, _ = run.get_kx_ky_zed()
    times = [10.0, 20.0, 30.0]

    for ncols, header in [(7, HEADER_7COL), (5, HEADER_5AVG), (5, HEADER_5RAW)]:
        _write_omega_file(run.omega_file, kx, ky, times=times, ncols=ncols, header=header)
        time, omega_r, omega_i = read_omega_t(run)
        assert time.shape == (3,)
        assert omega_r.shape == (3, len(ky), len(kx))
        assert np.allclose(time, times)
