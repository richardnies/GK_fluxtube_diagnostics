"""Shared fixtures for the mock-netCDF smoke tests.

There is no real STELLA output available in this environment, so these
fixtures build a minimal synthetic ``code="stella"`` dataset -- enough
to exercise the constructor, the core grid readers, and a couple of
real analysis code paths end-to-end. It intentionally does not cover
every quantity/branch (e.g. the ~40-way quantity dispatch in
quantities/registry.py and quantities/realspace.py needs many more
netCDF variables than are worth fabricating here); see the README's
"Manual verification required" section for what still needs checking
against a real run.
"""
import numpy as np
import netCDF4 as nc4
import pytest


@pytest.fixture
def synthetic_stella_run(tmp_path):
    from stella_diagnostics.io.run import StellaRun

    filename_base = str(tmp_path / "run")
    n_t, n_kx, n_ky, n_zed, n_species, n_tube = 6, 5, 4, 9, 1, 1
    rng = np.random.default_rng(0)

    ncpath = filename_base + ".out.nc"
    ds = nc4.Dataset(ncpath, "w")
    ds.createDimension("t", n_t)
    ds.createDimension("kx", n_kx)
    ds.createDimension("ky", n_ky)
    ds.createDimension("zed", n_zed)
    ds.createDimension("species", n_species)
    ds.createDimension("tube", n_tube)
    ds.createDimension("theta0", 1)
    ds.createDimension("ri", 2)

    v = ds.createVariable("t", "f8", ("t",))
    v[:] = np.linspace(0, 50, n_t)

    kx_vals = np.linspace(-2, 2, n_kx)
    kx_vals[kx_vals == 0] = 1e-6  # avoid divide-by-zero in read_avg_kx_rhoi
    v = ds.createVariable("kx", "f8", ("kx",))
    v[:] = kx_vals

    v = ds.createVariable("ky", "f8", ("ky",))
    v[:] = np.linspace(0, 2, n_ky)

    zed_vals = np.linspace(-np.pi, np.pi, n_zed)
    v = ds.createVariable("zed", "f8", ("zed",))
    v[:] = zed_vals

    v = ds.createVariable("theta0", "f8", ("theta0",))
    v[:] = [0.0]

    v = ds.createVariable("shat", "f8", ())
    v[...] = 0.8

    v = ds.createVariable("phi2_vs_kxky", "f8", ("t", "kx", "ky"))
    v[:] = rng.random((n_t, n_kx, n_ky)) + 0.1

    v = ds.createVariable("bmag", "f8", ("zed",))
    v[:] = 1.0 + 0.1 * np.cos(zed_vals)

    v = ds.createVariable("gradpar", "f8", ("zed",))
    v[:] = np.ones(n_zed)

    v = ds.createVariable("grho", "f8", ("zed", "tube"))
    v[:] = np.ones((n_zed, n_tube))

    # phi_vs_t indexed as [t, tube, zed, kx, ky, ri] (see
    # stella_diagnostics.plotting.zed_plots.read_phi_vs_zed).
    v = ds.createVariable("phi_vs_t", "f8", ("t", "tube", "zed", "kx", "ky", "ri"))
    v[:] = rng.random((n_t, n_tube, n_zed, n_kx, n_ky, 2))

    ds.close()

    # .vmec.geo: first line has >=4 whitespace-separated numeric columns
    # (safety_factor at [1], aspect_ratio at [0]*[3]), then a header line,
    # then per-zed rows where column index 1 is used as "zed" when
    # zed_times_nfield_periods=True (see stella_diagnostics.grid /
    # plotting.zed_plots).
    geo_path = filename_base + ".vmec.geo"
    with open(geo_path, "w") as f:
        f.write("1.0 1.4 0.0 1.0\n")
        f.write("# header\n")
        for z in zed_vals:
            f.write(f"0.0 {z} " + " ".join(["0.0"] * 14) + "\n")

    # .fluxes: [time, pflx*ns, vflx*ns, qflx*ns] columns, one row per t
    # (see stella_diagnostics.physics.fluxes.get_fluxes_over_time).
    fluxes_path = filename_base + ".fluxes"
    fluxes = np.zeros((n_t, 1 + 3 * n_species))
    fluxes[:, 0] = np.linspace(0, 50, n_t)
    fluxes[:, 1] = rng.random(n_t)        # pflx
    fluxes[:, 2] = rng.random(n_t)        # vflx
    fluxes[:, 3] = rng.random(n_t) + 0.5  # qflx
    np.savetxt(fluxes_path, fluxes)

    return StellaRun(filename_base, code="stella")
