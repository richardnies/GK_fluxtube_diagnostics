"""Kx/Ky-resolved nonlinear energy transfer: the "T(Kx)"/"T(Ky)" spectral
transfer functions PiNZ (nonzonal), PiZ_kxadv (zonal, kx-advection,
Kx-by-kx resolved), and the heat-flux-driven Kx/Ky transfer contribution.

Reads the PiNZ_Kx/PiNZ_Ky/PiZ_kxadv_Kx netCDF diagnostics directly (not
written by every stella build -- requires the corresponding compile-time
diagnostic output to be enabled) plus qflx_kxky. No stella_minimal_scan
run has these variables, so this could only be verified by careful
reading against the original example_plots/plot_energyflux_Pi_Kx_Ky_NEW.py
this was extracted from, not by a real-data numeric comparison.
"""

import numpy as np

from stella_diagnostics.io.cache import cached


@cached(version=1)
def get_energy_transfer_kx_ky(run, time_min, time_max, time_idx_skip=1):
    """dict of time-averaged Kx/Ky-resolved nonlinear energy transfer
    quantities for one run (`time_max=None` evaluates only the single
    frame at `time_min`):

    - Kx_vals/Ky_vals: the kx/ky>=0 grids
    - PiNZ_Kx/PiNZ_Ky: nonzonal transfer vs Kx/Ky
    - PiZ_Kx_kxadv: zonal kx-advection transfer, Kx (source) x kx (target)
    - dKx_PiQ_Kx/dKy_PiQ_Ky: heat-flux-driven Kx/Ky transfer contribution
      (NaN for a given time frame if qflx_kxky isn't available there)
    - tprim
    """
    time_all = run.get_time_array()
    time_idx_min = run.get_time_idx(time_min)
    if time_max is None:
        time_idx_vals = np.array([time_idx_min])
    else:
        time_idx_max = run.get_time_idx(time_max)
        time_idx_vals = np.arange(time_idx_min, time_idx_max, time_idx_skip)
    Ntime = len(time_idx_vals)

    kx_all = run.ncdata.variables['kx'][:]
    ky_all = run.ncdata.variables['ky'][:]
    Kx_vals = kx_all[kx_all >= 0]
    Ky_vals = ky_all[ky_all >= 0]
    NKx = len(Kx_vals)
    NKy = len(Ky_vals)
    tprim = run.ncdata.variables['tprim'][0]

    PiNZ_Kx_time = np.zeros((NKx, Ntime))
    PiNZ_Ky_time = np.zeros((NKy, Ntime))
    PiZ_Kx_kxadv_time = np.zeros((NKx, NKx, Ntime))
    dKx_PiQ_Kx_time = np.zeros((NKx, Ntime))
    dKy_PiQ_Ky_time = np.zeros((NKy, Ntime))

    for i_time_idx, time_idx in enumerate(time_idx_vals):
        try:
            qflx_kx_ky = np.sum(run.ncdata.variables['qflx_kxky'][time_idx, 0, 0, :, :, :], axis=0)
            load_Q = True
        except Exception:
            load_Q = False

        PiNZ_Kx_time[:, i_time_idx] = -run.ncdata.variables['PiNZ_Kx'][time_idx, 0, 0, :NKx]
        PiNZ_Ky_time[:, i_time_idx] = -run.ncdata.variables['PiNZ_Ky'][time_idx, 0, 0, :NKy]
        PiZ_Kx_kxadv_time[:, :, i_time_idx] = -run.ncdata.variables['PiZ_kxadv_Kx'][time_idx, 0, 0, :NKx, :NKx]

        for i_kx in range(NKx):
            dKx_PiQ_Kx_time[i_kx, i_time_idx] = 4 * tprim * np.sum(qflx_kx_ky[i_kx, :]) if load_Q else np.nan

        for i_ky in range(NKy):
            dKy_PiQ_Ky_time[i_ky, i_time_idx] = 2 * tprim * np.sum(qflx_kx_ky[:, i_ky]) if load_Q else np.nan

    dt_vals = np.gradient(time_all)[time_idx_vals]
    PiNZ_Kx = np.sum(PiNZ_Kx_time * dt_vals[None, :], axis=1) / np.sum(dt_vals)
    PiNZ_Ky = np.sum(PiNZ_Ky_time * dt_vals[None, :], axis=1) / np.sum(dt_vals)
    PiZ_Kx_kxadv = np.sum(PiZ_Kx_kxadv_time * dt_vals[None, None, :], axis=2) / np.sum(dt_vals)
    dKx_PiQ_Kx = np.sum(dKx_PiQ_Kx_time * dt_vals[None, :], axis=1) / np.sum(dt_vals)
    dKy_PiQ_Ky = np.sum(dKy_PiQ_Ky_time * dt_vals[None, :], axis=1) / np.sum(dt_vals)

    return {
        "time_idx_vals": time_idx_vals,
        "Kx_vals": Kx_vals, "Ky_vals": Ky_vals,
        "PiNZ_Kx": PiNZ_Kx, "PiNZ_Ky": PiNZ_Ky, "PiZ_Kx_kxadv": PiZ_Kx_kxadv,
        "dKx_PiQ_Kx": dKx_PiQ_Kx, "dKy_PiQ_Ky": dKy_PiQ_Ky,
        "tprim": tprim,
    }
