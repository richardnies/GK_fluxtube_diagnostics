"""Growth rate / zonal ExB shear / Rosenbluth-Hinton power-transfer
diagnostics for one run, and a many-run "R/L_T scan" comparison built from
them -- historically the "Dimits shift" study, but the functions here are
general-purpose (no "Dimits" naming below): they estimate zonal-flow-related
transport quantities for any gradient-driven turbulence run.

Extracted from example_plots/get_Dimits.py and plot_param_scan_Dimits.py.
Per the request that motivated this extraction, the computation is split
into several independently-@cached functions rather than one big one: each
one succeeds or fails on its own, so plot_zonal_flow_scan (which calls all
of them per run) can still plot whatever subset of panels a given run's
data supports, instead of an all-or-nothing per-run failure. This also
replaces the original's hand-rolled ``data_Dimits.json`` cache (a single
flat dict covering the whole computation) transparently -- no more
generate-then-separately-read script pair.
"""

import matplotlib.pyplot as plt
import numpy as np

from stella_diagnostics.io.cache import cached
from stella_diagnostics.io.codes import get_rho_label, get_vt_label
from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.physics.rosenbluth_hinton import get_P_RH_breakdown


@cached(version=2)
def get_growth_rate_from_flux(run, time_max=1e10, qflx_rel_idx_min=1e-7, qflx_rel_idx_max=1e-3, time_val_avg=None, time_avg=5) -> dict:
    """Max linear growth rate estimated from the heat-flux rise between the
    times it crosses qflx_rel_idx_min/qflx_rel_idx_max of its peak, plus the
    saturated (or windowed, if time_val_avg is given) heat-flux mean/std.
    Cheapest of the four get_* functions here (only needs .fluxes), and
    the one most likely to still succeed when a run is otherwise unusable
    for the rest of this module.

    Also returns the underlying qflx(t) trace and idx_min/idx_max (not
    just the derived scalars), since plot_zonal_shear_diagnostic_page's
    Q(t) panel plots the trace itself plus the exponential-growth
    reference line through those two indices -- version bumped to 2 since
    this widens the return dict's keys.
    """
    tprim = float(run.ncdata.variables["tprim"][0])
    qinp = float(run.ncdata.variables["q"].getValue())

    _, _, qflx, time = run.get_fluxes_over_time(load_from_nc=True)

    qflx = qflx[time < time_max]
    time = time[time < time_max]

    if time_val_avg is None:
        qflx_avg = np.mean(qflx[time > time[-1] - time_avg])
        qflx_std = np.std(qflx[time > time[-1] - time_avg])
    else:
        qflx_avg = np.mean(qflx[(time > time_val_avg - time_avg / 2) & (time < time_val_avg + time_avg / 2)])
        qflx_std = np.std(qflx[(time > time_val_avg - time_avg / 2) & (time < time_val_avg + time_avg / 2)])

    idx_qflx_max = max(np.argmax(qflx), 5)
    idx_max = np.argmin(np.abs(qflx[:idx_qflx_max] - qflx_rel_idx_max * qflx.max()))
    idx_min = np.argmin(np.abs(qflx[:idx_qflx_max] - qflx_rel_idx_min * qflx.max()))

    gamma_lin_max = 0.5 * np.log(qflx_rel_idx_max / qflx_rel_idx_min) / (time[idx_max] - time[idx_min])

    return {
        "tprim": tprim,
        "qinp": qinp,
        "gamma_lin_max": gamma_lin_max,
        "qflx_avg": qflx_avg,
        "qflx_std": qflx_std,
        "time": time,
        "qflx": qflx,
        "idx_min": idx_min,
        "idx_max": idx_max,
    }


def estimate_eps_from_bmag(run):
    """Inverse-aspect-ratio estimate from the min/max of 1/bmag over zed.
    Trivial arithmetic on an already-loaded netCDF variable -- not cached."""
    bmag = run.ncdata.variables["bmag"][:]
    bmag_inv = 1 / bmag
    return (bmag_inv.max() - bmag_inv.min()) / (bmag_inv.max() + bmag_inv.min())


@cached(version=1)
def get_zonal_shear_time_series(run, time_idx_skip=10, exp_avg=2) -> dict:
    """Zonal shear/flow/potential-fluctuation RMS-over-x time series
    (sampled every time_idx_skip-th timestep across the whole run) --
    the data behind get_Dimits.py's Q(t) diagnostic panel, which overlays
    these traces on the heat-flux time trace to visually cross-check the
    (windowed) time-averages get_zonal_shear_profiles reports.
    """
    time = run.get_time_array()
    time_sampled = time[::time_idx_skip]
    n = len(time_sampled)

    gammaE_t = np.zeros(n)
    gammaE_RH_t = np.zeros(n)
    upar_t = np.zeros(n)
    uparcos_t = np.zeros(n)
    dxT_t = np.zeros(n)
    phi2_vEpos_t = np.zeros(n)
    phi2_vEneg_t = np.zeros(n)

    i_t = 0
    for time_idx in np.arange(len(time))[::time_idx_skip]:
        dxphi_x_y, x, _, _ = run.get_quantity_x_y(quantity="phi", only_zonal=True, kx_order=1, time_idx=time_idx)
        dx2phi_x_y, x, _, _ = run.get_quantity_x_y(quantity="phi", only_zonal=True, kx_order=2, time_idx=time_idx)
        dx2phiRH_x_y, x, _, _ = run.get_quantity_x_y(quantity="RH_phi", only_zonal=True, kx_order=2, time_idx=time_idx)
        upar_x_y, x, _, _ = run.get_quantity_x_y(quantity="upar", only_zonal=True, kx_order=0, time_idx=time_idx)
        uparcos_x_y, x, _, _ = run.get_quantity_x_y(quantity="upar", only_zonal=True, kx_order=0, time_idx=time_idx, mult_zed="cos")
        dxT_x_y, x, _, _ = run.get_quantity_x_y(quantity="temperature", only_zonal=True, kx_order=1, time_idx=time_idx)
        phi2_x_y, x, y, _ = run.get_quantity_x_y(quantity="phi", ky_order=0, kx_order=0, abs_squared=True, time_idx=time_idx, remove_zonal=True)
        vEzonal_x = -dxphi_x_y[:, 0]

        gammaE_t[i_t] = (np.mean(dx2phi_x_y[:, 0] ** exp_avg)) ** (1 / exp_avg)
        gammaE_RH_t[i_t] = (np.mean(dx2phiRH_x_y[:, 0] ** exp_avg)) ** (1 / exp_avg)
        upar_t[i_t] = (np.mean(upar_x_y[:, 0] ** exp_avg)) ** (1 / exp_avg)
        uparcos_t[i_t] = (np.mean(uparcos_x_y[:, 0] ** exp_avg)) ** (1 / exp_avg)
        dxT_t[i_t] = (np.mean(dxT_x_y[:, 0] ** exp_avg)) ** (1 / exp_avg)
        phi2_vEpos_t[i_t] = np.sum(phi2_x_y[vEzonal_x > 0, :]) * (x[1] - x[0]) * (y[1] - y[0])
        phi2_vEneg_t[i_t] = np.sum(phi2_x_y[vEzonal_x < 0, :]) * (x[1] - x[0]) * (y[1] - y[0])
        i_t += 1

    return {
        "time": time_sampled,
        "gammaE_t": gammaE_t,
        "gammaE_RH_t": gammaE_RH_t,
        "upar_t": upar_t,
        "uparcos_t": uparcos_t,
        "dxT_t": dxT_t,
        "phi2_vEpos_t": phi2_vEpos_t,
        "phi2_vEneg_t": phi2_vEneg_t,
    }


@cached(version=2)
def get_zonal_shear_profiles(run, time_val_avg=None, time_avg=5, kx_max=0.3) -> dict:
    """Time-averaged zonal ExB shear/flow/temperature-gradient profiles vs
    x, plus their RMS-over-x reductions.

    NOTE: gammaE_std (and gammaE_RH_std etc.) use an unusual definition --
    sqrt(mean((|gammaE_x| - gammaE_avg)**2)), i.e. the RMS *deviation from
    the RMS*, not a plain standard deviation of gammaE_x itself. Preserved
    exactly as in the original.

    Version bumped to 2: quantities/realspace.py::get_quantity_zed_x_y
    (reached via run.get_quantity_x_y) had a boundary-clamp bug where the
    time_val_avg=None (trailing, ending at the run's last sample) case
    silently averaged over only time_avg/2 of the requested width instead
    of the full time_avg -- fixed there; this function's own numeric
    output for that default case changes as a result (a real, deliberate
    fix, not a regression), so old cache entries must not be reused.
    """
    dxphi_x_y, x, _, _ = run.get_quantity_x_y(quantity="phi", time_val=time_val_avg, only_zonal=True, kx_order=1, time_avg=time_avg)
    dxphi_RH_x_y, x, _, _ = run.get_quantity_x_y(quantity="RH_phi", time_val=time_val_avg, only_zonal=True, kx_order=1, time_avg=time_avg)
    dx2phi_x_y, x, _, _ = run.get_quantity_x_y(quantity="phi", time_val=time_val_avg, only_zonal=True, kx_order=2, time_avg=time_avg)
    dx2phi_RH_x_y, x, _, _ = run.get_quantity_x_y(quantity="RH_phi", time_val=time_val_avg, only_zonal=True, kx_order=2, time_avg=time_avg)
    dx2phi_LW_x_y, x, _, _ = run.get_quantity_x_y(quantity="phi", time_val=time_val_avg, only_zonal=True, kx_order=2, time_avg=time_avg, kx_lowpass_cutoff=kx_max)
    dx2phi_RH_LW_x_y, x, _, _ = run.get_quantity_x_y(quantity="RH_phi", time_val=time_val_avg, only_zonal=True, kx_order=2, time_avg=time_avg, kx_lowpass_cutoff=kx_max)
    dxT_x_y, x, _, _ = run.get_quantity_x_y(quantity="temperature", time_val=time_val_avg, only_zonal=True, kx_order=1, time_avg=time_avg)
    upar_x_y, x, _, _ = run.get_quantity_x_y(quantity="upar", time_val=time_val_avg, only_zonal=True, kx_order=0, time_avg=time_avg)
    dxupar_x_y, x, _, _ = run.get_quantity_x_y(quantity="upar", time_val=time_val_avg, only_zonal=True, kx_order=1, time_avg=time_avg)
    uparcos_x_y, x, _, _ = run.get_quantity_x_y(quantity="upar", time_val=time_val_avg, only_zonal=True, kx_order=0, time_avg=time_avg, mult_zed="cos")
    dyphi2_x_y, x, _, _ = run.get_quantity_x_y(quantity="phi", time_val=time_val_avg, ky_order=1, kx_order=0, time_avg=time_avg, abs_squared=True)

    vE_x = -dxphi_x_y[:, 0]
    vE_RH_x = -dxphi_RH_x_y[:, 0]
    gammaE_x = -dx2phi_x_y[:, 0]
    gammaE_RH_x = -dx2phi_RH_x_y[:, 0]
    gammaE_LW_x = -dx2phi_LW_x_y[:, 0]
    gammaE_RH_LW_x = -dx2phi_RH_LW_x_y[:, 0]
    dxT_x = dxT_x_y[:, 0]
    upar_x = upar_x_y[:, 0]
    dxupar_x = dxupar_x_y[:, 0]
    uparcos_x = uparcos_x_y[:, 0]
    dyphi2_x = dyphi2_x_y[:, 0]

    gammaE_avg = np.sqrt(np.mean(gammaE_x ** 2))
    gammaE_RH_avg = np.sqrt(np.mean(gammaE_RH_x ** 2))
    gammaE_std = np.sqrt(np.mean((np.abs(gammaE_x) - gammaE_avg) ** 2))
    gammaE_RH_std = np.sqrt(np.mean((np.abs(gammaE_RH_x) - gammaE_RH_avg) ** 2))
    gammaE_LW_avg = np.sqrt(np.mean(gammaE_LW_x ** 2))
    gammaE_RH_LW_avg = np.sqrt(np.mean(gammaE_RH_LW_x ** 2))
    gammaE_LW_std = np.sqrt(np.mean((np.abs(gammaE_LW_x) - gammaE_LW_avg) ** 2))
    gammaE_RH_LW_std = np.sqrt(np.mean((np.abs(gammaE_RH_LW_x) - gammaE_RH_LW_avg) ** 2))
    vE_avg = np.sqrt(np.mean(vE_x ** 2))
    vE_RH_avg = np.sqrt(np.mean(vE_RH_x ** 2))
    upar_avg = np.sqrt(np.mean(upar_x ** 2))
    uparcos_avg = np.sqrt(np.mean(uparcos_x ** 2))
    dxT_avg = np.sqrt(np.mean(dxT_x ** 2))

    return {
        "x": x,
        "vE_x": vE_x,
        "vE_RH_x": vE_RH_x,
        "gammaE_x": gammaE_x,
        "gammaE_RH_x": gammaE_RH_x,
        "gammaE_LW_x": gammaE_LW_x,
        "gammaE_RH_LW_x": gammaE_RH_LW_x,
        "dxT_x": dxT_x,
        "upar_x": upar_x,
        "dxupar_x": dxupar_x,
        "uparcos_x": uparcos_x,
        "dyphi2_x": dyphi2_x,
        "gammaE_avg": gammaE_avg,
        "gammaE_RH_avg": gammaE_RH_avg,
        "gammaE_std": gammaE_std,
        "gammaE_RH_std": gammaE_RH_std,
        "gammaE_LW_avg": gammaE_LW_avg,
        "gammaE_RH_LW_avg": gammaE_RH_LW_avg,
        "gammaE_LW_std": gammaE_LW_std,
        "gammaE_RH_LW_std": gammaE_RH_LW_std,
        "vE_avg": vE_avg,
        "vE_RH_avg": vE_RH_avg,
        "upar_avg": upar_avg,
        "uparcos_avg": uparcos_avg,
        "dxT_avg": dxT_avg,
    }


def _get_time_idxs_for_avg(time, time_val_avg, time_avg):
    if time_val_avg is None:
        return np.argwhere(time > time[-1] - time_avg).flatten()
    return np.argwhere((time > time_val_avg - time_avg / 2) & (time < time_val_avg + time_avg / 2)).flatten()


@cached(version=2)
def get_RH_power_transfer_profiles(run, time_val_avg=None, time_avg=5) -> dict:
    """Time-averaged parallel/perpendicular momentum-transport power
    (dE_Pi_parallel_x, dE_Pi_perp_x) and Rosenbluth-Hinton power-transfer
    (P_RH_even/odd_x, split further into passing/trapped contributions)
    profiles vs x, plus their box-integrated (and vE-sign-conditioned)
    scalar reductions.

    Version bumped to 2 along with the get_zonal_shear_profiles call this
    composes with (see its own version-2 note) -- its time_val_avg=None
    default case now gets the full requested time_avg width instead of
    half.

    Composes with get_zonal_shear_profiles (called internally, with the
    same time_val_avg/time_avg) for the time-averaged vE_x used both to
    weight dE_Pi_perp_x and to split every scalar reduction into a
    vE>=0/vE<0 half.

    NOTE: preserves the original's compatibility fallback exactly -- older
    stella output has one combined 'RH_fluxes_phi_even'/'RH_fluxes_phi_odd'
    pair; newer output splits each into '..._passing'/'..._trapped'. This
    is a real stella-version difference (mirroring the same fallback in
    stella_diagnostics.physics.rosenbluth_hinton.get_RH_fluxes), not a bug,
    so both branches are kept rather than collapsed to one.
    """
    shear = get_zonal_shear_profiles(run, time_val_avg=time_val_avg, time_avg=time_avg)
    x = shear["x"]
    vE_x = shear["vE_x"]

    time = run.get_time_array()
    time_idxs = _get_time_idxs_for_avg(time, time_val_avg, time_avg)
    dt_vals = np.gradient(time[time_idxs])

    dl_over_B_avg = run.dl_over_B_avg()

    dE_Pi_parallel_x = np.zeros_like(x)
    dE_Pi_perp_x = np.zeros_like(x)
    P_RH_even_x = np.zeros_like(x)
    P_RH_odd_x = np.zeros_like(x)
    P_RH_even_passing_x = np.zeros_like(x)
    P_RH_odd_passing_x = np.zeros_like(x)
    P_RH_even_trapped_x = np.zeros_like(x)
    P_RH_odd_trapped_x = np.zeros_like(x)

    for i, time_idx in enumerate(time_idxs):
        dx_Pi_parallel_zed_x_y, _, x, _, _ = run.get_quantity_zed_x_y(quantity="par_mom_transport", only_zonal=True, kx_order=1, time_idx=time_idx)
        dx_Pi_perp_zed_x_y, _, x, _, _ = run.get_quantity_zed_x_y(quantity="Reynolds", only_zonal=True, kx_order=1, time_idx=time_idx)
        uparZ_zed_x_y, _, x, _, _ = run.get_quantity_zed_x_y(quantity="upar", only_zonal=True, kx_order=0, time_idx=time_idx)

        dE_Pi_parallel_x += -np.sum(dl_over_B_avg[:, None] * dx_Pi_parallel_zed_x_y[:, :, 0] * uparZ_zed_x_y[:, :, 0], axis=0) * dt_vals[i] / np.sum(dt_vals)
        dE_Pi_perp_x += -np.sum(dl_over_B_avg[:, None] * dx_Pi_perp_zed_x_y[:, :, 0], axis=0) * (-vE_x) * dt_vals[i] / np.sum(dt_vals)

        try:
            RH_flux_phi_even_x_y, x, _, _ = run.get_quantity_x_y(quantity="RH_fluxes_phi_even", only_zonal=True, kx_order=0, time_idx=time_idx)
            RH_flux_phi_odd_x_y, x, _, _ = run.get_quantity_x_y(quantity="RH_fluxes_phi_odd", only_zonal=True, kx_order=0, time_idx=time_idx)
            nan_array = np.zeros_like(RH_flux_phi_even_x_y)
            nan_array[:] = np.nan
            RH_flux_phi_even_passing_x_y = nan_array
            RH_flux_phi_odd_passing_x_y = nan_array
            RH_flux_phi_even_trapped_x_y = nan_array
            RH_flux_phi_odd_trapped_x_y = nan_array
        except Exception:
            RH_flux_phi_even_passing_x_y, x, _, _ = run.get_quantity_x_y(quantity="RH_fluxes_phi_even_passing", only_zonal=True, kx_order=0, time_idx=time_idx)
            RH_flux_phi_odd_passing_x_y, x, _, _ = run.get_quantity_x_y(quantity="RH_fluxes_phi_odd_passing", only_zonal=True, kx_order=0, time_idx=time_idx)
            RH_flux_phi_even_trapped_x_y, x, _, _ = run.get_quantity_x_y(quantity="RH_fluxes_phi_even_trapped", only_zonal=True, kx_order=0, time_idx=time_idx)
            RH_flux_phi_odd_trapped_x_y, x, _, _ = run.get_quantity_x_y(quantity="RH_fluxes_phi_odd_trapped", only_zonal=True, kx_order=0, time_idx=time_idx)
            RH_flux_phi_even_x_y = RH_flux_phi_even_passing_x_y + RH_flux_phi_even_trapped_x_y
            RH_flux_phi_odd_x_y = RH_flux_phi_odd_passing_x_y + RH_flux_phi_odd_trapped_x_y

        RH_flux_phi_even_x = RH_flux_phi_even_x_y[:, 0]
        RH_flux_phi_odd_x = RH_flux_phi_odd_x_y[:, 0]
        RH_flux_phi_even_passing_x = RH_flux_phi_even_passing_x_y[:, 0]
        RH_flux_phi_odd_passing_x = RH_flux_phi_odd_passing_x_y[:, 0]
        RH_flux_phi_even_trapped_x = RH_flux_phi_even_trapped_x_y[:, 0]
        RH_flux_phi_odd_trapped_x = RH_flux_phi_odd_trapped_x_y[:, 0]

        dxphi_RH_x_y_inst, x, _, _ = run.get_quantity_x_y(quantity="RH_phi", only_zonal=True, kx_order=1, time_idx=time_idx)
        vE_RH_x_inst = -dxphi_RH_x_y_inst[:, 0]

        P_RH_even_x += -RH_flux_phi_even_x * vE_RH_x_inst * dt_vals[i] / np.sum(dt_vals)
        P_RH_odd_x += -RH_flux_phi_odd_x * vE_RH_x_inst * dt_vals[i] / np.sum(dt_vals)
        P_RH_even_passing_x += -RH_flux_phi_even_passing_x * vE_RH_x_inst * dt_vals[i] / np.sum(dt_vals)
        P_RH_odd_passing_x += -RH_flux_phi_odd_passing_x * vE_RH_x_inst * dt_vals[i] / np.sum(dt_vals)
        P_RH_even_trapped_x += -RH_flux_phi_even_trapped_x * vE_RH_x_inst * dt_vals[i] / np.sum(dt_vals)
        P_RH_odd_trapped_x += -RH_flux_phi_odd_trapped_x * vE_RH_x_inst * dt_vals[i] / np.sum(dt_vals)

    dyphi2_x = shear["dyphi2_x"]

    dx = x[1] - x[0]
    vEpos = vE_x >= 0
    vEneg = vE_x < 0

    def _box_integral(f_x):
        return np.sum(dx * f_x)

    result = {
        "x": x,
        "dE_Pi_parallel_x": dE_Pi_parallel_x,
        "dE_Pi_perp_x": dE_Pi_perp_x,
        "P_RH_even_x": P_RH_even_x,
        "P_RH_odd_x": P_RH_odd_x,
        "P_RH_even_passing_x": P_RH_even_passing_x,
        "P_RH_odd_passing_x": P_RH_odd_passing_x,
        "P_RH_even_trapped_x": P_RH_even_trapped_x,
        "P_RH_odd_trapped_x": P_RH_odd_trapped_x,
    }
    for name, f_x in (
        ("P_RH_even_avg_alt", P_RH_even_x),
        ("P_RH_odd_avg_alt", P_RH_odd_x),
        ("P_RH_even_passing_avg_alt", P_RH_even_passing_x),
        ("P_RH_odd_passing_avg_alt", P_RH_odd_passing_x),
        ("P_RH_even_trapped_avg_alt", P_RH_even_trapped_x),
        ("P_RH_odd_trapped_avg_alt", P_RH_odd_trapped_x),
        ("dyphi2_avg", dyphi2_x),
        ("dE_Pi_parallel", dE_Pi_parallel_x),
        ("dE_Pi_perp", dE_Pi_perp_x),
    ):
        result[name] = _box_integral(f_x)
        result[name + "_vEpos"] = _box_integral(f_x[vEpos])
        result[name + "_vEneg"] = _box_integral(f_x[vEneg])

    return result


@cached(version=3)
def get_RH_power_time_averages(run, time_max=1e10, time_idx_skip=10, time_val_avg=None, time_avg=5, kx_max=0.3) -> dict:
    """Time-averaged Rosenbluth-Hinton power P_RH (even/odd/collisional),
    summed over all kx and over |kx|<=kx_max ("LW", long-wavelength) only
    -- a second, independent estimate of similar quantities to
    get_RH_power_transfer_profiles's P_RH_even_x/P_RH_odd_x (via
    run.get_P_RH, not the raw RH_fluxes_phi_* variables). Kept as a
    separate function/cache entry rather than reconciled with the other
    estimate -- the two are a deliberate cross-check, not redundant.

    Also returns the underlying (time, P_RH_*_t[_LW]) time series (not
    just their windowed avg/std), since
    plot_zonal_shear_diagnostic_page's P_RH(t) panel needs them too.

    Version bumped to 3: now built on top of
    physics.rosenbluth_hinton.get_P_RH_breakdown (the same breakdown
    machinery plot_RH_phi_E_P_t_all_kx.py uses) instead of two raw
    run.get_P_RH calls, so phi/apar/bpar are exposed individually
    (P_RH_phi_t/P_RH_apar_t/P_RH_bpar_t[_LW]) instead of being silently
    folded into "P_RH_even_t"/"P_RH_odd_t" with no way to tell how much
    of it was apar/bpar. "P_RH_even_t"/"P_RH_odd_t" keep their original
    meaning here (phi+apar+bpar, excluding coll) -- NOT the same as
    get_P_RH_breakdown's own P_RH_even_t_kx/P_RH_odd_t_kx, which also
    folds coll in.
    """
    (_, _, P_RH_phi_even_t_kx, P_RH_phi_odd_t_kx,
     P_RH_apar_even_t_kx, P_RH_apar_odd_t_kx,
     P_RH_bpar_even_t_kx, P_RH_bpar_odd_t_kx,
     P_RH_coll_even_t_kx, P_RH_coll_odd_t_kx,
     _, time, kx) = get_P_RH_breakdown(run, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max)

    P_RH_even_t_kx = P_RH_phi_even_t_kx + P_RH_apar_even_t_kx + P_RH_bpar_even_t_kx
    P_RH_odd_t_kx = P_RH_phi_odd_t_kx + P_RH_apar_odd_t_kx + P_RH_bpar_odd_t_kx
    P_RH_coll_t_kx = P_RH_coll_even_t_kx + P_RH_coll_odd_t_kx
    P_RH_phi_t_kx = P_RH_phi_even_t_kx + P_RH_phi_odd_t_kx
    P_RH_apar_t_kx = P_RH_apar_even_t_kx + P_RH_apar_odd_t_kx
    P_RH_bpar_t_kx = P_RH_bpar_even_t_kx + P_RH_bpar_odd_t_kx

    LW = np.abs(kx) <= kx_max

    def _sum_kx(t_kx, lw_only=False):
        return np.sum(t_kx[:, LW] if lw_only else t_kx, axis=1)

    P_RH_even_t, P_RH_even_t_LW = _sum_kx(P_RH_even_t_kx), _sum_kx(P_RH_even_t_kx, True)
    P_RH_odd_t, P_RH_odd_t_LW = _sum_kx(P_RH_odd_t_kx), _sum_kx(P_RH_odd_t_kx, True)
    P_RH_coll_t, P_RH_coll_t_LW = _sum_kx(P_RH_coll_t_kx), _sum_kx(P_RH_coll_t_kx, True)
    P_RH_phi_t, P_RH_phi_t_LW = _sum_kx(P_RH_phi_t_kx), _sum_kx(P_RH_phi_t_kx, True)
    P_RH_apar_t, P_RH_apar_t_LW = _sum_kx(P_RH_apar_t_kx), _sum_kx(P_RH_apar_t_kx, True)
    P_RH_bpar_t, P_RH_bpar_t_LW = _sum_kx(P_RH_bpar_t_kx), _sum_kx(P_RH_bpar_t_kx, True)

    dt = np.gradient(time)
    idxs_avg = _get_time_idxs_for_avg(time, time_val_avg, time_avg)

    def _avg(t_arr):
        return np.sum((dt * t_arr)[idxs_avg]) / np.sum(dt[idxs_avg])

    return {
        "time": time,
        "kx": kx,
        "P_RH_even_t": P_RH_even_t,
        "P_RH_odd_t": P_RH_odd_t,
        "P_RH_coll_t": P_RH_coll_t,
        "P_RH_phi_t": P_RH_phi_t,
        "P_RH_apar_t": P_RH_apar_t,
        "P_RH_bpar_t": P_RH_bpar_t,
        "P_RH_even_t_LW": P_RH_even_t_LW,
        "P_RH_odd_t_LW": P_RH_odd_t_LW,
        "P_RH_coll_t_LW": P_RH_coll_t_LW,
        "P_RH_phi_t_LW": P_RH_phi_t_LW,
        "P_RH_apar_t_LW": P_RH_apar_t_LW,
        "P_RH_bpar_t_LW": P_RH_bpar_t_LW,
        "P_RH_even_avg": _avg(P_RH_even_t),
        "P_RH_odd_avg": _avg(P_RH_odd_t),
        "P_RH_coll_avg": _avg(P_RH_coll_t),
        "P_RH_phi_avg": _avg(P_RH_phi_t),
        "P_RH_apar_avg": _avg(P_RH_apar_t),
        "P_RH_bpar_avg": _avg(P_RH_bpar_t),
        "P_RH_even_avg_LW": _avg(P_RH_even_t_LW),
        "P_RH_odd_avg_LW": _avg(P_RH_odd_t_LW),
        "P_RH_coll_avg_LW": _avg(P_RH_coll_t_LW),
        "P_RH_phi_avg_LW": _avg(P_RH_phi_t_LW),
        "P_RH_apar_avg_LW": _avg(P_RH_apar_t_LW),
        "P_RH_bpar_avg_LW": _avg(P_RH_bpar_t_LW),
        "P_RH_even_std": np.std(P_RH_even_t[idxs_avg]),
        "P_RH_odd_std": np.std(P_RH_odd_t[idxs_avg]),
        "P_RH_coll_std": np.std(P_RH_coll_t[idxs_avg]),
        "P_RH_even_std_LW": np.std(P_RH_even_t_LW[idxs_avg]),
        "P_RH_odd_std_LW": np.std(P_RH_odd_t_LW[idxs_avg]),
        "P_RH_coll_std_LW": np.std(P_RH_coll_t_LW[idxs_avg]),
    }


def _panel_Q_t(run, ax, cfg, fig=None):
    growth = get_growth_rate_from_flux(run, time_max=cfg["time_max"], qflx_rel_idx_min=cfg["qflx_rel_idx_min"], qflx_rel_idx_max=cfg["qflx_rel_idx_max"], time_val_avg=cfg["time_val_avg"], time_avg=cfg["time_avg"])
    time_flux, qflx = growth["time"], growth["qflx"]
    idx_min, idx_max, gamma_max = growth["idx_min"], growth["idx_max"], growth["gamma_lin_max"]
    qflx_avg, qflx_std = growth["qflx_avg"], growth["qflx_std"]

    ax.axvline(time_flux[idx_min], c="crimson", alpha=0.5)
    ax.axvline(time_flux[idx_max], c="crimson", alpha=0.5)
    ax.semilogy(time_flux, qflx, c="k", label=r"$Q/Q_\mathrm{gB}$")
    ax.semilogy(time_flux[:idx_max], qflx[idx_min] * np.exp(2 * gamma_max * (time_flux[:idx_max] - time_flux[idx_min])), ls="--", c="crimson", label=r"$\sim$ e$^{\gamma_\mathrm{lin}^\mathrm{max}t}$")
    ax.axhline(qflx_avg, c="k", ls="--")
    ax.fill_between(time_flux, (qflx_avg - qflx_std) * np.ones_like(time_flux), (qflx_avg + qflx_std) * np.ones_like(time_flux), color="k", alpha=0.25)

    ts = get_zonal_shear_time_series(run, time_idx_skip=cfg["time_idx_skip"], exp_avg=cfg["exp_avg"])
    time_ts = ts["time"]

    ax.semilogy(time_ts, ts["gammaE_t"], c="orange", label=r"$\langle \gamma_E^2(x) \rangle^{1/2}$")
    ax.semilogy(time_ts, ts["gammaE_RH_t"], c="orange", label=r"$\langle \gamma_{E,\mathrm{RH}}^2(x) \rangle^{1/2}$", alpha=0.5, lw=2)
    ax.semilogy(time_ts, ts["upar_t"], c="forestgreen", label=r"$\langle u_\parallel^2(x) \rangle^{1/2}$")
    ax.semilogy(time_ts, ts["uparcos_t"], c="forestgreen", label=r"$\langle (u_\parallel \cos\theta)^2(x) \rangle^{1/2}$", ls="--")

    norm_phi2 = 2 * np.max(qflx) / np.nanmax(np.array([ts["phi2_vEpos_t"], ts["phi2_vEneg_t"]]))

    ax.semilogy(time_ts, ts["phi2_vEpos_t"] * norm_phi2, c="mediumblue", label=r"$\langle \tilde\varphi^2 (v_E>0) \rangle$")
    ax.semilogy(time_ts, ts["phi2_vEneg_t"] * norm_phi2, c="crimson", label=r"$\langle \tilde\varphi^2 (v_E<0) \rangle$")
    ax.set_xlabel(r"$t %s/a$" % get_vt_label(run.ncdata))
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    ax.set_ylim(ymin=1e-3 * qflx[idx_max])

    time_val_avg, time_avg = cfg["time_val_avg"], cfg["time_avg"]
    if time_val_avg is None:
        ax.fill_betweenx(ax.get_ylim(), time_flux[-1] - time_avg, time_flux[-1], color="0.5", alpha=0.15)
    else:
        ax.fill_betweenx(ax.get_ylim(), time_val_avg - time_avg / 2, time_val_avg + time_avg / 2, color="0.5", alpha=0.15)


def _panel_gammaE_x(run, ax, cfg, fig=None):
    shear = get_zonal_shear_profiles(run, time_val_avg=cfg["time_val_avg"], time_avg=cfg["time_avg"], kx_max=cfg["kx_max"])
    x = shear["x"]
    ax.plot(x, shear["gammaE_x"], c="k")
    ax.plot(x, shear["gammaE_RH_x"], c="k", alpha=0.5)
    ax.plot(x, shear["dxT_x"], c="forestgreen", label=r"$\partial_x T$")
    ax.axhline(shear["gammaE_avg"], c="mediumblue", label=r"$\langle \gamma_E^2(x) \rangle^{1/2}$")
    ax.axhline(-shear["gammaE_avg"], c="mediumblue")
    ax.axhline(shear["gammaE_RH_avg"], c="mediumblue", alpha=0.5, label=r"$\langle \gamma_{E, \mathrm{RH}}^2(x) \rangle^{1/2}$")
    ax.axhline(-shear["gammaE_RH_avg"], c="mediumblue", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    ax.grid(True)
    ax.set_xlabel(r"$x/%s$" % get_rho_label(run.ncdata))
    ax.set_xlim([x[0], x[-1]])


def _panel_vE_x_profile(run, ax, cfg, fig=None):
    shear = get_zonal_shear_profiles(run, time_val_avg=cfg["time_val_avg"], time_avg=cfg["time_avg"], kx_max=cfg["kx_max"])
    x = shear["x"]
    vE_x = shear["vE_x"]
    eps = estimate_eps_from_bmag(run)
    qinp = float(run.ncdata.variables["q"].getValue())  # same read as get_growth_rate_from_flux's own qinp

    transfer = get_RH_power_transfer_profiles(run, time_val_avg=cfg["time_val_avg"], time_avg=cfg["time_avg"])
    P_RH_even_x, P_RH_odd_x = transfer["P_RH_even_x"], transfer["P_RH_odd_x"]

    ax.plot(x, vE_x, c="k", label=r"$v_E$")
    ax.plot(x, shear["vE_RH_x"], c="k", alpha=0.5)
    ax.plot(x, 2 * shear["upar_x"] / (1.6 * np.sqrt(eps) * qinp), c="orange", label=r"$\langle u_\parallel \rangle_\theta/(1.6 \epsilon^{1/2} q)$")
    ax.plot(x, 2 * shear["uparcos_x"] * 2 / (2 * qinp), c="forestgreen", label=r"$2 \langle u_\parallel \cos\theta \rangle_\theta / (2q)$")

    norm = np.abs(vE_x).max() / np.abs(shear["dyphi2_x"]).max()
    ax.plot(x, shear["dyphi2_x"] * norm, c="c", label=r"$v_{Ex}^2$")

    norm = np.abs(vE_x).max() / max(np.abs(P_RH_even_x).max(), np.abs(P_RH_odd_x).max())
    ax.plot(x, P_RH_even_x * norm, c="crimson", lw=2, label=r"$\mathcal{P}_\varphi^+$")
    ax.plot(x, P_RH_odd_x * norm, c="mediumblue", lw=2, label=r"$\mathcal{P}_\varphi^-$")
    ax.axhline(transfer["P_RH_even_avg_alt"] * norm / 2, c="crimson", lw=2, alpha=0.5)
    ax.axhline(transfer["P_RH_odd_avg_alt"] * norm / 2, c="mediumblue", lw=2, alpha=0.5)

    # NOTE: each region below (a sign-of-vE_x x a sign-of-x quadrant) can be
    # empty for a given run's zonal profile (e.g. an asymmetric x-domain
    # where vE_x never changes sign on one side) -- the original crashed on
    # min()/max() of an empty array in that case; skipping an empty
    # quadrant's hline is a plotting-robustness fix, not a change to any
    # computed value.
    for vE_sign, P_RH_even_avg_alt_v, P_RH_odd_avg_alt_v in (
        (vE_x > 0, transfer["P_RH_even_avg_alt_vEpos"], transfer["P_RH_odd_avg_alt_vEpos"]),
        (vE_x < 0, transfer["P_RH_even_avg_alt_vEneg"], transfer["P_RH_odd_avg_alt_vEneg"]),
    ):
        for x_sign in (x < 0, x > 0):
            x_region = x[vE_sign & x_sign]
            if len(x_region) == 0:
                continue
            ax.hlines(P_RH_even_avg_alt_v * norm, xmin=x_region.min(), xmax=x_region.max(), colors="crimson", lw=1, alpha=0.5)
            ax.hlines(P_RH_odd_avg_alt_v * norm, xmin=x_region.min(), xmax=x_region.max(), colors="mediumblue", lw=1, alpha=0.5)

    norm = np.abs(vE_x).max() / np.abs(transfer["dE_Pi_parallel_x"]).max()
    ax.plot(x, transfer["dE_Pi_parallel_x"] * norm, c="brown", label=r"$- u_\parallel^Z \partial_x\Pi_\parallel $")

    norm = np.abs(vE_x).max() / np.abs(transfer["dE_Pi_perp_x"]).max()
    ax.plot(x, transfer["dE_Pi_perp_x"] * norm, c="pink", label=r"$- v_E^Z \partial_x\Pi_\perp $")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    ax.grid(True)
    ax.set_xlabel(r"$x/%s$" % get_rho_label(run.ncdata))
    ax.set_xlim([x[0], x[-1]])


def _panel_P_RH_t(run, ax, cfg, fig=None):
    """Rosenbluth-Hinton power vs time, broken down both by parity
    (even/odd, the original two lines) and by field (phi/apar/bpar, new
    -- apar/bpar are all-zero lines for a run without electromagnetic
    effects rather than absent, so their absence from a run is visible
    rather than silently unrepresented)."""
    tavgs = get_RH_power_time_averages(run, time_max=cfg["time_max"], time_idx_skip=cfg["time_idx_skip"], time_val_avg=cfg["time_val_avg"], time_avg=cfg["time_avg"], kx_max=cfg["kx_max"])
    time_prh = tavgs["time"]

    ax.plot(time_prh, tavgs["P_RH_even_t"], c="crimson", label=r"$P_{\mathrm{RH}}^+$")
    ax.plot(time_prh, tavgs["P_RH_odd_t"], c="mediumblue", label=r"$P_{\mathrm{RH}}^-$")
    ax.plot(time_prh, tavgs["P_RH_coll_t"], c="orange", label=r"$P_{\mathrm{RH}}^C$")
    ax.plot(time_prh, tavgs["P_RH_phi_t"], c="purple", label=r"$P_{\mathrm{RH}}^\varphi$")
    ax.plot(time_prh, tavgs["P_RH_apar_t"], c="teal", label=r"$P_{\mathrm{RH}}^{A_\parallel}$")
    ax.plot(time_prh, tavgs["P_RH_bpar_t"], c="goldenrod", label=r"$P_{\mathrm{RH}}^{B_\parallel}$")
    ax.plot(time_prh, tavgs["P_RH_coll_t"] + tavgs["P_RH_even_t"] + tavgs["P_RH_odd_t"], c="k", label=r"$P_{\mathrm{RH}}$", alpha=0.5)

    ax.axhline(tavgs["P_RH_even_avg"], c="crimson", ls="--")
    ax.axhline(tavgs["P_RH_odd_avg"], c="mediumblue", ls="--")
    ax.axhline(tavgs["P_RH_coll_avg"], c="orange", ls="--")
    ax.axhline(tavgs["P_RH_phi_avg"], c="purple", ls="--")
    ax.axhline(tavgs["P_RH_apar_avg"], c="teal", ls="--")
    ax.axhline(tavgs["P_RH_bpar_avg"], c="goldenrod", ls="--")

    ax.set_xlabel(r"$t %s/a$" % get_vt_label(run.ncdata))
    ax.set_ylabel(r"$P_\mathrm{RH}$")
    if np.abs(tavgs["P_RH_even_avg"]) > 0:
        ax.set_yscale("symlog", linthresh=1e-1 * np.abs(tavgs["P_RH_even_avg"]))
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    ax.set_ylim(ax.get_ylim())
    time_val_avg, time_avg = cfg["time_val_avg"], cfg["time_avg"]
    if time_val_avg is None:
        ax.fill_betweenx(ax.get_ylim(), time_prh[-1] - time_avg, time_prh[-1], color="0.5", alpha=0.15)
    else:
        ax.fill_betweenx(ax.get_ylim(), time_val_avg - time_avg / 2, time_val_avg + time_avg / 2, color="0.5", alpha=0.15)


def _overlay_zonal_profiles(run, ax, cfg, theta_val):
    dxphizonal, x2, y, _ = run.get_quantity_x_y(quantity="phi", time_idx=-1, only_zonal=True, kx_order=1, nx=cfg["nx_padded"], zed_val=theta_val)
    vEzonal = -dxphizonal
    uparzonal, x2, y, _ = run.get_quantity_x_y(quantity="upar", time_idx=-1, only_zonal=True, kx_order=0, nx=cfg["nx_padded"], zed_val=theta_val)
    tempzonal, x2, y, _ = run.get_quantity_x_y(quantity="temperature", time_idx=-1, only_zonal=True, kx_order=0, nx=cfg["nx_padded"], zed_val=theta_val)
    vE_norm = 0.5 * y[-1] / np.abs(vEzonal).max()
    upar_norm = 0.5 * y[-1] / np.abs(uparzonal).max()
    temp_norm = 0.5 * y[-1] / np.abs(tempzonal).max()

    ax.plot(x2, vEzonal[:, 0] * vE_norm, c="forestgreen", label=r"$v_E$")
    ax.plot(x2, uparzonal[:, 0] * upar_norm, c="c", label=r"$u_\parallel$")
    ax.plot(x2, tempzonal[:, 0] * temp_norm, c="crimson", label=r"$T_\mathrm{tot}$")
    ax.set_aspect("equal")
    ax.set_xlim([x2[0], x2[-1]])
    # NOTE: shown on every instance now (was only the vEx_xy panel before)
    # -- with a flexible panel layout the two panels aren't guaranteed to
    # be adjacent any more, so each needs its own legend.
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)


def _panel_vEx_xy(run, ax, cfg, theta_val, fig=None):
    ax.set_title(r"$\tilde v_{Ex} (\theta = %.2f)$" % (theta_val))
    run.plot_quantity_x_y(quantity="phi", time_idx=-1, ky_order=1, nx=cfg["nx_padded"], ny=cfg["ny_padded"], fig=fig, ax=ax, zed_val=theta_val, symm=True, remove_zonal=True)
    _overlay_zonal_profiles(run, ax, cfg, theta_val)


def _panel_upar_xy(run, ax, cfg, theta_val, fig=None):
    ax.set_title(r"$\tilde u_\parallel (\theta = %.2f)$" % (theta_val))
    run.plot_quantity_x_y(quantity="upar", time_idx=-1, nx=cfg["nx_padded"], ny=cfg["ny_padded"], fig=fig, ax=ax, zed_val=theta_val, symm=True, remove_zonal=True)
    _overlay_zonal_profiles(run, ax, cfg, theta_val)


def _panel_dxuparZ_xzed(run, ax, cfg, fig=None):
    ax.set_title(r"$\partial_x u_\parallel^Z$")
    run.plot_quantity_x_zed(quantity="upar", fig=fig, ax=ax, only_zonal=True, kx_order=1, nx=cfg["nx_padded"])


def _panel_Pi_par_xzed(run, ax, cfg, fig=None):
    ax.set_title(r"$\langle \Pi_\parallel \rangle_y$")
    run.plot_quantity_x_zed(quantity="par_mom_transport", fig=fig, ax=ax, only_zonal=True, nx=cfg["nx_padded"])


def _panel_Pi_perp_xzed(run, ax, cfg, fig=None):
    ax.set_title(r"$\langle \Pi_\perp \rangle_y$")
    run.plot_quantity_x_zed(quantity="Reynolds", fig=fig, ax=ax, only_zonal=True, nx=cfg["nx_padded"])


def _panel_Pi_perp_nablax2_xzed(run, ax, cfg, fig=None):
    ax.set_title(r"$\langle \Pi_{\perp, |\nabla x|^2} \rangle_y$")
    run.plot_quantity_x_zed(quantity="Reynolds_nablax2", fig=fig, ax=ax, only_zonal=True, nx=cfg["nx_padded"])


def _panel_Pi_perp_nablaxy_xzed(run, ax, cfg, fig=None):
    ax.set_title(r"$\langle \Pi_{\perp, \nabla x \cdot \nabla y} \rangle_y$")
    run.plot_quantity_x_zed(quantity="Reynolds_nablaxy", fig=fig, ax=ax, only_zonal=True, nx=cfg["nx_padded"])


def _panel_vEx2_xzed(run, ax, cfg, fig=None):
    ax.set_title(r"$\langle \tilde v_{Ex}^2 \rangle_y$" if cfg["avg_norm"] == 2 else r"$\tilde v_{Ex} (y=0)$")
    run.plot_quantity_x_zed(quantity="phi", fig=fig, ax=ax, avg_norm=cfg["avg_norm"], remove_zonal=True, nx=cfg["nx_padded"], ky_order=1)


def _panel_T2_xzed(run, ax, cfg, fig=None):
    ax.set_title(r"$\langle \tilde T^2 \rangle_y$" if cfg["avg_norm"] == 2 else r"$\tilde T (y=0)$")
    run.plot_quantity_x_zed(quantity="temperature", fig=fig, ax=ax, avg_norm=cfg["avg_norm"], remove_zonal=True, nx=cfg["nx_padded"])


def _panel_upar2_xzed(run, ax, cfg, fig=None):
    ax.set_title(r"$\langle \tilde u_\parallel^2 \rangle_y$" if cfg["avg_norm"] == 2 else r"$\tilde u_\parallel (y=0)$")
    run.plot_quantity_x_zed(quantity="upar", fig=fig, ax=ax, avg_norm=cfg["avg_norm"], remove_zonal=True, nx=cfg["nx_padded"])


def _make_P_RH_xzed_panel(quantity, title):
    """Builds a panel function for one P_RH-family (x,zed) contour --
    P_RH_full_even/odd (phi+apar+bpar, the new default panels) and the
    per-field phi/apar/bpar-only variants (extra, non-default panels for
    inspecting the breakdown spatially) all share the same call shape."""
    def _panel(run, ax, cfg, fig=None):
        ax.set_title(title)
        run.plot_quantity_x_zed(quantity=quantity, fig=fig, ax=ax, only_zonal=True, nx=cfg["nx_padded"], vmin="symm")
    return _panel


PANEL_REGISTRY = {
    "Q_t": _panel_Q_t,
    "gammaE_x": _panel_gammaE_x,
    "P_RH_t": _panel_P_RH_t,
    "vE_x_profile": _panel_vE_x_profile,
    "vEx_xy": _panel_vEx_xy,
    "upar_xy": _panel_upar_xy,
    "P_RH_even_xzed": _make_P_RH_xzed_panel("P_RH_full_even", r"$P_\mathrm{RH}^+$"),
    "P_RH_odd_xzed": _make_P_RH_xzed_panel("P_RH_full_odd", r"$P_\mathrm{RH}^-$"),
    "P_RH_phi_even_xzed": _make_P_RH_xzed_panel("P_RH_even", r"$P_\mathrm{RH}^{\varphi,+}$"),
    "P_RH_phi_odd_xzed": _make_P_RH_xzed_panel("P_RH_odd", r"$P_\mathrm{RH}^{\varphi,-}$"),
    "P_RH_apar_even_xzed": _make_P_RH_xzed_panel("P_RH_apar_even", r"$P_\mathrm{RH}^{A_\parallel,+}$"),
    "P_RH_apar_odd_xzed": _make_P_RH_xzed_panel("P_RH_apar_odd", r"$P_\mathrm{RH}^{A_\parallel,-}$"),
    "P_RH_bpar_even_xzed": _make_P_RH_xzed_panel("P_RH_bpar_even", r"$P_\mathrm{RH}^{B_\parallel,+}$"),
    "P_RH_bpar_odd_xzed": _make_P_RH_xzed_panel("P_RH_bpar_odd", r"$P_\mathrm{RH}^{B_\parallel,-}$"),
    "dxuparZ_xzed": _panel_dxuparZ_xzed,
    "Pi_par_xzed": _panel_Pi_par_xzed,
    "Pi_perp_xzed": _panel_Pi_perp_xzed,
    "Pi_perp_nablax2_xzed": _panel_Pi_perp_nablax2_xzed,
    "Pi_perp_nablaxy_xzed": _panel_Pi_perp_nablaxy_xzed,
    "vEx2_xzed": _panel_vEx2_xzed,
    "T2_xzed": _panel_T2_xzed,
    "upar2_xzed": _panel_upar2_xzed,
}

# Panels that take a theta/zed value and get expanded into one panel per
# entry in plot_zonal_shear_diagnostic_page's theta_vals.
THETA_MULTIPLIED_PANELS = {"vEx_xy", "upar_xy"}

# A smaller, curated starting point -- not the full PANEL_REGISTRY -- so a
# no-`panels`-override call stays a readable page. Includes both new
# P_RH (x,zed) panels; the per-field phi/apar/bpar-only P_RH (x,zed)
# variants and the 8 other (x,zed) contour panels from the original
# fixed layout are available (see PANEL_REGISTRY) but not on by default.
DEFAULT_PANELS = [
    "Q_t", "gammaE_x", "P_RH_t", "vE_x_profile",
    "vEx_xy", "upar_xy", "P_RH_even_xzed", "P_RH_odd_xzed",
]


def plot_zonal_shear_diagnostic_page(
    run,
    panels=None,
    theta_vals=(0,),
    ncols=4,
    time_val_avg=None,
    time_avg=5,
    time_max=1e10,
    qflx_rel_idx_min=1e-7,
    qflx_rel_idx_max=1e-3,
    kx_max=0.3,
    time_idx_skip=10,
    exp_avg=2,
    avg_norm=2,
    nx_padded=256,
    ny_padded=256,
    fig=None,
    axs=None,
):
    """Configurable single-run diagnostic page built from a registry of
    independent panel-drawing functions (PANEL_REGISTRY) -- pass
    `panels` (a list of names, default DEFAULT_PANELS) to choose which
    panels appear and in what order; the grid is sized to fit exactly
    that many panels (wrapped at `ncols` columns per row), so there are
    no blank cells regardless of how many panels are requested.

    `theta_vals` (a sequence of zed/theta values, default (0,)) expands
    every panel in THETA_MULTIPLIED_PANELS ("vEx_xy", "upar_xy") into
    one panel per value -- e.g. panels=["vEx_xy"], theta_vals=[0, 1]
    produces 2 panels, one for tilde-v_Ex at theta=0 and one at theta=1.

    Composes get_growth_rate_from_flux/get_zonal_shear_time_series/
    get_zonal_shear_profiles/get_RH_power_transfer_profiles/
    get_RH_power_time_averages/estimate_eps_from_bmag for the data panels
    (each already @cached, so panels sharing the same underlying data --
    e.g. gammaE_x and vE_x_profile both use get_zonal_shear_profiles --
    only compute it once per run, regardless of panel order); calls the
    existing (unchanged) StellaRun.plot_quantity_x_y/plot_quantity_x_zed
    methods directly for the (x,y)/(x,zed) contour panels.
    """
    if panels is None:
        panels = DEFAULT_PANELS

    cfg = dict(
        time_val_avg=time_val_avg, time_avg=time_avg, time_max=time_max,
        qflx_rel_idx_min=qflx_rel_idx_min, qflx_rel_idx_max=qflx_rel_idx_max,
        kx_max=kx_max, time_idx_skip=time_idx_skip, exp_avg=exp_avg,
        avg_norm=avg_norm, nx_padded=nx_padded, ny_padded=ny_padded,
    )

    expanded = []
    for name in panels:
        if name not in PANEL_REGISTRY:
            raise ValueError("Unknown panel %r -- available: %s" % (name, sorted(PANEL_REGISTRY)))
        if name in THETA_MULTIPLIED_PANELS:
            for theta_val in theta_vals:
                expanded.append((name, theta_val))
        else:
            expanded.append((name, None))

    n = len(expanded)
    nrows = -(-n // ncols)  # ceil division
    if axs is None:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9 * ncols, 6 * nrows), squeeze=False)
    axs_flat = list(np.ravel(axs))

    for ax, (name, theta_val) in zip(axs_flat, expanded):
        panel_fn = PANEL_REGISTRY[name]
        if name in THETA_MULTIPLIED_PANELS:
            panel_fn(run, ax, cfg, theta_val, fig=fig)
        else:
            panel_fn(run, ax, cfg, fig=fig)

    for ax in axs_flat[len(expanded):]:
        ax.set_visible(False)

    fig.suptitle(None)
    return fig, axs


def plot_zonal_flow_scan(
    dirnames,
    series_labels,
    aLT_lin_vals=None,
    base_colors=None,
    tprim_exclude=None,
    substract_lin=False,
    xlim=None,
    markersize=10,
    filename="CBC",
    code="stella",
    time_val_avg=None,
    time_avg=5,
    time_max=1e10,
    qflx_rel_idx_min=1e-7,
    qflx_rel_idx_max=1e-3,
    kx_max=0.3,
    time_idx_skip=10,
    fig=None,
    axs=None,
):
    """15-panel R/L_T scan comparison, one series per outer dirnames entry
    (matching stella_diagnostics.scan.rh_flux_scan.plot_ERH_Ephi_vs_tprim's
    convention).

    dirnames: nested list, dirnames[i_series] = flat list of run directories
    in that series. tprim is read directly from each run's own netCDF
    output (via get_growth_rate_from_flux), not supplied separately; each
    series is sorted by that value internally, so caller order within a
    series doesn't matter. A config that wants a single run instead of a
    whole series just passes a single-entry inner list.

    Per run, get_growth_rate_from_flux/estimate_eps_from_bmag/
    get_zonal_shear_profiles/get_RH_power_transfer_profiles/
    get_RH_power_time_averages are each called in their own try/except --
    a run missing e.g. the RH_fluxes_* variables still contributes its
    growth-rate/zonal-shear panels, it just leaves the RH-power-transfer
    panels as nan for that point (matching the original's per-field JSON
    fallback in spirit, but at the level of an independently-cacheable
    computation instead of a JSON dict key).

    Fixes (not preserved) relative to the original plot_param_scan_Dimits.py:
    every missing field defaults to nan uniformly (the original defaulted
    gammaE_std/gammaE_RH_std to 0 on failure -- inconsistent with every
    other field's nan default, and silently made a missing run look like a
    real zero-shear-variance run instead of missing data).

    Only the first 4 panels (heat flux x2, gammaE, gammaE_LW) are masked by
    np.isfinite(qflx_avg_vals) & ~np.isin(tprim_vals, tprim_exclude) --
    matching the original exactly; the remaining panels plot the raw
    (possibly-nan) arrays, relying on nan values not being drawn.
    """
    import seaborn as sns

    if aLT_lin_vals is None:
        aLT_lin_vals = np.zeros(len(dirnames))
    if tprim_exclude is None:
        tprim_exclude = []
    if base_colors is None:
        base_colors = sns.color_palette("rocket", len(dirnames))

    nrows = 15
    if axs is None:
        fig, axs = plt.subplots(nrows=nrows, figsize=(8, 4 * nrows))

    for i_series, series_dirnames in enumerate(dirnames):
        label = series_labels[i_series]
        color = base_colors[i_series]

        ndirs = len(series_dirnames)

        tprim_vals = np.full(ndirs, np.nan)
        qinp_vals = np.full(ndirs, np.nan)
        eps_vals = np.full(ndirs, np.nan)
        qflx_avg_vals = np.full(ndirs, np.nan)
        qflx_std_vals = np.full(ndirs, np.nan)
        gammaE_avg_vals = np.full(ndirs, np.nan)
        gammaE_std_vals = np.full(ndirs, np.nan)
        gammaE_LW_avg_vals = np.full(ndirs, np.nan)
        gammaE_LW_std_vals = np.full(ndirs, np.nan)
        upar_avg_vals = np.full(ndirs, np.nan)
        vE_avg_vals = np.full(ndirs, np.nan)
        vE_RH_avg_vals = np.full(ndirs, np.nan)
        uparcos_avg_vals = np.full(ndirs, np.nan)
        dxT_avg_vals = np.full(ndirs, np.nan)
        gammaE_RH_avg_vals = np.full(ndirs, np.nan)
        gamma_lin_max_vals = np.full(ndirs, np.nan)
        P_RH_even_avg_vals = np.full(ndirs, np.nan)
        P_RH_odd_avg_vals = np.full(ndirs, np.nan)
        P_RH_coll_avg_vals = np.full(ndirs, np.nan)
        P_RH_even_avg_LW_vals = np.full(ndirs, np.nan)
        P_RH_odd_avg_LW_vals = np.full(ndirs, np.nan)
        P_RH_coll_avg_LW_vals = np.full(ndirs, np.nan)
        dyphi2_avg_vals = np.full(ndirs, np.nan)

        run = None  # stays None if series_dirnames is empty or every load fails; guards the v_T label below
        for i_dir, dirname in enumerate(series_dirnames):
            try:
                run = StellaRun(dirname + "/" + filename, code=code)
            except Exception as e:
                print(e)
                print("Could not load run for " + dirname)
                continue

            try:
                growth = get_growth_rate_from_flux(run, time_max=time_max, qflx_rel_idx_min=qflx_rel_idx_min, qflx_rel_idx_max=qflx_rel_idx_max, time_val_avg=time_val_avg, time_avg=time_avg)
                tprim_vals[i_dir] = growth["tprim"]
                qinp_vals[i_dir] = growth["qinp"]
                qflx_avg_vals[i_dir] = growth["qflx_avg"]
                qflx_std_vals[i_dir] = growth["qflx_std"]
                gamma_lin_max_vals[i_dir] = growth["gamma_lin_max"]
            except Exception as e:
                print(e)
                print("Could not compute growth rate for " + dirname)

            try:
                eps_vals[i_dir] = estimate_eps_from_bmag(run)
            except Exception as e:
                print(e)
                print("Could not estimate eps for " + dirname)

            try:
                shear = get_zonal_shear_profiles(run, time_val_avg=time_val_avg, time_avg=time_avg, kx_max=kx_max)
                gammaE_avg_vals[i_dir] = shear["gammaE_avg"]
                gammaE_std_vals[i_dir] = shear["gammaE_std"]
                gammaE_LW_avg_vals[i_dir] = shear["gammaE_LW_avg"]
                gammaE_LW_std_vals[i_dir] = shear["gammaE_LW_std"]
                upar_avg_vals[i_dir] = shear["upar_avg"]
                vE_avg_vals[i_dir] = shear["vE_avg"]
                vE_RH_avg_vals[i_dir] = shear["vE_RH_avg"]
                uparcos_avg_vals[i_dir] = shear["uparcos_avg"]
                dxT_avg_vals[i_dir] = shear["dxT_avg"]
                gammaE_RH_avg_vals[i_dir] = shear["gammaE_RH_avg"]
            except Exception as e:
                print(e)
                print("Could not compute zonal shear profiles for " + dirname)

            try:
                transfer = get_RH_power_transfer_profiles(run, time_val_avg=time_val_avg, time_avg=time_avg)
                dyphi2_avg_vals[i_dir] = transfer["dyphi2_avg"]
            except Exception as e:
                print(e)
                print("Could not compute RH power transfer profiles for " + dirname)

            try:
                tavgs = get_RH_power_time_averages(run, time_max=time_max, time_idx_skip=time_idx_skip, time_val_avg=time_val_avg, time_avg=time_avg, kx_max=kx_max)
                P_RH_even_avg_vals[i_dir] = tavgs["P_RH_even_avg"]
                P_RH_odd_avg_vals[i_dir] = tavgs["P_RH_odd_avg"]
                P_RH_even_avg_LW_vals[i_dir] = tavgs["P_RH_even_avg_LW"]
                P_RH_odd_avg_LW_vals[i_dir] = tavgs["P_RH_odd_avg_LW"]
                if tavgs["P_RH_coll_avg"] != 0:
                    P_RH_coll_avg_vals[i_dir] = tavgs["P_RH_coll_avg"]
                    P_RH_coll_avg_LW_vals[i_dir] = tavgs["P_RH_coll_avg_LW"]
            except Exception as e:
                print(e)
                print("Could not compute RH power time averages for " + dirname)

        # Directories may arrive in any order (previously implicit via
        # sorted(glob(...)) sorting by directory name); sort every parallel
        # array by the tprim actually read from each run instead.
        idx_sort = np.argsort(tprim_vals)
        (
            tprim_vals, qinp_vals, eps_vals, qflx_avg_vals, qflx_std_vals,
            gammaE_avg_vals, gammaE_std_vals, gammaE_LW_avg_vals, gammaE_LW_std_vals,
            upar_avg_vals, vE_avg_vals, vE_RH_avg_vals, uparcos_avg_vals, dxT_avg_vals,
            gammaE_RH_avg_vals, gamma_lin_max_vals, P_RH_even_avg_vals, P_RH_odd_avg_vals,
            P_RH_coll_avg_vals, P_RH_even_avg_LW_vals, P_RH_odd_avg_LW_vals,
            P_RH_coll_avg_LW_vals, dyphi2_avg_vals,
        ) = (
            arr[idx_sort] for arr in (
                tprim_vals, qinp_vals, eps_vals, qflx_avg_vals, qflx_std_vals,
                gammaE_avg_vals, gammaE_std_vals, gammaE_LW_avg_vals, gammaE_LW_std_vals,
                upar_avg_vals, vE_avg_vals, vE_RH_avg_vals, uparcos_avg_vals, dxT_avg_vals,
                gammaE_RH_avg_vals, gamma_lin_max_vals, P_RH_even_avg_vals, P_RH_odd_avg_vals,
                P_RH_coll_avg_vals, P_RH_even_avg_LW_vals, P_RH_odd_avg_LW_vals,
                P_RH_coll_avg_LW_vals, dyphi2_avg_vals,
            )
        )

        if substract_lin:
            tprim_vals = tprim_vals - aLT_lin_vals[i_series]

        mask = np.isfinite(qflx_avg_vals) & ~np.isin(tprim_vals, tprim_exclude)

        ##### Plot
        # Heat flux
        i = 0
        ax = axs[i]
        ax.errorbar(tprim_vals[mask], qflx_avg_vals[mask], qflx_std_vals[mask], c=color, label=label, lw=2, marker="o", markersize=markersize)
        ax.set_ylabel(r"$Q/Q_\mathrm{gB}$")
        ax.set_ylim(ymin=0)

        # Heat flux on log scale
        i += 1
        ax = axs[i]
        ax.errorbar(tprim_vals[mask], qflx_avg_vals[mask], qflx_std_vals[mask], c=color, label=label, lw=2, marker="o", markersize=markersize)
        ax.set_yscale("log")
        ax.set_ylabel(r"$Q/Q_\mathrm{gB}$")

        # Flow shear and max linear growth rate
        i += 1
        ax = axs[i]
        if i_series == 0:
            vt_label = get_vt_label(run.ncdata) if run is not None else "T"
            labels = [r"$\gamma_E \, R/%s$" % vt_label, r"$\gamma_{E, \mathrm{RH}} \, R/%s$" % vt_label, r"$\gamma_\mathrm{lin}^\mathrm{max} \, R / %s$" % vt_label]
        else:
            labels = [None, None, None]
        ax.errorbar(tprim_vals[mask], gammaE_avg_vals[mask], 0.5 * gammaE_std_vals[mask], lw=2, c=color, ls="--", marker="o", markersize=markersize)
        ax.errorbar(tprim_vals[mask], gammaE_RH_avg_vals[mask], 0.5 * gammaE_std_vals[mask], lw=2, c=color, ls="--", marker="s", markersize=markersize, alpha=0.3)
        ax.plot(tprim_vals[mask], gamma_lin_max_vals[mask], lw=4, c=color, ls="-", marker="x", markersize=markersize, alpha=0.5)

        ax.plot([], [], c="k", ls="None", lw=2, marker="o", label=labels[0], markersize=markersize)
        ax.plot([], [], c="k", ls="--", lw=2, marker="s", label=labels[1], markersize=markersize, alpha=0.3)
        ax.plot([], [], c="k", ls="-", lw=4, marker="x", label=labels[2], markersize=markersize)

        ax.set_ylim(ymin=0)

        # LW Flow shear and max linear growth rate
        i += 1
        ax = axs[i]
        if i_series == 0:
            vt_label = get_vt_label(run.ncdata) if run is not None else "T"
            labels = [r"$\gamma_E^\mathrm{LW} \, R/%s$" % vt_label, r"$\gamma_{E, \mathrm{RH}}^\mathrm{LW} \, R/%s$" % vt_label, r"$\gamma_\mathrm{lin}^\mathrm{max} \, R / %s$" % vt_label]
        else:
            labels = [None, None, None]
        ax.errorbar(tprim_vals[mask], gammaE_LW_avg_vals[mask], 0.5 * gammaE_LW_std_vals[mask], lw=2, c=color, ls="--", marker="o", markersize=markersize)
        ax.plot(tprim_vals[mask], gamma_lin_max_vals[mask], lw=4, c=color, ls="-", marker="x", markersize=markersize, alpha=0.5)

        ax.plot([], [], c="k", ls="--", lw=2, marker="o", label=labels[0], markersize=markersize)
        ax.plot([], [], c="k", ls="-", lw=4, marker="x", label=labels[2], markersize=markersize)

        ax.set_ylim(ymin=0, ymax=0.25)

        # P_RH/P_RH^+
        i += 1
        ax = axs[i]
        if i_series == 0:
            labels = [r"$-P_\mathrm{RH}^C/P_\mathrm{RH}^+$", r"$-P_\mathrm{RH}^-/P_\mathrm{RH}^+$", r"$-(P_\mathrm{RH}^C+P_\mathrm{RH}^-)/P_\mathrm{RH}^+$"]
        else:
            labels = [None, None, None]

        ax.plot(tprim_vals, -P_RH_coll_avg_vals / P_RH_even_avg_vals, c=color, ls="-.", lw=2, marker="v", markersize=markersize, alpha=0.5)
        ax.plot(tprim_vals, -P_RH_odd_avg_vals / P_RH_even_avg_vals, c=color, ls=":", lw=2, marker="s", markersize=markersize, alpha=0.5)
        ax.plot(tprim_vals, -(P_RH_coll_avg_vals + P_RH_odd_avg_vals) / P_RH_even_avg_vals, c=color, ls="-", lw=4, marker="o", markersize=2 * markersize)

        ax.plot([], [], c="k", ls="-.", lw=2, marker="v", label=labels[0], markersize=markersize, alpha=0.5)
        ax.plot([], [], c="k", ls=":", lw=2, marker="s", label=labels[1], markersize=markersize, alpha=0.5)
        ax.plot([], [], c="k", ls="-", lw=4, marker="o", label=labels[2], markersize=markersize)

        # P_RH +-
        i += 1
        ax = axs[i]
        if i_series == 0:
            labels = [r"$P_\mathrm{RH}^+$", r"$-P_\mathrm{RH}^-$", r"$-P_\mathrm{RH}^C$"]
        else:
            labels = [None, None, None]

        ax.plot(tprim_vals, P_RH_even_avg_vals, c=color, ls="-", lw=2, marker="o", markersize=markersize)
        ax.plot(tprim_vals, -P_RH_odd_avg_vals, c=color, ls=":", lw=2, marker="s", markersize=markersize)
        ax.plot(tprim_vals, -P_RH_coll_avg_vals, c=color, ls="-.", lw=2, marker="v", markersize=markersize)

        ax.plot([], [], c="k", ls="-", lw=2, marker="o", label=labels[0], markersize=markersize)
        ax.plot([], [], c="k", ls=":", lw=2, marker="s", label=labels[1], markersize=markersize)
        ax.plot([], [], c="k", ls="-.", lw=2, marker="v", label=labels[2], markersize=markersize)
        ax.set_yscale("symlog", linthresh=1e-5)

        # P_RH (linear)
        i += 1
        ax = axs[i]
        if i_series == 0:
            labels = [r"$P_\mathrm{RH}^+$", r"$P_\mathrm{RH}^-$", r"$P_\mathrm{RH}$"]
        else:
            labels = [None, None, None]

        ax.plot(tprim_vals, P_RH_even_avg_vals, c=color, ls="-", lw=2, marker="o", markersize=markersize)
        ax.plot(tprim_vals, P_RH_odd_avg_vals, c=color, ls=":", lw=2, marker="s", markersize=markersize)
        P_RH_tot_avg_vals = P_RH_even_avg_vals + P_RH_odd_avg_vals
        ax.plot(tprim_vals, P_RH_tot_avg_vals, c=color, ls="-.", lw=2, marker=".", markersize=markersize)

        ax.plot([], [], c="k", ls="-", lw=2, marker="o", label=labels[0], markersize=markersize)
        ax.plot([], [], c="k", ls=":", lw=2, marker="s", label=labels[1], markersize=markersize)
        ax.plot([], [], c="k", ls="-.", lw=2, marker=".", label=labels[2], markersize=markersize)

        # P_RH +- (LW)
        i += 1
        ax = axs[i]
        if i_series == 0:
            labels = [r"$P_\mathrm{RH}^+$ (LW)", r"$-P_\mathrm{RH}^-$ (LW)", r"$-P_\mathrm{RH}^C$ (LW)"]
        else:
            labels = [None, None, None]

        ax.plot(tprim_vals, P_RH_even_avg_LW_vals, c=color, ls="-", lw=2, marker="o", markersize=markersize)
        ax.plot(tprim_vals, -P_RH_odd_avg_LW_vals, c=color, ls=":", lw=2, marker="s", markersize=markersize)
        ax.plot(tprim_vals, -P_RH_coll_avg_LW_vals, c=color, ls="-.", lw=2, marker="v", markersize=markersize)

        ax.plot([], [], c="k", ls="-", lw=2, marker="o", label=labels[0], markersize=markersize)
        ax.plot([], [], c="k", ls=":", lw=2, marker="s", label=labels[1], markersize=markersize)
        ax.plot([], [], c="k", ls="-.", lw=2, marker="v", label=labels[2], markersize=markersize)
        ax.set_yscale("symlog", linthresh=1e-5)

        # Zonal parallel flow
        i += 1
        ax = axs[i]
        if i_series == 0:
            labels = [r"$u_\parallel / (q/\epsilon \cdot v_E)$", r"$2 u_\parallel \cos\theta / (q v_E)$"]
        else:
            labels = [None, None, None]
        ax.plot(tprim_vals, 2 * upar_avg_vals / (qinp_vals / eps_vals * vE_RH_avg_vals), lw=2, c=color, ls="-", marker="o", markersize=markersize)
        ax.plot(tprim_vals, 2 * 2 * uparcos_avg_vals / (2 * qinp_vals * vE_RH_avg_vals), lw=2, c=color, ls=":", marker="s", markersize=markersize)

        ax.plot([], [], c="k", ls="-", lw=2, marker="o", label=labels[0], markersize=markersize)
        ax.plot([], [], c="k", ls=":", lw=2, marker="s", label=labels[1], markersize=markersize)
        ax.set_ylim(ymin=0)

        # Ratio of <upar>/<upar*cos(theta)>
        i += 1
        ax = axs[i]
        ax.plot(tprim_vals, 0.5 * upar_avg_vals / uparcos_avg_vals, lw=2, c=color, ls="-", marker="o", markersize=markersize)
        ax.set_ylabel(r"$\langle u_\parallel \rangle_\theta / 2\langle u_\parallel \cos\theta \rangle_\theta$")

        # <upar>
        i += 1
        ax = axs[i]
        ax.plot(tprim_vals, upar_avg_vals, lw=2, c=color, ls="-", marker="o", markersize=markersize)
        ax.set_ylabel(r"$ \langle u_\parallel \rangle_\theta $")

        # Zonal temperature gradient
        i += 1
        ax = axs[i]
        if i_series == 0:
            labels = [r"$v_\mathrm{dia}/v_E$"]
        else:
            labels = [None, None, None]

        ax.plot(tprim_vals, dxT_avg_vals / vE_avg_vals, lw=2, c=color, ls="--", marker="x", markersize=markersize)
        ax.plot([], [], c="k", ls="--", lw=2, marker="x", label=labels[0], markersize=markersize)
        ax.set_ylim(ymin=0)

        # Relative difference between RH and non-RH ZF shear
        i += 1
        ax = axs[i]
        ax.plot(tprim_vals, np.abs(gammaE_RH_avg_vals - gammaE_avg_vals) / gammaE_avg_vals, lw=2, c=color, marker="s", markersize=markersize)
        ax.set_ylabel(r"$|\gamma_{E}-\gamma_{E, \mathrm{EH}}|/\gamma_{E}$")
        ax.set_ylim(ymax=1)

        # Plot turbulence amplitude (total and vE><0)
        i += 1
        ax = axs[i]
        ax.semilogy(tprim_vals, qflx_avg_vals * 10, lw=2, c=color, marker="o", markersize=markersize, alpha=0.5)
        ax.semilogy(tprim_vals, dyphi2_avg_vals, lw=2, c=color, marker="s", markersize=markersize)
        ax.set_ylabel(r"$\tilde v_{Ex}^2$")

        # Plot ratio of vEx^2/vEZ^2
        i += 1
        ax = axs[i]
        ax.semilogy(tprim_vals, dyphi2_avg_vals / vE_avg_vals ** 2, lw=2, c=color, marker="s", markersize=markersize)
        ax.set_ylabel(r"$(\tilde v_{Ex}/v_E^Z)^2$")

    # Beautify and save plot
    for ax in axs:
        if substract_lin:
            ax.set_xlabel(r"$R/L_T-(R/L_T)_\mathrm{lin}$")
            ax.set_xlim(xmin=0)
        else:
            ax.set_xlabel(r"$R/L_T$")
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=22)

        if xlim is not None:
            ax.set_xlim(xlim)

    return fig, axs
