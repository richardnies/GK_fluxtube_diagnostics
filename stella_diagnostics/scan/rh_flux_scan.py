"""Qflx-vs-collisionality and E_RH/E_phi-vs-tprim scan comparisons.

Extracted from example_plots/plot_flux_coll.py and plot_ERH_Ephi.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as specialfunc

from stella_diagnostics.io.codes import get_vt_label
from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.physics.gradients import get_aLT_lin_analytic
from stella_diagnostics.scan.zonal_flow_scan import (
    estimate_eps_from_bmag,
    get_growth_rate_from_flux,
    get_zonal_shear_profiles,
)


def _dt_weighted_mean(x, t):
    """dt-weighted mean of x(t) -- collapses the
    np.sum(x*np.gradient(t))/np.sum(np.gradient(t)) pattern repeated across
    this module and stella_diagnostics.scan.flux_energy_scan."""
    dt = np.gradient(t)
    return np.sum(x * dt) / np.sum(dt)


def plot_qflx_vs_nu_scan(
    dirs_nu,
    vals_nu,
    tprim_vals,
    filename="CBC",
    code="stella",
    time_avg=200,
    rhoc=0.18,
    q=1.4,
    shat=0.8,
    colors_tprim=None,
    fig=None,
    axs=None,
):
    """Qflx(nu)/gammaE(nu)/vE_RH(nu)/upar(nu) comparison across a
    collisionality x tprim sweep, one errorbar/line series per tprim.

    dirs_nu/vals_nu: parallel lists of base directories and their
    collisionality (nu_ii) values; each run is
    f"{dirs_nu[i]}/run_tprim-{tprim:.4f}/{filename}".
    time_avg: trailing-window width (time units before the run's last
    sample) used both for the qflx average here and (as a fallback, and
    for the zonal-shear-profile fields) via
    stella_diagnostics.scan.zonal_flow_scan's own time_avg -- same
    trailing-from-run-end convention (matches zonal_flow_scan's default
    time_val_avg=None branch exactly, so this is a shared, not merely
    similarly-named, window).
    Returns (fig, axs) with 5 panels.
    """
    import seaborn as sns

    if colors_tprim is None:
        colors_tprim = sns.color_palette("rocket", len(tprim_vals))
    if axs is None:
        fig, axs = plt.subplots(nrows=5, figsize=(9, 25))

    run = None  # stays None if every load below fails; guards the v_T label at the end
    for i_tprim, tprim in enumerate(tprim_vals):
        qflx_avg_nu, qflx_std_nu = [], []
        vE_RH_avg_nu, vE_avg_nu, gammaE_avg_nu, upar_avg_nu = [], [], [], []
        nu_plot = []

        for i_nu, nu in enumerate(vals_nu):
            dirname = dirs_nu[i_nu] + "/" + "run_tprim-%.4f" % tprim

            try:
                run = StellaRun(dirname + "/" + filename, code=code)
            except Exception as e:
                print(e)
                print("Could not load run for " + dirname)
                continue

            try:
                _, vflx, qflx, time = run.get_fluxes_over_time(norm=False)

                qflx = qflx[time > time[-1] - time_avg]
                time = time[time > time[-1] - time_avg]

                qflx_avg_nu.append(_dt_weighted_mean(qflx, time))
                qflx_std_nu.append(np.std(qflx))
                nu_plot.append(nu)
            except Exception as e:
                print(e)
                print("TRYING get_growth_rate_from_flux (reads qflx from the netCDF output instead of the .fluxes file)")
                try:
                    growth = get_growth_rate_from_flux(run, time_avg=time_avg)
                    qflx_avg_nu.append(growth["qflx_avg"])
                    qflx_std_nu.append(growth["qflx_std"])
                    nu_plot.append(nu)
                except Exception as e2:
                    print(e2)
                    continue

            try:
                shear = get_zonal_shear_profiles(run, time_avg=time_avg)
                vE_RH_avg_nu.append(shear["vE_RH_avg"])
                vE_avg_nu.append(shear["vE_avg"])
                gammaE_avg_nu.append(shear["gammaE_avg"])
                upar_avg_nu.append(shear["upar_avg"])
            except Exception as e:
                print(e)
                vE_RH_avg_nu.append(None)
                vE_avg_nu.append(None)
                gammaE_avg_nu.append(None)
                upar_avg_nu.append(None)

        Delta_tprim = tprim - get_aLT_lin_analytic(rhoc=rhoc, q=q, shat=shat)

        idx_sort = np.argsort(nu_plot)
        nu_plot = np.array(nu_plot)[idx_sort]
        qflx_avg_nu = np.array(qflx_avg_nu)[idx_sort]
        qflx_std_nu = np.array(qflx_std_nu)[idx_sort]
        vE_RH_avg_nu = np.array(vE_RH_avg_nu)[idx_sort]
        vE_avg_nu = np.array(vE_avg_nu)[idx_sort]
        gammaE_avg_nu = np.array(gammaE_avg_nu)[idx_sort]
        upar_avg_nu = np.array(upar_avg_nu)[idx_sort]

        ax = axs[0]
        ax.errorbar(nu_plot[0], qflx_avg_nu[0] / Delta_tprim, qflx_std_nu[0] / Delta_tprim, c=colors_tprim[i_tprim], marker="s", markersize=20)
        ax.errorbar(nu_plot[1:], qflx_avg_nu[1:] / Delta_tprim, qflx_std_nu[1:] / Delta_tprim, c=colors_tprim[i_tprim], marker="o", label=r"$R/L_T = %.2f$" % tprim)

        try:
            ax = axs[1]
            ax.plot(nu_plot[0], gammaE_avg_nu[0], c=colors_tprim[i_tprim], marker="s", markersize=20)
            ax.plot(nu_plot[1:], gammaE_avg_nu[1:], c=colors_tprim[i_tprim], marker="o", label=r"$R/L_T = %.2f$" % tprim)

            ax = axs[2]
            ax.plot(nu_plot[0], vE_RH_avg_nu[0], c=colors_tprim[i_tprim], marker="s", markersize=20)
            ax.plot(nu_plot[1:], vE_RH_avg_nu[1:], c=colors_tprim[i_tprim], marker="o", label=r"$R/L_T = %.2f$" % tprim)

            ax = axs[3]
            ax.plot(nu_plot[0], upar_avg_nu[0], c=colors_tprim[i_tprim], marker="s", markersize=20)
            ax.plot(nu_plot[1:], upar_avg_nu[1:], c=colors_tprim[i_tprim], marker="o", label=r"$R/L_T = %.2f$" % tprim)

            ax = axs[4]
            ax.plot(nu_plot[0], upar_avg_nu[0] / vE_RH_avg_nu[0], c=colors_tprim[i_tprim], marker="s", markersize=20)
            ax.plot(nu_plot[1:], upar_avg_nu[1:] / vE_RH_avg_nu[1:], c=colors_tprim[i_tprim], marker="o", label=r"$R/L_T = %.2f$" % tprim)
        except Exception:
            continue

    nu_shift = vals_nu[0]
    nu_th = np.linspace(1e-4, 1e-2, 100)
    for i in (1, 2, 3):
        axs[i].plot(nu_th, (nu_th / 1e-4) ** (-1 / 2), ls="--", c="0.5", label=r"$\propto \nu_{ii}^{-1/2}$")

    # axs[0] (Q/Q_gB / (R/L_T - Jenko)) can be negative -- symlog (not log)
    # so it doesn't just silently disappear; axs[1:] (gammaE/vE_RH/upar,
    # all non-negative by construction, being sqrt(<x^2>)) stay log.
    axs[0].set_yscale("symlog")
    for ax in axs[1:]:
        ax.set_yscale("log")
    vt_label = get_vt_label(run.ncdata) if run is not None else "v_T"
    for ax in axs:
        ax.set_xscale("log")
        ax.legend(fontsize=12)
        ax.set_xlabel(r"$\nu_{ii}R/%s$" % vt_label)
        ax.set_xlim(xmin=nu_shift)
        ax.grid()

    axs[0].set_ylabel(r"$(Q/Q_\mathrm{gB})/(R/L_T - (R/L_T)_\mathrm{Jenko})$")
    axs[1].set_ylabel(r"$\langle \gamma_{E}^2 \rangle^{1/2}$")
    axs[2].set_ylabel(r"$\langle v_{E,\mathrm{RH}}^2 \rangle^{1/2}$")
    axs[3].set_ylabel(r"$\langle u_\parallel^2 \rangle^{1/2}$")
    axs[4].set_ylabel(r"$\langle u_\parallel^2 \rangle^{1/2} / \langle v_{E,\mathrm{RH}}^2 \rangle^{1/2}$")

    return fig, axs


def _gamma0_kx_only(kx):
    """(1-Gamma0) approximation using kperp2 ~= kx**2 directly, with no
    geometric (gds22/bmag/shat) correction, operating on stella's own
    pre-zed-integrated phi2_vs_kxky diagnostic.

    NOTE: this is a third, distinct Gamma0 approximation from the ones in
    stella_diagnostics.scan.flux_energy_scan (per-zed array at ky=0, from
    raw phi_vs_t, with full gds22/bmag geometric correction) and
    stella_diagnostics.quantities.registry.get_Gamma0 (single (kx,ky) point
    read directly from the netCDF kperp2 variable). Confirmed non-equivalent
    by direct comparison of their inputs/formulas -- kept separate rather
    than silently merged, per this project's established convention.
    """
    return specialfunc.iv(0, kx**2 / 2) * np.exp(-(kx**2) / 2)


def plot_ERH_Ephi_vs_tprim(
    base_dirs,
    base_labels,
    aLT_lin_vals,
    filename="CBC",
    code="stella",
    time_avg=800,
    base_colors=None,
    markersize=10,
    fig=None,
    axs=None,
):
    """E_RH(tprim)/E_phi(tprim)/E_RH/E_phi/chihat/gammaE comparison across a
    set of base_dirs, each glob-discovered for run_tprim*00 subdirectories,
    one series per base_dir.

    aLT_lin_vals: one linear critical-gradient value per base_dir (typically
    from get_aLT_lin_analytic), subtracted from that dir's tprim values.
    time_avg: trailing-window width (time units before the run's last
    sample), shared across qflx/gammaE (via zonal_flow_scan's own
    matching trailing-from-run-end convention) and the E_RH/E_phi
    integrals computed directly in this function.
    Returns (fig, axs) with 5 panels: E_RH, E_phi, E_RH/E_phi, chihat
    (normalized to the last dir's own largest-R/L_T value), gammaE.
    """
    from glob import glob

    import seaborn as sns

    if base_colors is None:
        base_colors = sns.color_palette("rocket", len(base_dirs))
    if axs is None:
        fig, axs = plt.subplots(nrows=5, figsize=(9, 25))

    chihat_norm = None

    for i_base, base_dir in enumerate(base_dirs):
        label = base_labels[i_base]
        color = base_colors[i_base]

        dirnames = sorted(glob(base_dir + "/run_tprim*00/"))
        ndirs = len(dirnames)

        tprim_vals = np.zeros(ndirs)
        qinp_vals = np.zeros(ndirs)
        eps_vals = np.zeros(ndirs)
        qflx_avg_vals = np.zeros(ndirs)
        gammaE_avg_vals = np.zeros(ndirs)
        gammaE_std_vals = np.zeros(ndirs)
        ERH_vals = np.zeros(ndirs)
        Ephi_vals = np.zeros(ndirs)

        for i_dir, dirname in enumerate(dirnames):
            try:
                run = StellaRun(dirname + "/" + filename, code=code)

                growth = get_growth_rate_from_flux(run, time_avg=time_avg)
                tprim_vals[i_dir] = growth["tprim"]
                qinp_vals[i_dir] = growth["qinp"]
                qflx_avg_vals[i_dir] = growth["qflx_avg"]

                eps_vals[i_dir] = estimate_eps_from_bmag(run)

                shear = get_zonal_shear_profiles(run, time_avg=time_avg)
                gammaE_avg_vals[i_dir] = shear["gammaE_avg"]
                gammaE_std_vals[i_dir] = shear["gammaE_std"]

                time_min = run.ncdata.variables["t"][-1] - time_avg
                time_max = run.ncdata.variables["t"][-1]

                E_RH_t_kx, RH_time, RH_kx = run.get_E_RH_t_kx(time_min=time_min, time_max=time_max)
                ERH_vals[i_dir] = np.sum(E_RH_t_kx * np.gradient(RH_time)[:, None]) / np.sum(np.gradient(RH_time))

                phi2_t_kx_ky, time, kx, ky = run.read_phi2_spectra(time_min=time_min, time_max=time_max)
                Gamma0 = _gamma0_kx_only(kx)
                Ephi_vals[i_dir] = 0.5 * np.sum((1 - Gamma0)[None, :] * phi2_t_kx_ky[:, :, 0] * np.gradient(time)[:, None]) / np.sum(np.gradient(time))

            except Exception as e:
                print(e)
                print("Could not load file for " + dirname)
                tprim_vals[i_dir] = np.nan
                qinp_vals[i_dir] = np.nan
                eps_vals[i_dir] = np.nan
                qflx_avg_vals[i_dir] = np.nan
                gammaE_avg_vals[i_dir] = np.nan
                gammaE_std_vals[i_dir] = np.nan
                ERH_vals[i_dir] = np.nan
                Ephi_vals[i_dir] = np.nan

        tprim_vals += -aLT_lin_vals[i_base]

        ax = axs[0]
        ax.plot(tprim_vals, ERH_vals, c=color, marker="o", markersize=markersize, label=label)
        ax.set_ylabel(r"$E_\mathrm{RH}$")
        ax.set_yscale("log")

        ax = axs[1]
        ax.plot(tprim_vals, Ephi_vals, c=color, marker="o", markersize=markersize, label=label)
        ax.set_ylabel(r"$E_\varphi$")
        ax.set_yscale("log")

        ax = axs[2]
        ax.plot(tprim_vals, ERH_vals / Ephi_vals, c=color, marker="o", markersize=markersize, label=label)
        ax.set_ylabel(r"$E_\mathrm{RH}/E_\varphi$")

        ax = axs[3]
        chihat = qflx_avg_vals / tprim_vals
        if chihat_norm is None:
            chihat_norm = chihat[-1]
        ax.plot(tprim_vals, chihat / chihat_norm, c=color, marker="o", markersize=markersize, label=label)
        ax.set_ylabel(r"$\hat \chi_i / \hat \chi_i(R/L_T)_\mathrm{max}$")

        ax = axs[4]
        ax.errorbar(tprim_vals, gammaE_avg_vals, gammaE_std_vals, c=color, marker="o", markersize=markersize, label=label)
        ax.set_ylabel(r"$\langle \gamma_E^2 \rangle^{1/2}$")

        for ax in axs:
            ax.set_xlabel(r"$a/L_T-(a/L_T)_\mathrm{lin}$")

    for ax in axs:
        ax.grid(True)
        ax.legend(fontsize=22)
        ax.set_xlim(xmin=0)

    return fig, axs
