"""Per-kx Rosenbluth-Hinton residual-energy/power means, and the two
"summed over kx" / "vs kx" comparison figures built from them.

Extracted from example_plots/plot_RH_phi_E_P_t_all_kx.py. Individual per-kx
diagnostic figures (plot_RH_phi_I/plot_E_RH/plot_P_RH) are NOT extracted --
they already delegate correctly to stella_diagnostics.physics.rosenbluth_hinton
and stay a thin loop in the driver script.
"""

import matplotlib.pyplot as plt
import numpy as np

from stella_diagnostics.io.cache import cached
from stella_diagnostics.io.codes import get_rho_label, get_vt_label
from stella_diagnostics.physics.rosenbluth_hinton import get_E_RH_t_kx, get_P_RH_breakdown
from stella_diagnostics.spectral.stats import dt_weighted_mean, dt_weights


@cached(version=2)
def get_RH_per_kx_means(
    run,
    time_min=500,
    time_max=1e6,
    kx_max=1e4,
    passing_trapped="both",
    fphi=1,
    fapar=1,
    fbpar=1,
    fcoll=1,
    D_hyper=None,
) -> dict:
    """dt-weighted-per-kx-mean and cross-kx running-sum-vs-time RH
    quantities, computed with one pass over kx (0 < kx <= kx_max).

    This is the structural fix for the original script's documented bug:
    with the old .dat-file cache, a cache HIT skipped the per-kx loop
    entirely, so the cross-kx sum-vs-time quantities (only ever assigned
    inside that loop) were silently never computed and the "summed over
    kx" figure was never generated. With @cached, cache hit or miss
    returns this same dict either way, so that figure's inputs are always
    present.
    """
    kx_all = run.ncdata["kx"][:]

    E_RH_mean_kx = np.zeros_like(kx_all)
    P_RH_num_mean_kx = np.zeros_like(kx_all)
    P_RH_even_mean_kx = np.zeros_like(kx_all)
    P_RH_odd_mean_kx = np.zeros_like(kx_all)
    P_RH_phi_even_mean_kx = np.zeros_like(kx_all)
    P_RH_phi_odd_mean_kx = np.zeros_like(kx_all)
    P_RH_apar_even_mean_kx = np.zeros_like(kx_all)
    P_RH_apar_odd_mean_kx = np.zeros_like(kx_all)
    P_RH_bpar_even_mean_kx = np.zeros_like(kx_all)
    P_RH_bpar_odd_mean_kx = np.zeros_like(kx_all)
    P_RH_coll_even_mean_kx = np.zeros_like(kx_all)
    P_RH_coll_odd_mean_kx = np.zeros_like(kx_all)
    P_RH_hyper_mean_kx = np.zeros_like(kx_all) if D_hyper is not None else None

    t_out = None
    E_RH_t_sumkx = P_RH_phi_even_t_sumkx = P_RH_phi_odd_t_sumkx = P_RH_coll_t_sumkx = None
    P_RH_apar_even_t_sumkx = P_RH_apar_odd_t_sumkx = None
    P_RH_bpar_even_t_sumkx = P_RH_bpar_odd_t_sumkx = None
    P_RH_hyper_t_sumkx = None

    i_fig = 0
    for i_kx in range(len(kx_all)):
        if kx_all[i_kx] <= 0 or np.abs(kx_all[i_kx]) > kx_max:
            continue

        idxs_kx = np.array([i_kx])

        E_RH_t_kx, t, kx = get_E_RH_t_kx(run, time_min=time_min, time_max=time_max, idxs_kx=idxs_kx)
        (
            P_RH_even_t_kx,
            P_RH_odd_t_kx,
            P_RH_phi_even_t_kx,
            P_RH_phi_odd_t_kx,
            P_RH_apar_even_t_kx,
            P_RH_apar_odd_t_kx,
            P_RH_bpar_even_t_kx,
            P_RH_bpar_odd_t_kx,
            P_RH_coll_even_t_kx,
            P_RH_coll_odd_t_kx,
            P_RH_hyper_t_kx,
            time2,
            kx2,
        ) = get_P_RH_breakdown(
            run,
            passing_trapped=passing_trapped,
            time_min=time_min,
            time_max=time_max,
            idxs_kx=idxs_kx,
            fphi=fphi,
            fapar=fapar,
            fbpar=fbpar,
            fcoll=fcoll,
            D_hyper=D_hyper,
        )

        if i_fig == 0:
            t_out = t
            E_RH_t_sumkx = np.sum(E_RH_t_kx, axis=1)
            P_RH_phi_even_t_sumkx = np.sum(P_RH_phi_even_t_kx, axis=1)
            P_RH_phi_odd_t_sumkx = np.sum(P_RH_phi_odd_t_kx, axis=1)
            P_RH_apar_even_t_sumkx = np.sum(P_RH_apar_even_t_kx, axis=1)
            P_RH_apar_odd_t_sumkx = np.sum(P_RH_apar_odd_t_kx, axis=1)
            P_RH_bpar_even_t_sumkx = np.sum(P_RH_bpar_even_t_kx, axis=1)
            P_RH_bpar_odd_t_sumkx = np.sum(P_RH_bpar_odd_t_kx, axis=1)
            P_RH_coll_t_sumkx = np.sum(P_RH_coll_even_t_kx + P_RH_coll_odd_t_kx, axis=1)
            if D_hyper is not None:
                P_RH_hyper_t_sumkx = np.sum(P_RH_hyper_t_kx, axis=1)
        else:
            E_RH_t_sumkx = E_RH_t_sumkx + np.sum(E_RH_t_kx, axis=1)
            P_RH_phi_even_t_sumkx = P_RH_phi_even_t_sumkx + np.sum(P_RH_phi_even_t_kx, axis=1)
            P_RH_phi_odd_t_sumkx = P_RH_phi_odd_t_sumkx + np.sum(P_RH_phi_odd_t_kx, axis=1)
            P_RH_apar_even_t_sumkx = P_RH_apar_even_t_sumkx + np.sum(P_RH_apar_even_t_kx, axis=1)
            P_RH_apar_odd_t_sumkx = P_RH_apar_odd_t_sumkx + np.sum(P_RH_apar_odd_t_kx, axis=1)
            P_RH_bpar_even_t_sumkx = P_RH_bpar_even_t_sumkx + np.sum(P_RH_bpar_even_t_kx, axis=1)
            P_RH_bpar_odd_t_sumkx = P_RH_bpar_odd_t_sumkx + np.sum(P_RH_bpar_odd_t_kx, axis=1)
            P_RH_coll_t_sumkx = P_RH_coll_t_sumkx + np.sum(P_RH_coll_even_t_kx + P_RH_coll_odd_t_kx, axis=1)
            if D_hyper is not None:
                P_RH_hyper_t_sumkx = P_RH_hyper_t_sumkx + np.sum(P_RH_hyper_t_kx, axis=1)

        dt = dt_weights(t)
        E_RH_mean_kx[idxs_kx] = dt_weighted_mean(E_RH_t_kx, weights=dt, axis=0)
        P_RH_num_mean_kx[idxs_kx] = (E_RH_t_kx[-1] - E_RH_t_kx[0]) / (t[-1] - t[0])
        P_RH_even_mean_kx[idxs_kx] = dt_weighted_mean(P_RH_even_t_kx, weights=dt, axis=0)
        P_RH_odd_mean_kx[idxs_kx] = dt_weighted_mean(P_RH_odd_t_kx, weights=dt, axis=0)
        P_RH_phi_even_mean_kx[idxs_kx] = dt_weighted_mean(P_RH_phi_even_t_kx, weights=dt, axis=0)
        P_RH_phi_odd_mean_kx[idxs_kx] = dt_weighted_mean(P_RH_phi_odd_t_kx, weights=dt, axis=0)
        P_RH_apar_even_mean_kx[idxs_kx] = dt_weighted_mean(P_RH_apar_even_t_kx, weights=dt, axis=0)
        P_RH_apar_odd_mean_kx[idxs_kx] = dt_weighted_mean(P_RH_apar_odd_t_kx, weights=dt, axis=0)
        P_RH_bpar_even_mean_kx[idxs_kx] = dt_weighted_mean(P_RH_bpar_even_t_kx, weights=dt, axis=0)
        P_RH_bpar_odd_mean_kx[idxs_kx] = dt_weighted_mean(P_RH_bpar_odd_t_kx, weights=dt, axis=0)
        P_RH_coll_even_mean_kx[idxs_kx] = dt_weighted_mean(P_RH_coll_even_t_kx, weights=dt, axis=0)
        P_RH_coll_odd_mean_kx[idxs_kx] = dt_weighted_mean(P_RH_coll_odd_t_kx, weights=dt, axis=0)
        if D_hyper is not None:
            P_RH_hyper_mean_kx[idxs_kx] = dt_weighted_mean(P_RH_hyper_t_kx, weights=dt, axis=0)

        i_fig += 1

    result = {
        "kx_all": kx_all,
        "t": t_out if t_out is not None else np.array([]),
        "E_RH_mean_kx": E_RH_mean_kx,
        "P_RH_num_mean_kx": P_RH_num_mean_kx,
        "P_RH_even_mean_kx": P_RH_even_mean_kx,
        "P_RH_odd_mean_kx": P_RH_odd_mean_kx,
        "P_RH_phi_even_mean_kx": P_RH_phi_even_mean_kx,
        "P_RH_phi_odd_mean_kx": P_RH_phi_odd_mean_kx,
        "P_RH_apar_even_mean_kx": P_RH_apar_even_mean_kx,
        "P_RH_apar_odd_mean_kx": P_RH_apar_odd_mean_kx,
        "P_RH_bpar_even_mean_kx": P_RH_bpar_even_mean_kx,
        "P_RH_bpar_odd_mean_kx": P_RH_bpar_odd_mean_kx,
        "P_RH_coll_even_mean_kx": P_RH_coll_even_mean_kx,
        "P_RH_coll_odd_mean_kx": P_RH_coll_odd_mean_kx,
        "E_RH_t_sumkx": E_RH_t_sumkx if E_RH_t_sumkx is not None else np.array([]),
        "P_RH_phi_even_t_sumkx": P_RH_phi_even_t_sumkx if P_RH_phi_even_t_sumkx is not None else np.array([]),
        "P_RH_phi_odd_t_sumkx": P_RH_phi_odd_t_sumkx if P_RH_phi_odd_t_sumkx is not None else np.array([]),
        "P_RH_apar_even_t_sumkx": P_RH_apar_even_t_sumkx if P_RH_apar_even_t_sumkx is not None else np.array([]),
        "P_RH_apar_odd_t_sumkx": P_RH_apar_odd_t_sumkx if P_RH_apar_odd_t_sumkx is not None else np.array([]),
        "P_RH_bpar_even_t_sumkx": P_RH_bpar_even_t_sumkx if P_RH_bpar_even_t_sumkx is not None else np.array([]),
        "P_RH_bpar_odd_t_sumkx": P_RH_bpar_odd_t_sumkx if P_RH_bpar_odd_t_sumkx is not None else np.array([]),
        "P_RH_coll_t_sumkx": P_RH_coll_t_sumkx if P_RH_coll_t_sumkx is not None else np.array([]),
        "has_hyper": D_hyper is not None,
    }
    if D_hyper is not None:
        result["P_RH_hyper_mean_kx"] = P_RH_hyper_mean_kx
        result["P_RH_hyper_t_sumkx"] = P_RH_hyper_t_sumkx
    return result


def _moving_average(f_t, t, dt, dt_val=20):
    f_avg_t = np.zeros_like(f_t)
    for i_t, t_val in enumerate(t):
        idx_min = np.argmin(np.abs(t - (t_val - dt_val / 2)))
        idx_max = min(np.argmin(np.abs(t - (t_val + dt_val / 2))) + 1, len(t))
        f_avg_t[i_t] = dt_weighted_mean(f_t[idx_min:idx_max], weights=dt[idx_min:idx_max])
    return f_avg_t


def plot_RH_per_kx_summary_vs_time(run, ylim_P_RH=None, linthresh=1e-4, fig=None, axs=None, combine_fields=False, combine_even_odd=False, **kwargs):
    """"Summed over kx" E_RH(t)/P_RH(t) figure. **kwargs forwarded to
    get_RH_per_kx_means (time_min/time_max/kx_max/passing_trapped/
    fphi/fapar/fbpar/fcoll/D_hyper).

    combine_fields/combine_even_odd: same two toggles as plot_P_RH (see
    its docstring), applied here to the kx-summed phi/apar/bpar
    contributions. coll is always shown as a single combined-parity
    line here (matching this figure's pre-existing convention), unlike
    plot_P_RH where it also respects combine_even_odd.
    """
    means = get_RH_per_kx_means(run, **kwargs)
    t = means["t"]
    dt = dt_weights(t)

    if axs is None:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 16))

    axs[0].plot(t, means["E_RH_t_sumkx"], c="k", alpha=0.5)
    axs[0].set_ylabel(r"$\sum_{k_x} E_\mathrm{RH}$")
    E_RH_sum_kx_mean = dt_weighted_mean(means["E_RH_t_sumkx"], weights=dt)
    axs[0].axhline(E_RH_sum_kx_mean, ls="--", c="k", alpha=0.5)

    if combine_fields:
        field_entries = [(
            "slateblue",
            means["P_RH_phi_even_t_sumkx"] + means["P_RH_apar_even_t_sumkx"] + means["P_RH_bpar_even_t_sumkx"],
            means["P_RH_phi_odd_t_sumkx"] + means["P_RH_apar_odd_t_sumkx"] + means["P_RH_bpar_odd_t_sumkx"],
            r"$\varphi{+}A_\parallel{+}B_\parallel$",
        )]
    else:
        field_entries = [
            ("mediumblue",  means["P_RH_phi_even_t_sumkx"],  means["P_RH_phi_odd_t_sumkx"],  r"$\varphi$"),
            ("crimson",     means["P_RH_apar_even_t_sumkx"], means["P_RH_apar_odd_t_sumkx"], r"$A_\parallel$"),
            ("forestgreen", means["P_RH_bpar_even_t_sumkx"], means["P_RH_bpar_odd_t_sumkx"],  r"$B_\parallel$"),
        ]

    axs[1].plot(t, means["P_RH_coll_t_sumkx"], c="orange")
    P_RH_t_sumkx = means["P_RH_coll_t_sumkx"]
    P_RH_coll_sum_kx_mean = dt_weighted_mean(means["P_RH_coll_t_sumkx"], weights=dt)
    P_RH_sum_kx_mean = P_RH_coll_sum_kx_mean

    for color, even_t, odd_t, math_label in field_entries:
        tot_t = even_t + odd_t
        P_RH_t_sumkx = P_RH_t_sumkx + tot_t
        if combine_even_odd:
            mean_tot = dt_weighted_mean(tot_t, weights=dt)
            P_RH_sum_kx_mean += mean_tot
            axs[1].plot(t, tot_t, c=color)
            axs[1].axhline(mean_tot, c=color, label=math_label)
        else:
            mean_even = dt_weighted_mean(even_t, weights=dt)
            mean_odd  = dt_weighted_mean(odd_t, weights=dt)
            P_RH_sum_kx_mean += mean_even + mean_odd
            axs[1].plot(t, even_t, c=color, ls='-')
            axs[1].plot(t, odd_t,  c=color, ls='--')
            axs[1].axhline(mean_even, c=color, ls='-',  label=math_label + " Even")
            axs[1].axhline(mean_odd,  c=color, ls='--', label=math_label + " Odd")

    if means["has_hyper"]:
        axs[1].plot(t, means["P_RH_hyper_t_sumkx"], c="purple")
        P_RH_t_sumkx = P_RH_t_sumkx + means["P_RH_hyper_t_sumkx"]
        P_RH_hyper_sum_kx_mean = dt_weighted_mean(means["P_RH_hyper_t_sumkx"], weights=dt)
        P_RH_sum_kx_mean += P_RH_hyper_sum_kx_mean
        axs[1].axhline(P_RH_hyper_sum_kx_mean, c="purple", label=r"Hyper")

    axs[1].plot(t, P_RH_t_sumkx, c="k", alpha=0.5)
    axs[1].set_ylabel(r"$\sum_{k_x} P_\mathrm{RH}$")

    P_RH_sum_kx_num = np.gradient(means["E_RH_t_sumkx"], t)

    try:
        axs[1].plot(t, _moving_average(P_RH_sum_kx_num, t, dt), c="c", ls="--", alpha=0.5)
    except Exception as e:
        print(e)

    axs[1].plot(t, P_RH_sum_kx_num, c="c", ls="--", alpha=0.1)
    P_RH_sum_kx_num_mean = dt_weighted_mean(P_RH_sum_kx_num, weights=dt)

    axs[1].axhline(P_RH_coll_sum_kx_mean, c="orange", label=r"Coll.")
    axs[1].axhline(P_RH_sum_kx_mean, c="k", label=r"Total", alpha=0.5)
    axs[1].axhline(P_RH_sum_kx_num_mean, c="c", label=r"Num", alpha=0.5)

    axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    axs[1].set_xlabel(r"$ t %s/a$" % get_vt_label(run.ncdata))
    for ax in axs:
        ax.grid(True)
    if ylim_P_RH is not None:
        axs[1].set_ylim(ylim_P_RH)
    axs[0].set_ylim(ymin=0)
    axs[1].set_yscale("symlog", linthresh=linthresh)

    return fig, axs


def plot_RH_per_kx_summary_vs_kx(run, fcoll=1, D_hyper=None, fig=None, axs=None, combine_fields=False, combine_even_odd=False, **kwargs):
    """"vs kx" time-averaged E_RH/P_RH figure. **kwargs forwarded to
    get_RH_per_kx_means (time_min/time_max/kx_max/passing_trapped/
    fphi/fapar/fbpar/D_hyper is also passed explicitly here since it
    changes which panels are drawn).

    combine_fields/combine_even_odd: same two toggles as plot_P_RH (see
    its docstring), applied here to phi/apar/bpar vs kx. coll is always
    shown as a single combined-parity line here (matching this figure's
    pre-existing convention), unlike plot_P_RH where it also respects
    combine_even_odd.
    """
    means = get_RH_per_kx_means(run, fcoll=fcoll, D_hyper=D_hyper, **kwargs)
    kx_all = means["kx_all"]

    P_RH_mean_kx = means["P_RH_even_mean_kx"] + means["P_RH_odd_mean_kx"]
    P_RH_coll_mean_kx = means["P_RH_coll_even_mean_kx"] + means["P_RH_coll_odd_mean_kx"]
    if means["has_hyper"]:
        P_RH_mean_kx = P_RH_mean_kx + means["P_RH_hyper_mean_kx"]

    if combine_fields:
        field_entries = [(
            "slateblue",
            means["P_RH_phi_even_mean_kx"] + means["P_RH_apar_even_mean_kx"] + means["P_RH_bpar_even_mean_kx"],
            means["P_RH_phi_odd_mean_kx"] + means["P_RH_apar_odd_mean_kx"] + means["P_RH_bpar_odd_mean_kx"],
            r"$\varphi{+}A_\parallel{+}B_\parallel$",
        )]
    else:
        field_entries = [
            ("mediumblue",  means["P_RH_phi_even_mean_kx"],  means["P_RH_phi_odd_mean_kx"],  r"$\varphi$"),
            ("crimson",     means["P_RH_apar_even_mean_kx"], means["P_RH_apar_odd_mean_kx"], r"$A_\parallel$"),
            ("forestgreen", means["P_RH_bpar_even_mean_kx"], means["P_RH_bpar_odd_mean_kx"],  r"$B_\parallel$"),
        ]

    if axs is None:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(11, 18))

    axs[0].loglog(kx_all[kx_all > 0], means["E_RH_mean_kx"][kx_all > 0], marker=".", c="k", alpha=0.5)
    axs[0].set_ylabel(r"$\langle E_\mathrm{RH} \rangle_t$")

    kx_plot = kx_all[kx_all > 0]
    axs[0].loglog(kx_plot, 5e-4 * kx_plot ** (-5 / 2), ls="--", c="0.5", lw=2)
    axs[0].loglog(kx_plot, 5e-4 * kx_plot ** (-3), ls="--", c="0.5", lw=2)
    axs[0].loglog(kx_plot, 5e-4 * kx_plot ** (-7 / 2), ls="--", c="0.5", lw=2)

    lw = 2
    for i_norm, norm in enumerate([1, 1 / means["E_RH_mean_kx"]]):
        ax = axs[1 + i_norm]

        for color, even_kx, odd_kx, math_label in field_entries:
            if combine_even_odd:
                ax.semilogx(kx_all, norm * (even_kx + odd_kx), lw=lw, label=math_label, marker=".", c=color)
            else:
                ax.semilogx(kx_all, norm * even_kx, lw=lw, label=math_label + " Even", marker=".", c=color, ls='-')
                ax.semilogx(kx_all, norm * odd_kx,  lw=lw, label=math_label + " Odd",  marker=".", c=color, ls='--')

        if fcoll != 0:
            ax.semilogx(kx_all, norm * P_RH_coll_mean_kx, lw=lw, label=r"$P_{\mathrm{RH}}^{C}$", marker=".", c="orange")
        if means["has_hyper"]:
            ax.semilogx(kx_all, norm * means["P_RH_hyper_mean_kx"], marker=".", c="purple", label=r"Hyper")
        ax.semilogx(kx_all, norm * P_RH_mean_kx, lw=lw, label=r"$P_{\mathrm{RH}}^\mathrm{NL}$", marker=".", c="k")
        ax.semilogx(kx_all[kx_all > 0], (norm * means["P_RH_num_mean_kx"])[kx_all > 0], lw=lw, label=r"$\mathrm{d}E_\mathrm{RH}/\mathrm{d}t$", marker=".", c="0.5")
        ax.set_xlabel(r"$k_x %s$" % get_rho_label(run.ncdata))
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

    axs[1].set_ylabel(r"$\langle P_\mathrm{RH}\rangle_t$")
    axs[2].set_ylabel(r"$\langle P_\mathrm{RH}\rangle_t/\langle E_\mathrm{RH}\rangle_t$")
    axs[2].set_yscale("symlog", linthresh=1e-3)

    for ax in axs:
        ax.set_xlim(xmin=0.5 * (kx_all[1] - kx_all[0]))
        ax.grid(True)

    return fig, axs
