"""Qflx(t)/E_phi(t)/E_upar(t) comparison across multiple runs.

Extracted from example_plots/plot_flux_time.py's inline loop -- the
per-run numerics (get_fluxes_over_time + a per-kx Gamma0 evaluation for
E_phi, plus E_upar) are unchanged from the original, just moved out of a
script that also hardcoded which directories to compare (see
stella_diagnostics.scan.config for how the two are now kept separate).
"""

import numpy as np
import scipy.special as specialfunc

from stella_diagnostics.io.cache import cached
from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import get_or_create_ax


@cached(version=1)
def _get_flux_energy_vs_time(run, species_idx=0, Q_div=10, skip_phi2=False, plot_ratio=False, epsilon=0.18, qinp=1.4):
    """(time, qflx/Q_div, E_phi(t) or None, E_upar(t) or None) for one run.

    If plot_ratio, E_phi/E_upar are replaced by the single combined ratio
    (epsilon/qinp)**2 * E_upar/E_phi, returned in the E_phi slot (E_upar is
    None), matching plot_flux_time.py's original plot_ratio branch.
    skip_phi2=True skips the phi/upar energy computation entirely (both
    returned as None) -- this is the expensive part of the computation, so
    callers that only need Qflx should pass it.
    """
    time = run.get_time_array()
    _, _, qflx, flux_time = run.get_fluxes_over_time(species_idx=species_idx, norm=False)
    qflx = qflx / Q_div

    if skip_phi2:
        return flux_time, qflx, None, None

    dl_over_B_avg = run.dl_over_B_avg()

    # phi_vs_t(t, tube, zed, kx, ky, ri)
    phiZ = run.ncdata.variables["phi_vs_t"][:, 0, :, :, 0, :]
    phiZ_C = phiZ[:, :, :, 0] + 1j * phiZ[:, :, :, 1]

    zed = run.ncdata.variables["zed"][:]
    kx = run.ncdata.variables["kx"][:]
    # Evaluate 1-Gamma0 (single species!) across the full kx array at ky=0,
    # zed-resolved -- distinct from quantities.registry.get_Gamma0(), which
    # evaluates at a single (kx_idx, ky_idx) point, not the full kx array.
    Gamma0_vals = np.zeros((len(zed), len(kx)))
    shat = run.ncdata.variables["shat"].getValue()
    gds22 = run.ncdata.variables["gds22"][:, 0] / shat**2  # |nabla(x)|^2
    bmag = run.ncdata.variables["bmag"][:, 0]
    for i_kx, kx_val in enumerate(kx):
        kperp2 = (kx_val / bmag) ** 2 * gds22
        Gamma0_vals[:, i_kx] = specialfunc.iv(0, kperp2 / 2) * np.exp(-kperp2 / 2)
    E_phi = np.sum((1 - Gamma0_vals[None, :, :]) * np.abs(phiZ_C) ** 2 * dl_over_B_avg[None, :, None], axis=(1, 2))

    # upar(t, species, tube, zed, kx, ky, ri)
    uparZ = run.ncdata.variables["upar"][:, species_idx, 0, :, :, 0, :]
    uparZ_C = uparZ[:, :, :, 0] + 1j * uparZ[:, :, :, 1]
    E_upar = np.sum(0.5 * np.abs(uparZ_C) ** 2 * dl_over_B_avg[None, :, None], axis=(1, 2))

    if plot_ratio:
        return time, qflx, (epsilon / qinp) ** 2 * E_upar / E_phi, None

    return time, qflx, E_phi, E_upar


def plot_qflx_and_energy_vs_time(
    dirnames,
    labels=None,
    colors=None,
    filename="CBC",
    code="stella",
    Q_div=10,
    skip_phi2=False,
    plot_ratio=False,
    fig=None,
    ax=None,
):
    """Qflx(t) (and, unless skip_phi2, E_phi(t)/E_upar(t) or their ratio)
    for each of `dirnames`, overlaid on one Axes.

    One StellaRun per dirname; a run that fails to load is skipped (logged,
    not raised), matching the original script's behavior. Does not set
    axis limits or save the figure -- those are per-comparison presentation
    choices that belong in the driver/config, not in this analysis
    function. Returns (fig, ax).
    """
    fig, ax = get_or_create_ax(fig=fig, ax=ax, figsize=(12, 9))

    labels = labels if labels is not None else [None] * len(dirnames)
    colors = colors if colors is not None else [None] * len(dirnames)

    for i_dir, dirname in enumerate(dirnames):
        try:
            run = StellaRun(dirname + "/" + filename, code=code)
        except Exception as e:
            print(f"Couldn't load {dirname}: {e!r}")
            continue

        label = labels[i_dir] if i_dir < len(labels) else None
        color = colors[i_dir] if i_dir < len(colors) else None

        time, qflx, e1, e2 = _get_flux_energy_vs_time(
            run, Q_div=Q_div, skip_phi2=skip_phi2, plot_ratio=plot_ratio
        )
        ax.plot(time, qflx, label=label, c=color, lw=2)

        if skip_phi2:
            continue

        if plot_ratio:
            label_ratio = (label + r" $( \epsilon^2/q^2 E_{u_\parallel}/E_\varphi)$") if label is not None else None
            ax.plot(time, e1, label=label_ratio, c=color, ls="-.")
        else:
            label_phi = (label + r" $(E_{\varphi})$") if label is not None else None
            label_upar = (label + r" $(E_{u_\parallel})$") if label is not None else None
            ax.plot(time, e1, label=label_phi, c=color, ls="-.")
            ax.plot(time, e2, label=label_upar, c=color, ls="--", alpha=0.5)

    ax.legend(fontsize=6, ncol=min(len([l for l in labels if l is not None]), 6) or 1)
    ax.grid()
    if Q_div != 1:
        ax.set_ylabel(r"$Q/ %.1f Q_\mathrm{gB}$" % (Q_div))
    else:
        ax.set_ylabel(r"$Q/Q_\mathrm{gB}$")
    ax.set_xlabel(r"$t v_T/a$")
    ax.set_yscale("log")

    return fig, ax
