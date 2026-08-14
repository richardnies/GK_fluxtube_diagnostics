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
from stella_diagnostics.io.codes import get_nspecies, get_species_name, get_vt_label
from stella_diagnostics.plotting.mpl_helpers import get_or_create_ax


@cached(version=1)
def _get_flux_energy_vs_time(run, species_idx=0, Q_div=10, skip_phi2=False, plot_ratio=False, epsilon=0.18, qinp=1.4):
    """(time, qflx/Q_div, E_phi(t) or None, E_upar(t) or None) for one run.

    If plot_ratio, E_phi/E_upar are replaced by the single combined ratio
    (epsilon/qinp)**2 * E_upar/E_phi, returned in the E_phi slot (E_upar is
    None), matching plot_flux_time.py's original plot_ratio branch.
    epsilon (inverse aspect ratio r/R) and qinp (safety factor q) default
    to the Cyclone Base Case's own values (0.18, 1.4 -- this package's
    default `filename="CBC"` run everywhere else), not this particular
    run's actual geometry (available as run.aspect_ratio/run.safety_factor
    if a run-specific ratio is ever needed instead).
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
    species_idx=None,
    fig=None,
    ax=None,
):
    """Qflx(t) (and, unless skip_phi2, E_phi(t)/E_upar(t) or their ratio)
    for each of `dirnames`, overlaid on one Axes.

    species_idx=None (default) plots every species in each run -- pass an
    int to restrict to just one (the previous, single-species-only default
    was species_idx=0), matching the species_idx=None generalization
    physics.flux_plots.plot_flux_over_time already applies to fig_fluxes.
    Qflx and E_upar are genuinely species-dependent and get one line per
    species, labeled with that species' readable name (see
    io.codes.get_species_name); E_phi is a field-only quantity (no species
    dependence) and is only plotted once per run regardless of how many
    species it has. All lines from the same run share that run's color
    (from `colors`, or matplotlib's auto-cycle if `colors` is None) --
    within a run, multiple species are told apart by their legend label
    and by progressively lower alpha for each species after the first;
    quantity type (Qflx solid / E_phi dash-dot / E_upar dashed) is
    unchanged. The time axis is labeled with the subscript of the first
    successfully loaded run's reference species (see
    io.codes.get_reference_species_idx) -- comparisons across runs with
    inconsistent species ordering would get a technically-mislabeled
    subscript on this shared axis, but the underlying values are unaffected.

    One StellaRun per dirname; a run that fails to load is skipped (logged,
    not raised), matching the original script's behavior. Does not set
    axis limits or save the figure -- those are per-comparison presentation
    choices that belong in the driver/config, not in this analysis
    function. Returns (fig, ax).
    """
    fig, ax = get_or_create_ax(fig=fig, ax=ax, figsize=(12, 9))

    labels = labels if labels is not None else [None] * len(dirnames)
    colors = colors if colors is not None else [None] * len(dirnames)

    ref_vt_label = None
    n_legend_entries = 0

    for i_dir, dirname in enumerate(dirnames):
        try:
            run = StellaRun(dirname + "/" + filename, code=code)
        except Exception as e:
            print(f"Couldn't load {dirname}: {e!r}")
            continue

        if ref_vt_label is None:
            ref_vt_label = get_vt_label(run.ncdata)

        label = labels[i_dir] if i_dir < len(labels) else None
        color = colors[i_dir] if i_dir < len(colors) else None
        prefix = (label + " ") if label else ""

        species_idxs = [species_idx] if species_idx is not None else list(range(get_nspecies(run.ncdata)))

        for i_sp, sp_idx in enumerate(species_idxs):
            sp_name = get_species_name(run.ncdata, sp_idx)
            sp_alpha = max(1 - 0.3 * i_sp, 0.3)

            time, qflx, e1, e2 = _get_flux_energy_vs_time(
                run, species_idx=sp_idx, Q_div=Q_div, skip_phi2=skip_phi2, plot_ratio=plot_ratio
            )
            ax.plot(time, qflx, label=prefix + sp_name, c=color, lw=2, alpha=sp_alpha)
            n_legend_entries += 1

            if skip_phi2:
                continue

            if plot_ratio:
                label_ratio = prefix + sp_name + r" $(\epsilon^2/q^2 E_{u_\parallel}/E_\varphi)$"
                ax.plot(time, e1, label=label_ratio, c=color, ls="-.", alpha=sp_alpha)
                n_legend_entries += 1
            else:
                if i_sp == 0:
                    label_phi = prefix + r"$(E_{\varphi})$"
                    ax.plot(time, e1, label=label_phi, c=color, ls="-.")
                    n_legend_entries += 1
                label_upar = prefix + sp_name + r" $(E_{u_\parallel})$"
                ax.plot(time, e2, label=label_upar, c=color, ls="--", alpha=0.5 * sp_alpha)
                n_legend_entries += 1

    ax.legend(fontsize=6, ncol=min(n_legend_entries, 6) or 1)
    ax.grid()
    if Q_div != 1:
        ax.set_ylabel(r"$Q/ %.1f Q_\mathrm{gB}$" % (Q_div))
    else:
        ax.set_ylabel(r"$Q/Q_\mathrm{gB}$")
    ax.set_xlabel(r"$t %s/a$" % (ref_vt_label or "v_T"))
    ax.set_yscale("log")

    return fig, ax
