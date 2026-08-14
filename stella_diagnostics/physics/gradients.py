"""Analytic linear critical-gradient threshold formulas.

Extracted from three byte-identical copies previously duplicated across
example_plots/plot_flux_coll.py, plot_ERH_Ephi.py, and
plot_param_scan_Dimits.py.
"""


def get_aLT_lin_analytic(rhoc, q, shat, eps=1, tau=1):
    """Jenko-type linear ITG critical-gradient threshold R/L_T(rhoc, q, shat, eps, tau).

    NOTE: every existing call site in the three originals this was extracted
    from passes only rhoc/q/shat, leaving eps/tau at these defaults (eps=1,
    tau=1) even where each run's actual eps is available elsewhere in the
    same script -- this changes the numeric threshold reported. Preserved
    exactly as called originally; not silently corrected, since it would
    change plotted numbers. Flagged here rather than fixed, per this
    project's convention for pre-existing numeric/physics-adjacent choices.
    """
    return (1 + tau) * (1.33 + 1.91 * shat / q) * (1 - 1.5 * eps * rhoc) * eps
