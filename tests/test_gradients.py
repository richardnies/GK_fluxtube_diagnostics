"""Data-free test for the shared analytic critical-gradient formula."""
from stella_diagnostics.physics.gradients import get_aLT_lin_analytic


def test_get_aLT_lin_analytic_matches_original_formula():
    rhoc, q, shat = 0.18, 1.4, 0.8
    expected = (1 + 1) * (1.33 + 1.91 * shat / q) * (1 - 1.5 * 1 * rhoc) * 1
    assert get_aLT_lin_analytic(rhoc=rhoc, q=q, shat=shat) == expected


def test_get_aLT_lin_analytic_respects_eps_tau_overrides():
    rhoc, q, shat, eps, tau = 0.045, 2.8, 0.32, 0.5, 1.0
    expected = (1 + tau) * (1.33 + 1.91 * shat / q) * (1 - 1.5 * eps * rhoc) * eps
    assert get_aLT_lin_analytic(rhoc=rhoc, q=q, shat=shat, eps=eps, tau=tau) == expected
