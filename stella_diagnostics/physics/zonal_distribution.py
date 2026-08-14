"""Velocity-space moments and free-energy decomposition of the zonal
(ky=0) part of the stella distribution function g(vpa, mu), read directly
from restart files via stella_diagnostics.io.restart.

Extracted from example_plots/plot_zonal_distribution.py.
"""

import numpy as np

from stella_diagnostics.io.restart import RestartReader

# --------------------------------------------------------------------------
# Zonal phase shift
# --------------------------------------------------------------------------
# Each Fourier coefficient g(kx) can be given its own phase without changing
# the physics of that mode; the only constraint is g(-kx) = conj(g(kx)) (the
# reality of the ky=0 field in x), which is exactly preserved as long as
# phase(-kx) = conj(phase(kx)). The turbulent zonal harmonics in a nonlinear
# snapshot generally do NOT share one coherent spatial phase (checked
# directly against this run's data: the x0 that would maximize a single kx
# mode's own <u_par*cos(theta)> varies widely from mode to mode), so instead
# of a single rigid shift x -> x - x0, each kx mode gets its own phase,
# chosen so that its own theta-average of (parallel flow * cos(theta)) is
# real and maximal.


def compute_moments_vs_kx_theta(restart_dir, stem, layout, blocksize, grids):
    """Density, parallel-flow, and temperature Fourier coefficients at ky=0
    for every (kx, theta) grid point, from velocity-space integrals of g
    (normalized velocity units; T = <(vpa^2+2*mu*Bmin)g> - 1.5*<g>), plus the
    free energy at each (kx, theta), split into the part carried by a local
    fluid-moment ("Maxwellian") reconstruction of g and the remainder
    ("kinetic"), each further split into vpa-even and vpa-odd pieces (see
    free_energy_pieces below). Each returned array has shape (nakx, nzed)."""
    reader = RestartReader(restart_dir, stem)
    akx, zed, nzgrid = grids["akx"], grids["zed"], grids["nzgrid"]
    mu, vpa = grids["mu"], grids["vpa"]
    mu_w, vpa_w = grids["mu_weights"], grids["vpa_weights"]
    vpa_moment_weights = vpa_w * vpa
    vpa_energy_w, mu_energy_w = grids["vpa_energy_weights"], grids["mu_energy_weights"]
    epsilon, temp = grids["epsilon"], grids["temp"]
    mu_axis = mu * grids["bmin"] / temp

    shape = (len(akx), len(zed))
    density = np.empty(shape, dtype=complex)
    upar = np.empty(shape, dtype=complex)
    temperature = np.empty(shape, dtype=complex)
    fe_M_even = np.empty(shape)
    fe_M_odd = np.empty(shape)
    fe_K_even = np.empty(shape)
    fe_K_odd = np.empty(shape)

    for k, iz in enumerate(range(-nzgrid, nzgrid + 1)):
        theta = zed[k]
        b_theta = local_bfield(theta, epsilon)
        # Local (theta-consistent) energy weight, as opposed to grids'
        # Bmin-based mu_energy_w: needed so the fluid reconstruction below
        # uses the same velocity-space shape as the local F0 it multiplies.
        mu_energy_w_local = mu_w * 2.0 * mu * b_theta
        f0 = maxwellian_factor(vpa, mu_axis, theta, epsilon, temp)  # (nmu, nvpa)
        shape_even = vpa[np.newaxis, :] ** 2 + 2 * b_theta * mu[:, np.newaxis] - 1.5

        for i in range(len(akx)):
            ikx = i + 1
            world_idx = layout.world_index(iky=1, ikx=ikx, iz=iz, it=1, is_=1)
            iproc = world_idx // blocksize
            local_glo = world_idx - iproc * blocksize
            gr, gi = reader.read_g(iproc, local_glo)  # (mu, vpa)
            g = gr + 1j * gi
            dens = mu_w @ g @ vpa_w
            u = mu_w @ g @ vpa_moment_weights
            energy = mu_w @ g @ vpa_energy_w + mu_energy_w @ g @ vpa_w
            density[i, k] = dens
            upar[i, k] = u
            temperature[i, k] = energy - 1.5 * dens

            energy_local = mu_w @ g @ vpa_energy_w + mu_energy_w_local @ g @ vpa_w
            dtemp_local = energy_local - 1.5 * dens

            g_even = 0.5 * (g + g[:, ::-1])
            g_odd = 0.5 * (g - g[:, ::-1])
            g_maxwellian_even = f0 * (dens + shape_even * dtemp_local)
            g_maxwellian_odd = f0 * (2.0 * vpa[np.newaxis, :] * u)

            def free_energy(x):
                return 0.5 * temp * (mu_w @ (np.abs(x) ** 2 / f0) @ vpa_w)

            fe_M_even[i, k] = free_energy(g_maxwellian_even)
            fe_M_odd[i, k] = free_energy(g_maxwellian_odd)
            fe_K_even[i, k] = free_energy(g_even - g_maxwellian_even)
            fe_K_odd[i, k] = free_energy(g_odd - g_maxwellian_odd)
    reader.close()
    return dict(
        density=density, upar=upar, temperature=temperature,
        fe_M_even=fe_M_even, fe_M_odd=fe_M_odd,
        fe_K_even=fe_K_even, fe_K_odd=fe_K_odd,
    )


def compute_per_kx_alignment_phase(zed, upar):
    """For each kx independently, the unit phase exp(-i*arg(q(kx))) that
    rotates q(kx) = <u_par(kx,theta)*cos(theta)>_theta onto the positive
    real axis (real and maximal). zed is a periodic grid (uniform, endpoints
    coincide), so a plain mean over its points is the correct
    periodic-trapezoidal average. Reality (phase(-kx)=conj(phase(kx))) holds
    automatically since q(-kx)=conj(q(kx))."""
    q_kx = (upar @ np.cos(zed)) / len(zed)
    amp = np.abs(q_kx)
    phase = np.ones_like(q_kx)
    nonzero = amp > 0
    phase[nonzero] = np.conj(q_kx[nonzero]) / amp[nonzero]
    return phase


def local_bfield(theta, epsilon):
    """B(theta)/B0 in the large-aspect-ratio circular approximation used
    throughout this module, with theta=0 at the outboard (Bmin) side."""
    return 1.0 / (1.0 + epsilon * np.cos(theta))


def trapped_passing_boundary(mu_axis_max, theta, epsilon, temp, n=200):
    """Trapped/passing boundary vpa(mu) at a given theta, in the same units
    as the plotted vpa and mu axes (mu_axis = mu*Bmin/T): a particle is
    marginally trapped when its energy (1/2)vpa^2 + mu*B(theta) equals
    mu*Bmax, using the large-aspect-ratio approximation B(theta) =
    1/(1+epsilon*cos(theta)) with theta=0 at the outboard side (confirmed
    against stella's Miller geometry, where theta=0 gives the largest major
    radius R, i.e. the weakest field / Bmin)."""
    bmin = 1.0 / (1.0 + epsilon)
    bmax = 1.0 / (1.0 - epsilon)
    b_theta = local_bfield(theta, epsilon)
    mu_fine = np.linspace(0.0, mu_axis_max, n)
    mu = mu_fine * temp / bmin  # invert mu_axis = mu*bmin/temp
    vpa_sq = 2 * mu * (bmax - b_theta)
    return mu_fine, np.sqrt(np.maximum(vpa_sq, 0.0))


def maxwellian_factor(vpa, mu_axis, theta, epsilon, temp):
    """F0(vpa, mu) at a given theta, matching stella's own normalization
    exactly (grids_velocity.f90: maxwell_vpa=exp(-vpa^2), maxwell_mu=
    exp(-2*mu*bmag(theta)*temp_psi0/temp), for a single non-radially-varying
    species), rewritten in terms of the plotted mu_axis = mu*Bmin/T and the
    same large-aspect-ratio B(theta) approximation used elsewhere. Shape
    (nmu, nvpa), matching the plotted (gr, gi) arrays."""
    bmin = 1.0 / (1.0 + epsilon)
    b_theta = local_bfield(theta, epsilon)
    exponent = vpa[np.newaxis, :] ** 2 + 2 * temp * (b_theta / bmin) * mu_axis[:, np.newaxis]
    return np.exp(-exponent)
