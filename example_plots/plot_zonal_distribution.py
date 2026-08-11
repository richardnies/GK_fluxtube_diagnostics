#!/usr/bin/env python3
"""
Plot the zonal (ky=0) part of the STELLA distribution function g(vpa, mu)
for user-chosen (kx, theta) combinations, reading directly from restart files.

g is split across one restart file per MPI processor via STELLA's kxkyz_lo
domain decomposition: each file holds a contiguous block of the flattened
(ky, kx, zed, tube, species) index, but the full (vpa, mu) velocity grid for
whatever indices it owns. This script reconstructs that decomposition (and
the physical kx/theta/vpa/mu grids) from the run's input namelist, so it can
find and read the right file/row for each requested (kx, theta) point.

No command-line options: edit the CONFIG block below and run
`python3 plot_zonal_distribution.py`. BASEDIR is searched recursively for
any directory containing a restart/ folder with restart files; one PDF is
produced per run found, as BASEDIR/fig_<run>_dist_fn_zonal.pdf.
"""
import glob
import os
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import netCDF4
import numpy as np
from scipy.special import roots_laguerre

if shutil.which("latex"):
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "serif"
else:
    print("Warning: no LaTeX installation found, falling back to mathtext.", file=sys.stderr)
matplotlib.rcParams["font.size"] = 13

VPA_LABEL = r"$v_\parallel / v_t$"
MU_LABEL = r"$\mu B_{\mathrm{min}} / T$"


# --------------------------------------------------------------------------
# Namelist parsing
# --------------------------------------------------------------------------

def parse_namelist(path):
    """Flat dict of key -> value (str/float/int/bool) for every `key = value`
    line inside any &group ... / block. Good enough for stella's simple
    (non-array) input files; keys we need don't collide across groups."""
    text = open(path).read()
    params = {}
    for line in text.splitlines():
        line = line.split("!", 1)[0].strip()
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?),?\s*$", line)
        if not m:
            continue
        key, raw = m.group(1).lower(), m.group(2).strip()
        if raw.lower() in (".true.", ".false."):
            val = raw.lower() == ".true."
        elif re.match(r"^['\"].*['\"]$", raw):
            val = raw[1:-1]
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw.replace("d", "e").replace("D", "E"))
                except ValueError:
                    val = raw
        params[key] = val
    return params


# --------------------------------------------------------------------------
# kxkyz_lo layout: index formulas for all 6 xyzs_layout orderings
# --------------------------------------------------------------------------
# Each entry gives the divisor chain (in units of naky/nakx/nzed/ntubes) used
# to peel iky, ikx, iz, it out of a flattened world index, matching stella's
# parallelisation_layouts.f90. Expressed here as the order in which the five
# axes are packed from fastest- to slowest-varying (species is always slowest).

_LAYOUT_ORDER = {
    "yxzs": ["iky", "ikx", "iz", "it"],
    "yzxs": ["iky", "iz", "it", "ikx"],
    "xyzs": ["ikx", "iky", "iz", "it"],
    "zyxs": ["iz", "it", "iky", "ikx"],
    "zxys": ["iz", "it", "ikx", "iky"],
    "xzys": ["ikx", "iz", "it", "iky"],
}


class KxkyzLayout:
    """Maps between (iky, ikx, iz, it, is) [1-based, iz signed] and the
    flattened 0-based world index used by stella's kxkyz_lo, for a given
    xyzs_layout ordering."""

    def __init__(self, naky, nakx, nzgrid, ntubes, nspec, xyzs_layout):
        self.naky, self.nakx, self.nzgrid = naky, nakx, nzgrid
        self.nzed = 2 * nzgrid + 1
        self.ntubes, self.nspec = ntubes, nspec
        if xyzs_layout not in _LAYOUT_ORDER:
            raise ValueError(f"Unknown xyzs_layout {xyzs_layout!r}")
        self.order = _LAYOUT_ORDER[xyzs_layout]
        self.sizes = {"iky": naky, "ikx": nakx, "iz": self.nzed, "it": ntubes}

    def world_index(self, iky, ikx, iz, it, is_):
        """0-based flattened index, species always slowest-varying."""
        components = {"iky": iky - 1, "ikx": ikx - 1, "iz": iz + self.nzgrid, "it": it - 1}
        idx = 0
        mult = 1
        for axis in self.order:
            idx += components[axis] * mult
            mult *= self.sizes[axis]
        idx += (is_ - 1) * mult
        return idx


# --------------------------------------------------------------------------
# Physical grids reconstructed from the input namelist
# --------------------------------------------------------------------------

def build_grids(params):
    grid_option = params.get("grid_option", "box")

    ny = params["ny"]
    nx = params["nx"]
    naky = (ny - 1) // 3 + 1
    nakx = 2 * ((nx - 1) // 3) + 1
    ikx_max = nakx // 2 + 1

    if grid_option == "range":
        akx_min, akx_max = params["akx_min"], params["akx_max"]
        naky = params.get("naky", naky)
        nakx = params.get("nakx", nakx)
        akx = np.linspace(akx_min, akx_max, nakx) if nakx > 1 else np.array([akx_min])
        dky = None
    else:
        y0 = params["y0"]
        jtwist = params["jtwist"]
        shat = params["shat"]
        dky = 1.0 / y0
        dkx = 2 * np.pi * shat * dky / jtwist
        ikx = np.arange(1, nakx + 1)
        akx = np.where(ikx <= ikx_max, (ikx - 1) * dkx, (ikx - 1 - nakx) * dkx)

    nzed_in = params["nzed"]
    nperiod = params.get("nperiod", 1)
    nzgrid = nzed_in // 2 + (nperiod - 1) * nzed_in
    iz = np.arange(-nzgrid, nzgrid + 1)
    zed = iz * np.pi / (nzed_in / 2)

    nvgrid = params["nvgrid"]
    nvpa = 2 * nvgrid
    vpa_max = params["vpa_max"]
    dvpa = 2.0 * vpa_max / (nvpa - 1)
    iv = np.arange(1, nvpa + 1)
    vpa = np.where(
        iv > nvgrid,
        (iv - nvgrid - 0.5) * dvpa,
        -((nvpa - iv + 1) - nvgrid - 0.5) * dvpa,
    )

    nmu = params["nmu"]
    x_gl, w_gl = roots_laguerre(nmu)
    # stella's namelist default is vperp_max=3.0 (namelist_velocity_grids.f90);
    # the sqrt(x_max) fallback only fires if a run explicitly sets a negative
    # vperp_max, not simply when it's absent from the input file.
    vperp_max = params.get("vperp_max", 3.0)
    if vperp_max is None or vperp_max < 0:
        vperp_max = np.sqrt(x_gl[-1])
    rhoc = params.get("rhoc", 0.0)
    rmaj = params.get("rmaj", 1.0)
    epsilon = rhoc / rmaj
    # B(theta) = 1/(1+epsilon*cos(theta)), theta=0 = outboard (confirmed against
    # stella's Miller geometry: R(r,theta)=R0+r*cos(theta+...), so theta=0 gives
    # the largest R i.e. weakest field). Bmin is therefore at theta=0.
    bmin = 1.0 / (1.0 + epsilon)
    mu_max = vperp_max ** 2 / (2.0 * bmin)
    mu = x_gl / x_gl[-1] * mu_max
    # Velocity-space integration weights, mirroring stella's own rescaling of
    # the Gauss-Laguerre quadrature (grids_velocity.f90); an overall constant
    # prefactor (2*bmag/sqrt(pi)) is dropped since it's common to every kx
    # mode and irrelevant to a phase/shift calculation.
    mu_weights = w_gl * np.exp(x_gl) / x_gl[-1] * mu_max
    vpa_weights = np.full(nvpa, dvpa)
    vpa_weights[0] *= 0.5
    vpa_weights[-1] *= 0.5
    # Energy-moment weights (parallel + perpendicular, using the Bmin
    # approximation above for the local field), for the temperature moment
    # T = <(vpa^2 + 2*mu*Bmin) g> - 1.5*<g> (normalized velocity units).
    vpa_energy_weights = vpa_weights * vpa ** 2
    mu_energy_weights = mu_weights * 2.0 * mu * bmin

    nspec = params.get("nspec", 1)
    ntubes = params.get("ntubes", 1)
    xyzs_layout = params.get("xyzs_layout", "yxzs")
    temp = params.get("temp", 1.0)

    return dict(
        naky=naky, nakx=nakx, akx=akx, nzgrid=nzgrid, zed=zed,
        nvpa=nvpa, vpa=vpa, vpa_weights=vpa_weights, vpa_energy_weights=vpa_energy_weights,
        nmu=nmu, mu=mu, mu_weights=mu_weights, mu_energy_weights=mu_energy_weights,
        bmin=bmin, epsilon=epsilon, temp=temp,
        nspec=nspec, ntubes=ntubes, xyzs_layout=xyzs_layout,
    )


# --------------------------------------------------------------------------
# Restart file discovery + data access
# --------------------------------------------------------------------------

def find_restart_stem(restart_dir):
    candidates = glob.glob(os.path.join(restart_dir, "*.nc.[0-9]*"))
    if not candidates:
        raise FileNotFoundError(f"No '*.nc.<n>' restart files found in {restart_dir}")
    stems = {re.sub(r"\.nc\.\d+$", "", os.path.basename(f)) for f in candidates}
    if len(stems) != 1:
        raise RuntimeError(f"Multiple restart-file stems found: {stems}")
    stem = stems.pop()
    nproc = len(candidates)
    return stem, nproc


def find_input_file(restart_dir):
    """The stella input file lives one directory above restart_dir (i.e. in
    the run directory that contains restart/ as a subfolder), not inside
    restart_dir itself."""
    parent_dir = os.path.dirname(os.path.normpath(restart_dir))
    in_files = glob.glob(os.path.join(parent_dir, "*.in"))
    if len(in_files) != 1:
        raise RuntimeError(
            f"Expected exactly one .in file in {parent_dir}, found {in_files}"
        )
    return in_files[0]


class RestartReader:
    def __init__(self, restart_dir, stem):
        self.restart_dir = restart_dir
        self.stem = stem
        self._handles = {}

    def _dataset(self, iproc):
        if iproc not in self._handles:
            path = os.path.join(self.restart_dir, f"{self.stem}.nc.{iproc}")
            ds = netCDF4.Dataset(path, "r")
            ds.set_auto_mask(False)  # plain ndarrays, not MaskedArray
            self._handles[iproc] = ds
        return self._handles[iproc]

    def read_g(self, iproc, local_glo):
        ds = self._dataset(iproc)
        gr = ds.variables["gr"][local_glo, :, :]  # (mu, vpa)
        gi = ds.variables["gi"][local_glo, :, :]
        return gr, gi

    def close(self):
        for ds in self._handles.values():
            ds.close()
        self._handles.clear()


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


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def local_bfield(theta, epsilon):
    """B(theta)/B0 in the large-aspect-ratio circular approximation used
    throughout this script, with theta=0 at the outboard (Bmin) side."""
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


def add_panel(ax, vpa, mu, data, title, cmap, boundary=None):
    vmax = np.max(np.abs(data))
    vmax = vmax if vmax > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    mesh = ax.pcolormesh(vpa, mu, data, shading="nearest", cmap=cmap, norm=norm)
    if boundary is not None:
        mu_fine, vpa_boundary = boundary
        ax.plot(vpa_boundary, mu_fine, "k--", lw=1.1, alpha=0.5, zorder=3)
        ax.plot(-vpa_boundary, mu_fine, "k--", lw=1.1, alpha=0.5, zorder=3)
    ax.set_title(title, fontsize=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.06)
    cb = plt.colorbar(mesh, cax=cax)
    cb.ax.tick_params(labelsize=7)


def add_free_energy_panel(ax, zed, fe, title):
    """Free energy W(theta) = (T/2)<|.|^2/F0>, split into the part carried
    by a local fluid (density/flow/temperature) reconstruction of g
    ("Maxwellian") vs the remainder ("kinetic"), each further split into
    vpa-even and vpa-odd pieces. All four pieces are real and non-negative
    by construction; plotted on a log scale since free energy densities
    routinely span several orders of magnitude along the field line."""
    x = zed / np.pi
    floor = max(
        1e-300,
        1e-12 * max(np.max(fe["fe_M_even"]), np.max(fe["fe_M_odd"]),
                     np.max(fe["fe_K_even"]), np.max(fe["fe_K_odd"])),
    )
    ax.plot(x, np.maximum(fe["fe_M_even"], floor), color="tab:blue", ls="-",
             label=r"$W_M$, even")
    ax.plot(x, np.maximum(fe["fe_M_odd"], floor), color="tab:blue", ls="--",
             label=r"$W_M$, odd")
    ax.plot(x, np.maximum(fe["fe_K_even"], floor), color="tab:green", ls="-",
             label=r"$W_K$, even")
    ax.plot(x, np.maximum(fe["fe_K_odd"], floor), color="tab:green", ls="--",
             label=r"$W_K$, odd")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$\theta/\pi$")
    ax.set_ylabel(r"$W$")


def add_1d_panel(ax, zed, upar, temperature, title):
    x = zed / np.pi
    ax.axhline(0, color="0.75", lw=0.8, zorder=0)
    ax.plot(x, upar.real, color="tab:blue", ls="-", label=r"$\mathrm{Re}(u_\parallel)$")
    ax.plot(x, upar.imag, color="tab:blue", ls="--", label=r"$\mathrm{Im}(u_\parallel)$")
    ax.plot(x, temperature.real, color="tab:red", ls="-", label=r"$\mathrm{Re}(T)$")
    ax.plot(x, temperature.imag, color="tab:red", ls="--", label=r"$\mathrm{Im}(T)$")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$\theta/\pi$")


def make_combined_page(pdf, grids, kx_values, theta_values, matched_kx, matched_theta,
                        re_grid, im_grid, moment_theta, cmap, quantity_label=r"g^{Z}"):
    """One page: for each kx row, a 1D column (parallel flow & temperature
    vs theta) followed by, for each theta, a tightly-spaced Re/Im heatmap
    pair (with normal spacing between different theta groups)."""
    nrows, ntheta = len(kx_values), len(theta_values)
    vpa = grids["vpa"]
    mu_axis = grids["mu"] * grids["bmin"] / grids["temp"]
    boundaries = [
        trapped_passing_boundary(mu_axis.max(), theta, grids["epsilon"], grids["temp"])
        for theta in matched_theta
    ]

    # Fixed per-column/per-row sizes (inches), used directly as the
    # GridSpec width_ratios too, so each theta-group panel and each kx row
    # keeps the same absolute size regardless of how many there are: total
    # figure width grows linearly with ntheta, height linearly with nrows.
    fe_col_width = 3.6      # free-energy column (leftmost)
    first_col_width = 3.6   # 1D flow/temperature column
    theta_group_width = 5.0  # one Re+Im heatmap pair (with its colorbars)
    row_height = 2.6
    width_ratios = [fe_col_width, first_col_width] + [theta_group_width] * ntheta
    fig = plt.figure(figsize=(sum(width_ratios), row_height * nrows))
    outer_gs = fig.add_gridspec(nrows, 2 + ntheta, width_ratios=width_ratios,
                                 wspace=0.55, hspace=0.55)

    for i in range(nrows):
        row_title = rf"$k_x\rho={matched_kx[i]:.2f}$"

        ax_fe = fig.add_subplot(outer_gs[i, 0])
        add_free_energy_panel(ax_fe, grids["zed"], moment_theta[i], row_title)
        if i == 0:
            ax_fe.legend(fontsize=7, loc="best", framealpha=0.9)

        ax1d = fig.add_subplot(outer_gs[i, 1])
        add_1d_panel(ax1d, grids["zed"], moment_theta[i]["upar"],
                     moment_theta[i]["temperature"], row_title)
        if i == 0:
            ax1d.legend(fontsize=7, loc="best", framealpha=0.9)

        for j in range(ntheta):
            inner_gs = outer_gs[i, j + 2].subgridspec(1, 2, wspace=0.5)
            ax_re = fig.add_subplot(inner_gs[0, 0])
            ax_im = fig.add_subplot(inner_gs[0, 1])
            theta_label = rf"\theta/\pi={matched_theta[j] / np.pi:.2f}"
            add_panel(ax_re, vpa, mu_axis, re_grid[i][j], rf"$\mathrm{{Re}},\ {theta_label}$", cmap,
                      boundary=boundaries[j])
            add_panel(ax_im, vpa, mu_axis, im_grid[i][j], r"$\mathrm{Im}$", cmap,
                      boundary=boundaries[j])
            ax_im.set_yticklabels([])
            if i == nrows - 1:
                ax_re.set_xlabel(VPA_LABEL)
                ax_im.set_xlabel(VPA_LABEL)
            if j == 0:
                ax_re.set_ylabel(MU_LABEL)

    fig.suptitle(rf"${quantity_label}$", fontsize=16)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Run discovery
# --------------------------------------------------------------------------

def find_runs(basedir):
    """Recursively find every directory under basedir containing a restart/
    subdirectory that itself contains restart files. Returns a list of
    (label, restart_dir) pairs, where label is the path from basedir to the
    run directory (the parent of restart/), with path separators flattened
    to underscores for use in a filename."""
    basedir_abs = os.path.abspath(basedir)
    runs = []
    for dirpath, dirnames, _filenames in os.walk(basedir):
        if "restart" not in dirnames:
            continue
        restart_dir = os.path.join(dirpath, "restart")
        if not glob.glob(os.path.join(restart_dir, "*.nc.[0-9]*")):
            continue
        rel = os.path.relpath(os.path.abspath(dirpath), basedir_abs)
        label = os.path.basename(basedir_abs) if rel == "." else rel.replace(os.sep, "_")
        runs.append((label, restart_dir))
    return sorted(runs)


# --------------------------------------------------------------------------
# Single-run processing
# --------------------------------------------------------------------------

def process_run(restart_dir, output_path, theta_values, kx_values, cmap,
                 phase_shift_to_flow_max=True, extra_phase_shift=0.0,
                 divide_by_maxwellian=False):
    input_file = find_input_file(restart_dir)
    params = parse_namelist(input_file)
    grids = build_grids(params)

    stem, nproc = find_restart_stem(restart_dir)
    naky, nakx, nzgrid = grids["naky"], grids["nakx"], grids["nzgrid"]
    nzed = 2 * nzgrid + 1
    ntubes, nspec = grids["ntubes"], grids["nspec"]
    world_size = naky * nakx * nzed * ntubes * nspec
    blocksize = world_size // nproc + 1

    layout = KxkyzLayout(naky, nakx, nzgrid, ntubes, nspec, grids["xyzs_layout"])

    akx, zed = grids["akx"], grids["zed"]
    if kx_values is None:
        positive = np.unique(np.round(akx[akx > 0], 12))
        n_default = min(4, len(positive))
        kx_values = sorted(positive[:n_default].tolist())
    ikx_list = [int(np.argmin(np.abs(akx - kx))) + 1 for kx in kx_values]
    iz_axis = np.arange(-nzgrid, nzgrid + 1)
    iz_list = [int(iz_axis[np.argmin(np.abs(zed - th))]) for th in theta_values]

    matched_kx = [akx[ikx - 1] for ikx in ikx_list]
    matched_theta = [zed[iz + nzgrid] for iz in iz_list]

    print(f"  Input file: {input_file}")
    print(f"  Restart files: {nproc} ({stem}.nc.0 .. {stem}.nc.{nproc - 1}), "
          f"xyzs_layout={grids['xyzs_layout']!r}, world_size={world_size}, blocksize={blocksize}")
    for req, mat in zip(kx_values, matched_kx):
        print(f"    kx requested={req:.6g} -> matched={mat:.6g}")
    for req, mat in zip(theta_values, matched_theta):
        print(f"    theta requested={req:.6g} -> matched={mat:.6g}")

    moments = compute_moments_vs_kx_theta(restart_dir, stem, layout, blocksize, grids)
    if phase_shift_to_flow_max:
        phase_per_kx = compute_per_kx_alignment_phase(zed, moments["upar"])
        print("    Phase-shifting each kx mode independently so its own "
              "<u_par * cos(theta)>_theta is real and maximal")
    else:
        phase_per_kx = np.ones(len(akx), dtype=complex)
    if extra_phase_shift:
        print(f"    Applying an additional constant phase shift of "
              f"{extra_phase_shift:.6g} rad (+ for kx>0, - for kx<0)")
    # phase_per_kx: mode's own alignment phase (independent per kx).
    # exp(i*extra_phase_shift*sign(kx)): an additional constant
    # (quadrature-type) phase shift, +/- so that reality g(-kx)=conj(g(kx))
    # is preserved; the kx=0 mode is left untouched (sign(0)=0).
    total_phase_per_kx = phase_per_kx * np.exp(1j * extra_phase_shift * np.sign(akx))

    vpa = grids["vpa"]
    mu_axis = grids["mu"] * grids["bmin"] / grids["temp"]
    if divide_by_maxwellian:
        print("    Dividing g by the Maxwellian F0 (stella's own vpa/mu normalization)")
        maxwell_per_theta = [
            maxwellian_factor(vpa, mu_axis, th, grids["epsilon"], grids["temp"])
            for th in matched_theta
        ]

    reader = RestartReader(restart_dir, stem)
    re_grid = [[None] * len(theta_values) for _ in kx_values]
    im_grid = [[None] * len(theta_values) for _ in kx_values]
    moment_theta = [None] * len(kx_values)
    for i, ikx in enumerate(ikx_list):
        phase = total_phase_per_kx[ikx - 1]
        moment_theta[i] = dict(
            upar=moments["upar"][ikx - 1, :] * phase,
            temperature=moments["temperature"][ikx - 1, :] * phase,
            # Free energy is |.|^2-based and therefore invariant under the
            # unit-modulus phase shift, so it's taken directly, unshifted.
            fe_M_even=moments["fe_M_even"][ikx - 1, :],
            fe_M_odd=moments["fe_M_odd"][ikx - 1, :],
            fe_K_even=moments["fe_K_even"][ikx - 1, :],
            fe_K_odd=moments["fe_K_odd"][ikx - 1, :],
        )
        for j, iz in enumerate(iz_list):
            world_idx = layout.world_index(iky=1, ikx=ikx, iz=iz, it=1, is_=1)
            iproc = world_idx // blocksize
            local_glo = world_idx - iproc * blocksize
            gr, gi = reader.read_g(iproc, local_glo)
            g_shifted = (gr + 1j * gi) * phase
            if divide_by_maxwellian:
                g_shifted = g_shifted / maxwell_per_theta[j]
            re_grid[i][j] = g_shifted.real
            im_grid[i][j] = g_shifted.imag
    reader.close()

    quantity_label = r"g^{Z} / F_0" if divide_by_maxwellian else r"g^{Z}"
    with PdfPages(output_path) as pdf:
        make_combined_page(pdf, grids, kx_values, theta_values, matched_kx, matched_theta,
                            re_grid, im_grid, moment_theta, cmap, quantity_label=quantity_label)

    print(f"  Wrote {output_path}")


# ============================================================================
# CONFIG - edit these and run: python3 plot_zonal_distribution.py
# ============================================================================

BASEDIR = "."

# Theta (zed) values in radians, one per column.
THETA = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

# kx values, one per row. Set to None to auto-select the four smallest
# positive nonzero kx grid values (independently for each run found).
KX = None

CMAP = "coolwarm"

# Each Fourier mode of a ky=0 (zonal) field can be given its own phase
# without changing the physics, as long as reality (g(-kx)=conj(g(kx))) is
# preserved. When True, each kx mode independently gets the phase that makes
# its own theta-average of (parallel flow * cos(theta)) real and maximal
# (the zonal harmonics in a turbulent snapshot generally do not share one
# coherent spatial phase, so this is done mode-by-mode rather than as a
# single rigid shift in x).
PHASE_SHIFT_TO_FLOW_MAX = True

# Additional constant phase shift (radians) applied on top of the above:
# +EXTRA_PHASE_SHIFT for kx>0 modes, -EXTRA_PHASE_SHIFT for kx<0 modes (the
# kx=0 mode is untouched).
EXTRA_PHASE_SHIFT = np.pi / 2

# Divide g by the background Maxwellian F0 before plotting the heatmaps
# (stella's own vpa/mu normalization, grids_velocity.f90: exp(-vpa^2 -
# 2*mu*B(theta))). Does not affect the parallel-flow/temperature moment
# lines, which are always moments of the raw g.
DIVIDE_BY_MAXWELLIAN = True

# ============================================================================


def main():
    runs = find_runs(BASEDIR)
    if not runs:
        print(f"No restart/ directories with restart files found under {BASEDIR}")
        return

    for label, restart_dir in runs:
        output_path = os.path.join(BASEDIR, f"fig_{label}_dist_fn_zonal.pdf")
        print(f"[{label}] {restart_dir}")
        try:
            process_run(restart_dir, output_path, THETA, KX, CMAP,
                        phase_shift_to_flow_max=PHASE_SHIFT_TO_FLOW_MAX,
                        extra_phase_shift=EXTRA_PHASE_SHIFT,
                        divide_by_maxwellian=DIVIDE_BY_MAXWELLIAN)
        except Exception as exc:
            print(f"  Skipped ({exc})")


if __name__ == "__main__":
    main()
