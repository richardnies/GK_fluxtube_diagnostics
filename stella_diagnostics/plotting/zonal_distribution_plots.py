"""Plotting and per-run/per-basedir orchestration for the zonal (ky=0)
distribution-function diagnostic.

Extracted from example_plots/plot_zonal_distribution.py.
"""

import glob
import os

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from stella_diagnostics.io.restart import (
    KxkyzLayout,
    RestartReader,
    build_grids,
    find_input_file,
    find_restart_stem,
    parse_namelist,
)
from stella_diagnostics.physics.zonal_distribution import (
    compute_moments_vs_kx_theta,
    compute_per_kx_alignment_phase,
    maxwellian_factor,
    trapped_passing_boundary,
)

VPA_LABEL = r"$v_\parallel / v_t$"
MU_LABEL = r"$\mu B_{\mathrm{min}} / T$"


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
    from matplotlib.backends.backend_pdf import PdfPages

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
