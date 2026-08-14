"""Qflx(t)/phi2(t) overlay traces (split by zonal kx-band) drawn on top of
a quantity(x, t) contour plot, plus the quantity-name -> title mapping for
that contour.

Extracted from example_plots/plot_contour_quantity_vs_t_x.py.
"""

import numpy as np

from stella_diagnostics.io.cache import cached
from stella_diagnostics.io.codes import get_rho_label

_QUANTITY_TITLES = {
    "phi": r"$\varphi$",
    "temperature": r"$T$",
    "pressure_perp": r"$P_\perp$",
    "Q_es": r"$\int\mathrm{d}y\; T \partial_y \varphi$",
    "dyphi-dxphi": r"$\int\mathrm{d}y\; \partial_x \varphi \partial_y \varphi$",
    "dyT-dxphi": r"$\int\mathrm{d}y\; \partial_x \varphi \partial_y T$",
    "dyT-dyphi": r"$\int\mathrm{d}y\; \partial_y \varphi \partial_y T$",
    "dyphi-dyphi": r"$\int\mathrm{d}y\; (\partial_y \varphi)^2 $",
    "dyphi-P": r"$\int\mathrm{d}y\; \partial_y \varphi P$",
    "dyphi-T": r"$\int\mathrm{d}y\; \partial_y \varphi T$",
    # NOTE: not present in the original quantity->title elif chain (a real
    # coverage gap -- P_RH_tot was the quantity actually in active use,
    # so the suptitle was silently blank). Added here rather than
    # preserved as a gap, since it's a pure label fix with no effect on
    # any computed number.
    "P_RH_tot": r"$P_\mathrm{RH}$",
    "P_RH_even": r"$P_\mathrm{RH}^\mathrm{even}$",
    "P_RH_odd": r"$P_\mathrm{RH}^\mathrm{odd}$",
}


def get_quantity_x_t_title(quantity, kx_order=0, remove_zonal=False, only_zonal=False, mult_zed=None):
    """LaTeX title string for a quantity(x, t) contour plot."""
    if kx_order == 1:
        title = r"$\partial_x$"
    elif kx_order == 2:
        title = r"$\partial^2_x$"
    else:
        title = ""

    title += _QUANTITY_TITLES.get(quantity, "")

    if remove_zonal:
        title += r"$_\mathrm{NZ}(y=0$"
    elif only_zonal:
        title += r"$_\mathrm{Z}$"

    if mult_zed == "vdriftx":
        title += r"$v_{Dx}$"

    return title


@cached(version=1)
def compute_qflx_phi2_overlay(run):
    """(time, qflx, time_phi, phi2_NZ, phi2_Z, phi2_Z_LW, phi2_Z_SW, xmax):
    heat flux and zonal/nonzonal phi^2 (split at |kx rho_i| = 0.3) vs
    time, plus the x-axis half-width xmax used to normalise/offset these
    traces when overplotting them on a quantity(x, t) contour.
    """
    _, _, qflx, time = run.get_fluxes_over_time()
    phi2_t_kx_ky = run.ncdata["phi2_vs_kxky"][:]
    kx = run.ncdata["kx"][:]
    xmax = np.pi / (kx[1] - kx[0])

    phi2_NZ = np.sum(phi2_t_kx_ky[:, :, 1:], axis=(1, 2))
    phi2_Z_LW = np.sum(phi2_t_kx_ky[:, np.abs(kx) < 0.3, 0], axis=1)
    phi2_Z_SW = np.sum(phi2_t_kx_ky[:, np.abs(kx) >= 0.3, 0], axis=1)
    phi2_Z = phi2_Z_LW + phi2_Z_SW

    time_phi = run.ncdata["t"][:]

    return time, qflx, time_phi, phi2_NZ, phi2_Z, phi2_Z_LW, phi2_Z_SW, xmax


def add_qflx_phi2_overlay(ax, run):
    """Draws the qflx/phi2 overlay traces (log-normalised, offset to the
    left half of the axes) onto an existing quantity(x, t) contour Axes."""
    time, qflx, time_phi, phi2_NZ, phi2_Z, phi2_Z_LW, phi2_Z_SW, xmax = compute_qflx_phi2_overlay(run)

    norm = np.log10(qflx.max()) / (xmax / 2)
    ax.plot(time, np.log10(qflx) / norm - xmax, c="k", label=r"$Q$")

    rho_label = get_rho_label(run.ncdata)
    norm = phi2_Z.max() / (xmax / 2)
    ax.plot(time_phi, phi2_Z / norm - xmax, c="0.5", label=r"$(\phi^Z)^2$")
    ax.plot(time_phi, phi2_Z_LW / norm - xmax, c="0.5", label=r"$(\phi^Z)^2 (k_x %s<0.3)$" % rho_label, ls="--", lw=2)
    ax.plot(time_phi, phi2_Z_SW / norm - xmax, c="0.5", label=r"$(\phi^Z)^2 (k_x %s \geq 0.3)$" % rho_label, ls=":", lw=2)

    norm = phi2_NZ.max() / (xmax / 2)
    ax.plot(time_phi, phi2_NZ / norm - xmax, c="forestgreen", label=r"$(\phi^{NZ})^2$")
