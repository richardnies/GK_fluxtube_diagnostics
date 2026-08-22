"""Zonal (ky=0) Rosenbluth-Hinton residual-flow ratios vs kx, for one or
more runs plotted as separate curves in the same axes:

    -2*eps/q * <u_par> / <v_E>
    -2/q     * <u_par*cos(theta)> / <v_E>

where <> is a field-line average (dl/B weighted, optionally *cos(theta))
combined with a time average over the configured window, `v_E =
-d(phi)/dx` (evaluated per-kx, i.e. i*kx*phi), `eps` is the inverse
aspect ratio, and `q` is the safety factor. A third panel above the other
two shows E_RH(kx) (log y-axis), time-averaged over that same window. The
left column's kx-axis is log-scale (shared across all three panels).

Only the real part of each ratio is plotted -- the imaginary part (a
residual phase/GAM-oscillation artifact of a finite, not-fully-converged
time window) is discarded. kx=0 is always excluded: both ratios are 0/0
there (u_par and v_E both vanish for the true zonal, x-independent
component) and E_RH is identically 0 there (Gamma0=1 at kx=0), matching
the same `if kx <= 0: continue` convention already used throughout
stella_diagnostics/physics/rosenbluth_hinton.py's own per-kx plots.

A second column of panels shows the same zonal `u_par` and `v_E =
-d(phi)/dx` (evaluated in real space this time, i.e. FFT'd back from kx to
x rather than divided kx-by-kx), each time-averaged over the same window
and plotted vs x for every run in one shared axes per quantity. Unlike the
left column's kx-space ratios, these are not divided by v_E (that would
be 0/0 pointwise in x for a real-valued profile) -- instead `<u_par>` and
`<u_par*cos(theta)>` are scaled by the same prefactors as the left
column's numerators (-2*eps/q and -2/q respectively) so the two columns'
row-2/row-3 traces are directly comparable in normalisation, and kx=0 is
not excluded here (there is no x=0/0 singularity in real space).

Usage:
    python plot_zonal_upar_vE_kx.py <config.py>

<config.py> defines `dirnames` (required, list of run directories) and
optionally:
    filename     -- run filename stem (default "CBC")
    code         -- (default "stella")
    labels       -- one label per run (default: dirnames themselves)
    colors       -- one color per run (default: matplotlib color cycle)
    time_avg     -- averaging window width (default 20)
    time_val_avg -- averaging window center (default None -> trailing
                    window ending at each run's own last time sample)
    kx_min       -- lower kx bound to plot (default 0.0; the kx-axis is
                    log-scale, so a value <= 0 is floored to half the
                    smallest positive kx on the grid, across every run)
    kx_max       -- upper kx bound to plot (default 0.5, clamped to the
                    kx grid's own largest positive value, across every
                    run, if that's smaller)
    x_min        -- lower x/rho_i bound to plot (default None -> the real-
                    space x grid's own lower edge, across every run)
    x_max        -- upper x/rho_i bound to plot (default None -> the real-
                    space x grid's own upper edge, across every run)
    nx           -- real-space x grid resolution passed to
                    StellaRun.get_quantity_x_y (default None -> package
                    default)
    eps          -- override the inverse aspect ratio (default None ->
                    estimated per-run from bmag via
                    stella_diagnostics.scan.zonal_flow_scan.estimate_eps_from_bmag)
    figname_add  -- suffix appended before the default filename's
                    extension (default "")
    figname      -- output filename (default "fig_zonal_upar_vE_kx.pdf")
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.zonal_flow_scan import estimate_eps_from_bmag
from stella_diagnostics.spectral.stats import dt_weighted_mean

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirnames",))

filename = getattr(config, "filename", "CBC")
code = getattr(config, "code", "stella")
labels = getattr(config, "labels", config.dirnames)
colors = getattr(config, "colors", None)
time_avg = getattr(config, "time_avg", 20)
time_val_avg = getattr(config, "time_val_avg", None)
kx_min = getattr(config, "kx_min", 0.0)
kx_max = getattr(config, "kx_max", 0.5)
x_min = getattr(config, "x_min", None)
x_max = getattr(config, "x_max", None)
nx = getattr(config, "nx", None)
eps_override = getattr(config, "eps", None)
figname_add = getattr(config, "figname_add", "")

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 13), sharex="col")
(ax_ERH, ax_vE_x), (ax_eps, ax_upar_x), (ax_cos, ax_uparcos_x) = axs

x_grid_min, x_grid_max = None, None
kx_grid_max = None
kx_grid_min_positive = None

for i, dirname in enumerate(config.dirnames):
    label = labels[i]
    color = colors[i] if colors is not None else None

    run = StellaRun(dirname + "/" + filename, code=code)
    time_all = run.get_time_array()

    # get_quantity_kx_ky (and its underlying get_quantity_zed_kx_ky) crash
    # if time_val and time_avg are passed together: the time_avg branch
    # resolves a time_idx window first, but the time_val branch then runs
    # AFTER it and overwrites time_idx with a single scalar index, so the
    # netCDF read drops the time axis entirely instead of averaging over
    # it (quantities/registry.py, get_quantity_zed_kx_ky lines ~56-65) --
    # confirmed by direct test, not just code reading. quantities/
    # realspace.py's get_quantity_zed_x_y has the two branches in the
    # opposite (correct) order, so real-space callers like
    # scan/zonal_flow_scan.py don't hit this. Worked around here by
    # resolving the window center to a plain time_idx ourselves and never
    # passing time_val alongside time_avg.
    if time_val_avg is None:
        time_idx_center = -1
        time_center = time_all[-1]
    else:
        time_idx_center = run.get_time_idx(time_val_avg)
        time_center = time_val_avg
    # Full-width (time_avg) window centered on time_center, SHIFTED (not
    # merely clamped/shrunk) when that would run past either edge of the
    # run's time range -- same convention as, and kept consistent with,
    # quantities.realspace.get_quantity_zed_x_y/quantities.registry.
    # get_quantity_zed_kx_ky (the upar/vE calls below), so E_RH is
    # averaged over the same window width as the flow quantities instead
    # of a naive clamp silently halving it whenever time_center sits at
    # or near either edge (e.g. the implicit time_all[-1] reference used
    # whenever time_val_avg is None).
    time_min_win = time_center - time_avg / 2
    time_max_win = time_center + time_avg / 2
    if time_min_win < 0:
        time_max_win += -time_min_win
        time_min_win = 0
    if time_max_win > time_all[-1]:
        time_min_win -= (time_max_win - time_all[-1])
        time_max_win = time_all[-1]
    time_min_win = max(0, time_min_win)

    eps = eps_override if eps_override is not None else estimate_eps_from_bmag(run)
    q = float(run.ncdata.variables["q"].getValue())

    upar_kx, kx, ky, _ = run.get_quantity_kx_ky(
        quantity="upar", time_idx=time_idx_center, time_avg=time_avg, only_zonal=True)
    uparcos_kx, _, _, _ = run.get_quantity_kx_ky(
        quantity="upar", time_idx=time_idx_center, time_avg=time_avg, only_zonal=True, mult_zed="cos")
    dxphi_kx, _, _, _ = run.get_quantity_kx_ky(
        quantity="phi", time_idx=time_idx_center, time_avg=time_avg, only_zonal=True, kx_order=1)

    upar0 = upar_kx[:, 0]
    uparcos0 = uparcos_kx[:, 0]
    vE0 = -dxphi_kx[:, 0]

    kx_grid_max = kx.max() if kx_grid_max is None else max(kx_grid_max, kx.max())
    kx_positive = kx[kx > 0]
    if kx_positive.size:
        kx_grid_min_positive = kx_positive.min() if kx_grid_min_positive is None else min(kx_grid_min_positive, kx_positive.min())

    # Same field-line- and time-averaged, zonal (ky=0) upar/vE as above, but
    # evaluated in real space (x) rather than reduced to kx.
    upar_x_y, x, _, _ = run.get_quantity_x_y(
        quantity="upar", time_idx=time_idx_center, time_avg=time_avg, only_zonal=True, nx=nx)
    uparcos_x_y, _, _, _ = run.get_quantity_x_y(
        quantity="upar", time_idx=time_idx_center, time_avg=time_avg, only_zonal=True, mult_zed="cos", nx=nx)
    dxphi_x_y, _, _, _ = run.get_quantity_x_y(
        quantity="phi", time_idx=time_idx_center, time_avg=time_avg, only_zonal=True, kx_order=1, nx=nx)
    upar_of_x = upar_x_y[:, 0]
    uparcos_of_x = uparcos_x_y[:, 0]
    vE_of_x = -dxphi_x_y[:, 0]

    x_grid_min = x.min() if x_grid_min is None else min(x_grid_min, x.min())
    x_grid_max = x.max() if x_grid_max is None else max(x_grid_max, x.max())

    ax_vE_x.plot(x, vE_of_x, marker=".", label=label, c=color)
    ax_upar_x.plot(x, -2 * eps / q * upar_of_x, marker=".", label=label, c=color)
    ax_uparcos_x.plot(x, -2 / q * uparcos_of_x, marker=".", label=label, c=color)

    E_RH_t_kx, time_E_RH, kx_E_RH = run.get_E_RH_t_kx(time_min=time_min_win, time_max=time_max_win, kx_max=1e5)
    E_RH_kx = dt_weighted_mean(E_RH_t_kx, time=time_E_RH, axis=0)

    mask = (kx > 0) & (kx >= kx_min) & (kx <= kx_max)
    mask_ERH = (kx_E_RH > 0) & (kx_E_RH >= kx_min) & (kx_E_RH <= kx_max)

    with np.errstate(divide="ignore", invalid="ignore"):
        term_eps = np.real(-2 * eps / q * upar0 / vE0)
        term_cos = np.real(-2 / q * uparcos0 / vE0)

    ax_ERH.semilogy(kx_E_RH[mask_ERH], E_RH_kx[mask_ERH], marker=".", label=label, c=color)
    ax_eps.plot(kx[mask], term_eps[mask], marker=".", label=label, c=color)
    ax_cos.plot(kx[mask], term_cos[mask], marker=".", label=label, c=color)

ax_ERH.set_ylabel(r"$E_\mathrm{RH}$")
ax_ERH.grid(True)
ax_ERH.legend(fontsize=12)

ax_eps.set_ylabel(r"$-\frac{2\epsilon}{q}\frac{\langle u_\parallel\rangle}{\langle v_E\rangle}$")
ax_eps.grid(True)

ax_cos.set_ylabel(r"$-\frac{2}{q}\frac{\langle u_\parallel\cos\theta\rangle}{\langle v_E\rangle}$")
ax_cos.set_xlabel(r"$k_x \rho_i$")
ax_cos.grid(True)

ax_cos.set_xscale("log")
# kx=0 is always excluded from what's plotted (see module docstring), and a
# log axis can't include it either -- floor the lower bound at half the
# smallest positive kx actually on the grid whenever kx_min itself is <= 0.
kx_xlim_min = kx_min if kx_min > 0 else kx_grid_min_positive / 2
ax_cos.set_xlim([kx_xlim_min, min(kx_max, kx_grid_max)])

ax_vE_x.set_ylabel(r"$\langle v_E\rangle$")
ax_vE_x.grid(True)
ax_vE_x.legend(fontsize=12)

ax_upar_x.set_ylabel(r"$-\frac{2\epsilon}{q}\langle u_\parallel\rangle$")
ax_upar_x.grid(True)

ax_uparcos_x.set_ylabel(r"$-\frac{2}{q}\langle u_\parallel\cos\theta\rangle$")
ax_uparcos_x.set_xlabel(r"$x/\rho_i$")
ax_uparcos_x.grid(True)

ax_uparcos_x.set_xlim([x_min if x_min is not None else x_grid_min, x_max if x_max is not None else x_grid_max])

plt.tight_layout()
fig.savefig(getattr(config, "figname", None) or "fig_zonal_upar_vE_kx" + figname_add + ".pdf")
