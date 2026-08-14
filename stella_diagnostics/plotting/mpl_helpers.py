"""Shared matplotlib figure/axes and color-palette helpers.

Collapses the "create a default figure/axes if the caller didn't pass
one" boilerplate (``if ax is None: fig, ax = plt.subplots(...)``)
repeated at ~34 call sites across the plotting modules, and the
``sns.color_palette("coolwarm", n)`` boilerplate repeated in the
Rosenbluth-Hinton plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_or_create_ax(fig=None, ax=None, figsize=(12, 9), **subplot_kwargs):
    """Return (fig, ax), creating a new figure/axes if ax is None.

    ``ax`` may be a single Axes or an array of Axes (for multi-panel
    layouts via ``nrows=``/``ncols=``) -- mirrors how the original code
    used ``ax``/``axs`` interchangeably. ``figsize`` and any other
    ``plt.subplots`` keyword stay explicit per call site rather than
    being unified to one default, since they vary intentionally.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)
    return fig, ax


def coolwarm_palette(n):
    """``seaborn`` "coolwarm" palette with n colors."""
    return sns.color_palette("coolwarm", n)


def resolve_vmin_vmax(Z, vmin, vmax, logarithmic, default_vmax, fill_vmin_default=True):
    """Resolves the vmin/vmax sentinel values shared (up to small,
    real per-caller differences) across the 2D pcolormesh plots in
    zed_plots.py/realspace_plots.py/physics.velocity_space: ``vmax=None``
    -> `default_vmax` (computed by the caller, since callers differ on
    whether that's ``Z.max()`` or ``np.abs(Z).max()``); ``vmax="last"``
    -> abs of the last time-column's max; ``vmin="symm"`` -> ``-vmax`` if
    not logarithmic, else ``1e-2*vmax`` (a negative vmin is invalid for
    LogNorm -- "symmetric" only means anything for a linear, divergent
    colormap; for logarithmic it instead falls back to the same
    fraction-of-vmax floor as the vmin=None case below); ``vmin=None`` ->
    ``1e-2*vmax`` if logarithmic else ``Z.min()``, unless
    `fill_vmin_default=False` (one caller leaves `vmin=None` as-is,
    relying on matplotlib's own autoscale).
    """
    if vmax is None:
        vmax = default_vmax
    if vmax == "last":
        vmax = np.abs(Z[:, -1]).max()
    if vmin == "symm":
        vmin = 1e-2*vmax if logarithmic else -vmax
    elif vmin is None and fill_vmin_default:
        vmin = 1e-2*vmax if logarithmic else Z.min()
    return vmin, vmax


def set_default_style(usetex=True, font_family="serif", font_size=24, axes_titlepad=15, **extra_rcparams):
    """Collapses the ``plt.rcParams.update({...})`` block duplicated
    verbatim at the top of most example_plots/*.py scripts. Call once,
    before creating any figures.
    """
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": font_family,
            "font.size": font_size,
            "axes.titlepad": axes_titlepad,
            **extra_rcparams,
        }
    )
