"""Quantity name -> LaTeX title/label lookup, used by the real-space plots.

Collapses the ``if quantity == "phi": title = r"$\\varphi$" elif ...``
dispatch that was pasted verbatim 4 times in plotting/realspace_plots.py.
"""


def get_quantity_label(quantity):
    if quantity == "phi":
        title = r"$\varphi$"
    elif quantity == "density":
        title = r"$n$"
    elif quantity == "upar":
        title = r"$u_\parallel$"
    elif quantity == "temperature":
        title = r"$T$"
    elif quantity == "pressure_perp":
        title = r"$P_\perp$"
    elif quantity == "qpar":
        title = r"$q_\parallel$"
    elif quantity == "qperp":
        title = r"$q_\perp$"
    elif quantity == "dyphi-dxphi":
        title = r"$\partial_y \varphi \partial_x \varphi$"
    elif quantity == "dyphi-dyphi":
        title = r"$(\partial_y \varphi)^2"
    elif quantity == "dyPrp-dxphi":
        title = r"$\partial_y P_\perp \partial_x \varphi$"
    elif quantity == "dyPrp-dyphi":
        title = r"$\partial_y P_\perp \partial_y \varphi$"
    elif quantity == "dyT-dxphi":
        title = r"$\partial_y T \partial_x \varphi$"
    else:
        title = ""
    return title
