"""Qflx(t)/E_phi(t)/E_upar(t) comparison across a set of runs.

Usage:
    python plot_flux_time.py <scan_config.py>

`<scan_config.py>` is a small data-only Python file defining `dirnames`
(required) and optionally `labels`, `colors`, `filename`, `code`, `Q_div`,
`skip_phi2`, `plot_ratio`, `ylim`, `figname_add` -- see scan_configs/ for
worked examples (scan_nu_var.py, scan_upwind.py, scan_nu_var2.py), each
corresponding to a comparison that used to be a hardcoded, commented-in/out
block inside this script.

This script itself is never copied or edited to add a new comparison --
only new config files are added. Any improvement to the underlying analysis
(stella_diagnostics.scan.flux_energy_scan) automatically applies to every
config that exists, current or future.
"""
import sys

from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config
from stella_diagnostics.scan.flux_energy_scan import plot_qflx_and_energy_vs_time

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <scan_config.py>")

set_default_style()
config = load_scan_config(sys.argv[1])

fig, ax = plot_qflx_and_energy_vs_time(
    config.dirnames,
    labels=getattr(config, "labels", None),
    colors=getattr(config, "colors", None),
    filename=getattr(config, "filename", "CBC"),
    code=getattr(config, "code", "stella"),
    Q_div=getattr(config, "Q_div", 10),
    skip_phi2=getattr(config, "skip_phi2", False),
    plot_ratio=getattr(config, "plot_ratio", False),
)

figname_add = getattr(config, "figname_add", "")

ax.set_xlim(xmin=0)
ax.set_ylim(getattr(config, "ylim", None))
fig.savefig("fig_qflx_over_time" + figname_add + ".pdf")

ax.set_xlim(xmin=1e2)
ax.set_xscale("log")
fig.savefig("fig_qflx_over_time" + figname_add + "_loglog.pdf")
