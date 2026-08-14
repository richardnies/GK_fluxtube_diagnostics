"""Multiple quantities vs zed for one run.

Usage:
    python plot_quantities_over_zed.py <config.py>

<config.py> defines `dirname` (required, the run's directory) and
optionally `filename`, `code`, `kwargs` (dict forwarded to
plot_quantities_over_zed), `ylim`, `figname` -- same dirname/filename
split as every other single-run driver in this package (changed from
treating dirname as the full filename_base, for config-vocabulary
consistency).

`kwargs` defaults to `{}`, which plots nothing (every quantity in
plot_quantities_over_zed is opt-in via its own `plot_<name>=True` flag,
e.g. `plot_phi`/`plot_B`/`plot_Gamma0`/`plot_omega_s_k`/`plot_nablax2`/
`plot_nablaxy`/`plot_nablay2`/`plot_qflx`/`plot_gi`/`plot_ge` -- see
plot_quantities_over_zed's own body for the full list and their
`<name>_idx`/`norm_<name>`-style modifiers) -- a config that doesn't set
`kwargs` at all silently produces a blank axes with no error, so always
set at least one `plot_*` flag.
"""
import sys

import matplotlib.pyplot as plt

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.plotting.mpl_helpers import set_default_style
from stella_diagnostics.scan.config import load_scan_config

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <config.py>")

set_default_style()
config = load_scan_config(sys.argv[1], required=("dirname",))

run = StellaRun(config.dirname + "/" + getattr(config, "filename", "CBC"), code=getattr(config, "code", "stella"))
fig, ax = run.plot_quantities_over_zed(**getattr(config, "kwargs", {}))

ax.legend()
ax.set_ylim(getattr(config, "ylim", None))
ax.grid()

plt.savefig(getattr(config, "figname", "fig_quantities_over_zed.pdf"))
