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

The namelist/restart-file I/O, grid reconstruction, and physics live in
stella_diagnostics.io.restart / stella_diagnostics.physics.zonal_distribution
/ stella_diagnostics.plotting.zonal_distribution_plots -- this script is
just the CONFIG block plus orchestration (find_runs + process_run), kept in
its original config-at-the-bottom shape rather than forced into the
load_scan_config+StellaRun driver pattern used elsewhere in example_plots/,
since this script involves no StellaRun/netCDF-output-file at all.
"""
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

if shutil.which("latex"):
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "serif"
else:
    print("Warning: no LaTeX installation found, falling back to mathtext.", file=sys.stderr)
matplotlib.rcParams["font.size"] = 13

from stella_diagnostics.plotting.zonal_distribution_plots import find_runs, process_run

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
