"""Movie overview of the collisional ZF relaxation: distribution-function
snapshots in (zed, vpa) and (vpa, mu), alongside a column of 1D
time-series {E_RH, P_RH_coll, toroidal-rotation coefficient,
Pfirsch-Schlueter-like coefficient} with a moving vertical dashed line
marking the current time (E_RH/E_RH(0), -P_RH_coll/(nu*E_RH),
-2(eps/q)<upar>/<vE>, -2(1/q)<upar cos theta>/<vE>).

Every quantity/panel below is loaded independently and wrapped so a missing
netCDF variable (a deck that didn't enable the write_* flag for it) only
disables THAT quantity's panel (drawn as a "not available" placeholder,
still keeping its slot and the shared time axis) -- it never aborts the
whole movie. E_RH is the one exception with real knock-on effects: it's
what the dominant-kx selection and the common frame-count/time-alignment
are both derived from, so if it's missing, kx_dominant becomes None (the
(vpa,mu) kx window and flow-quantity extraction both fall back to an
all-kx/RMS treatment, printing a warning) and the frame count falls back to
the run's full main time array instead of the usual E_RH/P_RH-trimmed one.

By default the kx-resolved panels/curves (P_RH_coll's E_RH, the (vpa,mu)
distribution panel, upar/vE) are all restricted to a single, auto-selected
"dominant" kx -- the kx with the largest time-averaged E_RH. Pass --total to show the
TOTAL (kx-summed) E_RH/P_RH_coll instead, and the RMS-over-x of the flow
quantities (upar, upar*cos(theta), vE) in place of the single-kx
phase-rotated amplitudes -- see get_rms_x_t below. RMS quantities are
signless, so the toroidal-rotation/Pfirsch-Schlueter-like coefficient
panels drop the leading minus sign in --total mode (see the code near
toroidal_coef/ps_coef) -- they're magnitude-only, not the signed
neoclassical-convention ratio the dominant-kx panels show.

The (vpa,mu) panel can optionally be kx-RESOLVED (via g2_vs_kxvpamus,
gated by write_g2_vs_kxvpamus=.true. in the deck) and restricted with
--kx_min/--kx_max to just the dominant kx (the default whenever
write_g2_vs_kxvpamus is present and --total isn't set) or a user-chosen
window. If write_g2_vs_kxvpamus isn't in the run's output, the panel
silently falls back to the plain (always kx-summed) g2_vs_vpamus -- fine
for a single-relevant-kx run (e.g. a single-kx linear run: one nonzero |kx|
plus its zero/conjugate partners, so kx-summed and kx-selected are the
same thing) but NOT a true per-kx view on a run with
several physically distinct nonzero kx; pass --kx_min/--kx_max explicitly
in that case and rerun with write_g2_vs_kxvpamus=.true. first (this script
errors out if you pass them without that variable present, rather than
silently ignoring them). The (zed,vpa) panel has no kx-resolved equivalent
at all in stella's diagnostics (only g2_vs_zvpas, always kx-summed) --
kx_min/kx_max can never restrict it.

Usage (run from wherever RUNDIR lives, or pass an absolute/relative RUN_PATH):
    python3 movie_RH_zonal_evolution.py RUN_PATH [--time_idx_step N] [--tmin T] [--tmax T] [--total] [--kx_min X] [--kx_max X]
where RUN_PATH is RUNDIR/FILE, the run's StellaRun path (RUNDIR/FILE.out.nc,
RUNDIR/FILE.in, ...) -- e.g. movie_kx0.2_RH/movie_kx0.2_RH. RUNDIR and FILE
need not match (unlike the old positional run_name argument, which assumed
they did); the movie title and output filename (RUNDIR_movie.mp4, or
RUNDIR_movie_total.mp4 if --total is given) are taken
from RUNDIR, not FILE. --time_idx_step (default 1) subsamples both the
ANIMATED frames (every time_idx_step-th snapshot is rendered) and the
flow-quantity (upar/vE) evaluation -- the latter is the actual bottleneck,
one run.get_quantity_kx_ky/get_quantity_x_y call per time index in a plain
Python loop, unlike E_RH/P_RH_coll (single vectorized netCDF-slice reads,
cheap regardless of resolution and so always computed at full resolution
regardless of time_idx_step) -- so a large time_idx_step actually gives a
fast preview instead of only shrinking the output file. --tmin/--tmax
(default: the run's full range) restrict BOTH the animated frames and the
static 1D panels' x-axis to t in [tmin, tmax] -- e.g. to zoom into the
early-time transient or skip a slow, uninteresting late-time tail; every
kx-selection/RMS quantity above is still computed from the FULL run
(tmin/tmax only crop what's displayed, not what feeds the dominant-kx
selection or the phase-rotation reference time).

nu, eps=rhoc/rmaj, and q=qinp are read automatically -- nu from the
run's own 'vnew' netCDF variable, rhoc/rmaj/qinp by parsing RUNDIR/FILE.in's
&millergeo_parameters (the run.safety_factor/run.aspect_ratio attributes on
StellaRun are not reliable for Miller geometry -- see io.run's
"unverified placeholder aspect_ratio" warning -- so this reads the deck
directly instead).
"""
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.physics.velocity_space import plot_contour_gzvs, plot_contour_gvmu_vpa
from stella_diagnostics.plotting.mpl_helpers import set_default_style

set_default_style(font_size=14)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run_path", help="RUNDIR/FILE, the StellaRun path")
parser.add_argument("--time_idx_step", type=int, default=1, help="render every Nth snapshot (default 1 = all)")
parser.add_argument("--tmin", type=float, default=None, help="restrict animated frames + static plots to t >= tmin (default: full range)")
parser.add_argument("--tmax", type=float, default=None, help="restrict animated frames + static plots to t <= tmax (default: full range)")
parser.add_argument("--total", action="store_true", help="show total (kx-summed) E_RH/P_RH_coll and RMS-in-x flow quantities instead of the single dominant-kx values")
parser.add_argument("--rms_tavg", type=float, default=None, help="time-averaging window (in t*vTi/a) applied BEFORE the RMS-in-x in --total/no-dominant-kx mode (default: the run's own snapshot spacing)")
parser.add_argument("--kx_min", type=float, default=None, help="(vpa,mu) panel |kx| lower bound (requires write_g2_vs_kxvpamus=.true. in the deck; default: dominant kx, or all kx in --total mode)")
parser.add_argument("--kx_max", type=float, default=None, help="(vpa,mu) panel |kx| upper bound (requires write_g2_vs_kxvpamus=.true. in the deck; default: dominant kx, or all kx in --total mode)")
args = parser.parse_args()

RUN_PATH = args.run_path
NAME = RUN_PATH.rsplit("/", 1)[0] if "/" in RUN_PATH else RUN_PATH  # RUNDIR, for the title and output filename
TIME_IDX_STEP = args.time_idx_step
UPAR_YLIM = (-1, 2)
FPS = 20


def safe(label, fn, *fn_args, **fn_kwargs):
    """Run fn(*fn_args, **fn_kwargs); on any failure (typically a missing
    netCDF variable -- the deck didn't enable the write_* flag for this
    quantity), print a warning and return None instead of crashing, so
    every OTHER quantity/panel can still be computed and plotted."""
    try:
        return fn(*fn_args, **fn_kwargs)
    except Exception as e:
        print(f"WARNING: '{label}' unavailable in this run's output ({type(e).__name__}: {e}) -- skipping its panel.")
        return None


def mark_unavailable(ax, label, xlim):
    """Static 'not available' placeholder for a panel whose underlying
    quantity failed to load -- keeps the panel's slot/shared time axis so
    the rest of the figure's layout is unaffected."""
    ax.text(0.5, 0.5, f"{label}\nnot available in this run's output", ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="gray", wrap=True)
    ax.set_xlim(*xlim)
    ax.set_xticks([])
    ax.set_yticks([])


def read_namelist_float(deck_path, key):
    """Regex-parse a plain 'key = value' float assignment out of a stella
    input namelist file -- robust to whichever namelist block it's under,
    unlike relying on StellaRun's own geometry-file parsing (see module
    docstring)."""
    text = open(deck_path).read()
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([-+0-9.eEdD]+)", text, re.MULTILINE)
    if m is None:
        raise ValueError(f"Could not find '{key} = ...' in {deck_path}")
    return float(m.group(1).replace("d", "e").replace("D", "E"))


def read_namelist_bool(deck_path, key, default=False):
    """Same as read_namelist_float but for a Fortran logical ('.true.'/'.false.'/
    't'/'f', any case) -- returns `default` if the key isn't found at all."""
    text = open(deck_path).read()
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*\.?(true|false|t|f)\.?", text, re.MULTILINE | re.IGNORECASE)
    if m is None:
        return default
    return m.group(1).lower() in ("true", "t")


def get_dominant_component_t(run, quantity, kx_order, time_idxs, kx_target, mult_zed=None, rel_tol=1e-2):
    """Same phase-rotation extraction as verify_relaxation_stages_v2.py --
    a single-kx complex quantity is purely real or purely imaginary
    depending on IC convention; rotate by the phase at the max-|z| time and
    return whichever component actually carries the signal. Picks out the
    kx index nearest kx_target (rather than the largest-|kx| mode) so this
    stays consistent with the E_RH-selected kx even on a run with more than
    one nonzero kx."""
    z = np.zeros(len(time_idxs), dtype=complex)
    kx_val = None
    for i, idx in enumerate(time_idxs):
        f_kx_ky, kx, ky, t_eval = run.get_quantity_kx_ky(quantity=quantity, only_zonal=True, kx_order=kx_order, mult_zed=mult_zed, time_idx=idx)
        i_kx0 = np.argmin(np.abs(kx - kx_target))
        z[i] = f_kx_ky[i_kx0, 0]
        kx_val = kx[i_kx0]
    ref_idx = np.argmax(np.abs(z))
    theta_ref = np.angle(z[ref_idx])
    z_rot = z * np.exp(-1j * theta_ref)
    scale = np.max(np.abs(z_rot.real))
    residual = np.max(np.abs(z_rot.imag)) / scale if scale > 0 else np.inf
    if residual < rel_tol:
        return z_rot.real, kx_val, residual
    return np.abs(z), kx_val, residual


def get_rms_x_t(run, quantity, kx_order, time_idxs, time_avg, mult_zed=None):
    """RMS-over-x of a zonal (ky=0) real-space quantity vs time -- the
    --total-mode alternative to get_dominant_component_t's single-kx
    phase-rotated amplitude: this includes every kx harmonic in the box
    (via the inverse-FFT'd real-space profile from get_quantity_x_y), not
    just the dominant one, at the cost of losing sign/phase information
    (RMS is signless by construction). Time-averages FIRST (a time_avg-wide
    window centered/trailing per get_quantity_x_y's own convention) and
    only then takes the RMS in x -- not the other way around -- so fast
    bounce/transit-scale oscillations in x get smoothed out of the
    per-point value before the spatial RMS sees it, rather than being
    folded into the RMS as if they were part of the flow's amplitude."""
    rms = np.zeros(len(time_idxs))
    for i, idx in enumerate(time_idxs):
        f_x_y, x, y, t_eval = run.get_quantity_x_y(quantity=quantity, only_zonal=True, kx_order=kx_order, mult_zed=mult_zed, time_idx=idx, time_avg=time_avg)
        rms[i] = np.sqrt(np.mean(np.asarray(f_x_y) ** 2))
    return rms


print(f"Loading {RUN_PATH} ...")
run = StellaRun(RUN_PATH)
time = run.get_time_array()
n_time = len(time)
print(f"n_time={n_time}, t_max={time[-1]:.1f}")

NU = float(np.sum(run.ncdata.variables["vnew"][:]))
rhoc = read_namelist_float(f"{RUN_PATH}.in", "rhoc")
rmaj = read_namelist_float(f"{RUN_PATH}.in", "rmaj")
EPS = rhoc / rmaj
Q = read_namelist_float(f"{RUN_PATH}.in", "qinp")
print(f"nu={NU:.5g}  eps=rhoc/rmaj={EPS:.5g} (rhoc={rhoc:.4g}, rmaj={rmaj:.4g})  q={Q:.4g}")

IS_NONLINEAR = read_namelist_bool(f"{RUN_PATH}.in", "nonlinear")
print(f"nonlinear={IS_NONLINEAR}" + (" -- adding P_RH_NL breakdown to the P_RH panel" if IS_NONLINEAR else ""))

# --- E_RH (backbone: drives dominant-kx selection and, when available, the
# common frame count) ---
erh_result = safe("E_RH (RH_phi_I/RH_inertia)", run.get_E_RH_t_kx)
HAS_ERH = erh_result is not None
if HAS_ERH:
    E_RH_t_kx, time_erh, kx_vals = erh_result
    # Auto-select the kx with the largest E_RH, TIME-AVERAGED over the whole
    # run (not its peak value -- a transient spike in a non-dominant kx
    # shouldn't win) -- rather than assuming a single nonzero kx, or the
    # largest-|kx| mode -- and use that same kx consistently for every
    # kx-resolved quantity below (E_RH, P_RH_coll, upar/vE) -- the
    # (zed,vpa)/(vpa,mu) distribution-function panels are already summed over
    # kx on the stella side (not kx-resolved), so no selection applies there.
    i_kx_dom = int(np.argmax(E_RH_t_kx.mean(axis=0)))
    kx_dominant = float(kx_vals[i_kx_dom])
    print(f"Dominant kx (largest time-averaged E_RH): kx={kx_dominant:.4f} (index {i_kx_dom} of {len(kx_vals)})")
else:
    E_RH_t_kx = time_erh = kx_vals = None
    i_kx_dom = None
    kx_dominant = None
    print("Dominant-kx selection unavailable -- (vpa,mu) kx window and flow quantities fall back to all-kx/RMS.")

# --- (vpa,mu) panel's kx window. Explicit --kx_min/--kx_max always require
# the kx-resolved g2_vs_kxvpamus variable (error out rather than silently
# falling back, since that would render a value the user didn't ask for
# without any indication). Otherwise: --total, or a missing dominant kx,
# shows every kx (matching the total E_RH/P_RH framing); the default
# dominant-kx window also needs g2_vs_kxvpamus and falls back to "all kx"
# (with a warning) if it's missing -- harmless for a single-relevant-kx
# run, wrong for a genuine multi-kx one (see module docstring). ---
HAS_KXVPAMUS = "g2_vs_kxvpamus" in run.ncdata.variables
if args.kx_min is not None or args.kx_max is not None:
    if not HAS_KXVPAMUS:
        raise ValueError(
            "--kx_min/--kx_max given but this run's output has no 'g2_vs_kxvpamus' variable -- "
            "rerun the deck with write_g2_vs_kxvpamus=.true. first."
        )
    KX_MIN = args.kx_min if args.kx_min is not None else 0.0
    KX_MAX = args.kx_max if args.kx_max is not None else 1e20
    print(f"(vpa,mu) panel kx window (explicit): [{KX_MIN:.4g}, {KX_MAX:.4g}]")
elif args.total or kx_dominant is None:
    KX_MIN = KX_MAX = None
    print("(vpa,mu) panel: all kx" + (" (--total mode)" if args.total else " (no dominant kx available)"))
elif HAS_KXVPAMUS:
    KX_MIN = abs(kx_dominant) - 1e-6
    KX_MAX = abs(kx_dominant) + 1e-6
    print(f"(vpa,mu) panel kx window (dominant kx): [{KX_MIN:.4g}, {KX_MAX:.4g}]")
else:
    KX_MIN = KX_MAX = None
    print(
        "(vpa,mu) panel: no 'g2_vs_kxvpamus' in this run's output -- showing all kx summed "
        "(fine if this run only has one physically distinct nonzero kx; rerun with "
        "write_g2_vs_kxvpamus=.true. for a true per-kx view otherwise)"
    )

# --- P_RH_coll (depends on E_RH) ---
HAS_PRH = False
PRH_IDXS_KX = None if args.total else np.array([i_kx_dom])
if HAS_ERH:
    print("Fetching P_RH_coll ...")
    if args.total:
        E_RH_t = E_RH_t_kx.sum(axis=1)
        prh_result = safe("P_RH_coll", run.get_P_RH_coll_over_vnew_E_RH_t)
    else:
        E_RH_t = E_RH_t_kx[:, i_kx_dom]
        prh_result = safe("P_RH_coll", run.get_P_RH_coll_over_vnew_E_RH_t, idxs_kx=PRH_IDXS_KX)
    HAS_PRH = prh_result is not None
    if HAS_PRH:
        P_RH_over_ERH_t, time_pe = prh_result
    if args.total:
        # Not normalized to E_RH(0) in --total mode -- E_RH(0) is not a
        # particularly meaningful reference scale for the kx-summed total (it's
        # dominated by whichever single kx started with the largest amplitude,
        # an IC artifact rather than a property of the total).
        E_RH_plot = E_RH_t
    else:
        E_RH_plot = E_RH_t / E_RH_t[0]

    # Numerically-differentiated total rate -d(ln E_RH)/dt / nu, straight from
    # the E_RH(t) trajectory (no physics assumptions beyond finite
    # differencing) -- same quantity/smoothing as the dash-dot curve in
    # verify_relaxation_stages_v2.py's instantaneous-rate panel, overlaid here
    # on the P_RH_coll curve for the same comparison. Only needs E_RH, so
    # available even if P_RH_coll itself failed.
    dE_RH_dt = np.gradient(E_RH_t, time_erh)
    neg_dlnERH_dt_over_nu = -(dE_RH_dt / E_RH_t) / NU
    window_d = max(1, len(time_erh) // 60)
    smoothed_dlnE = np.convolve(neg_dlnERH_dt_over_nu, np.ones(window_d) / window_d, mode="same")

    if HAS_PRH:
        neg_P_RH_over_ERH_t = -P_RH_over_ERH_t
        window = max(1, len(time_pe) // 60)
        smoothed_p = np.convolve(neg_P_RH_over_ERH_t, np.ones(window) / window, mode="same")

        # --- P_RH_NL: the nonlinear (ExB, "phi") channel of the same RH-flux
        # decomposition P_RH_coll comes from (fcoll=1 there vs fphi=1 here) --
        # only meaningful (nonzero) for a genuinely nonlinear run, since the
        # ExB nonlinear term is absent from the gyrokinetic equation whenever
        # nonlinear=.false.. Normalized the same way as P_RH_coll (by nu*E_RH,
        # using the SAME E_RH_t/kx selection), so directly comparable/summable.
        HAS_PRH_NL = False
        if IS_NONLINEAR:
            print("Fetching P_RH_NL (nonlinear RH flux) ...")

            def _load_prh_nl():
                # E_RH_t already reflects this same PRH_IDXS_KX selection
                # (summed over kx) -- no need to re-fetch it. Its own time
                # array (time_erh) can be a couple of samples longer than
                # get_P_RH's (time_pe/time_p_nl, a shared prefix -- same
                # truncated-prefix pattern documented for time_erh/time_pe
                # elsewhere in this script) -- truncate to match before
                # dividing elementwise.
                even_t_kx, odd_t_kx, time_p_nl, _ = run.get_P_RH(idxs_kx=PRH_IDXS_KX, fphi=1, fapar=0, fbpar=0, fcoll=0)
                even_t = even_t_kx.sum(axis=1)
                odd_t = odd_t_kx.sum(axis=1)
                n = min(len(even_t), len(E_RH_t), len(time_pe))
                return -even_t[:n] / (NU * E_RH_t[:n]), -odd_t[:n] / (NU * E_RH_t[:n])

            prh_nl_result = safe("P_RH_NL (nonlinear RH flux)", _load_prh_nl)
            HAS_PRH_NL = prh_nl_result is not None
            if HAS_PRH_NL:
                neg_P_RH_nl_even_over_ERH_t, neg_P_RH_nl_odd_over_ERH_t = prh_nl_result
                n_nl = len(neg_P_RH_nl_even_over_ERH_t)
                time_pe_nl = time_pe[:n_nl]  # see the length-truncation note in _load_prh_nl above
                neg_P_RH_nl_total_over_ERH_t = neg_P_RH_nl_even_over_ERH_t + neg_P_RH_nl_odd_over_ERH_t
                neg_P_RH_total_over_ERH_t = neg_P_RH_over_ERH_t[:n_nl] + neg_P_RH_nl_total_over_ERH_t
                window_nl = max(1, n_nl // 60)
                smoothed_nl_even = np.convolve(neg_P_RH_nl_even_over_ERH_t, np.ones(window_nl) / window_nl, mode="same")
                smoothed_nl_odd = np.convolve(neg_P_RH_nl_odd_over_ERH_t, np.ones(window_nl) / window_nl, mode="same")
                smoothed_nl_total = np.convolve(neg_P_RH_nl_total_over_ERH_t, np.ones(window_nl) / window_nl, mode="same")
                smoothed_total = np.convolve(neg_P_RH_total_over_ERH_t, np.ones(window_nl) / window_nl, mode="same")

# --- Flow quantities (upar, upar*cos(theta), vE) -- dominant-kx
# phase-rotated by default, time-averaged-then-RMS-over-x in --total mode
# OR whenever no dominant kx is available (E_RH missing). ---
HAS_FLOW = True
use_rms = args.total or (kx_dominant is None)
if kx_dominant is None and not args.total:
    print("No dominant kx available -- flow quantities (upar/vE) fall back to RMS-in-x.")

# Default RMS time-averaging window: the run's own snapshot spacing, so the
# per-point value fed to the spatial RMS is smoothed over roughly one
# output interval rather than being an instantaneous snapshot -- a
# conservative default that removes sub-frame noise without washing out
# anything the movie's own frame cadence wouldn't already show.
RMS_TAVG = args.rms_tavg if args.rms_tavg is not None else float(np.median(np.diff(time)))
if use_rms:
    print(f"RMS-in-x flow quantities: time-averaging over a {RMS_TAVG:.3g} window before the spatial RMS")


# get_dominant_component_t/get_rms_x_t are the actual bottleneck (one
# run.get_quantity_kx_ky/get_quantity_x_y call PER time index, in a plain
# Python loop) -- unlike everything else in this script (E_RH/P_RH are
# single vectorized netCDF-slice reads, fast regardless of resolution),
# so --time_idx_step needs to thin THIS loop too, not just the animated
# frames, for it to actually speed anything up. Strided over the full
# n_time range (not cropped to i_min:i_max) to match the existing
# full-run convention for the phase-rotation reference time/dominant-kx
# selection -- --tmin/--tmax remain a display-only crop.
flow_time_idxs = np.arange(0, n_time, TIME_IDX_STEP)
time_flow = time[flow_time_idxs]
print(f"Fetching flow quantities (upar/vE): {len(flow_time_idxs)}/{n_time} time points (time_idx_step={TIME_IDX_STEP}) ...")


def _load_flow():
    if use_rms:
        # RMS-in-x, so signless -- no sign to rotate/lose, unlike the
        # dominant-kx path below.
        u = get_rms_x_t(run, "upar", 0, flow_time_idxs, RMS_TAVG)
        uc = get_rms_x_t(run, "upar", 0, flow_time_idxs, RMS_TAVG, mult_zed="cos")
        v = get_rms_x_t(run, "phi", 1, flow_time_idxs, RMS_TAVG)  # RMS(dphidx) == RMS(vE): sign is irrelevant to an RMS
        return u, uc, v
    u, kx_val, r1 = get_dominant_component_t(run, "upar", 0, flow_time_idxs, kx_dominant)
    uc, _, r2 = get_dominant_component_t(run, "upar", 0, flow_time_idxs, kx_dominant, mult_zed="cos")
    dphidx, _, r3 = get_dominant_component_t(run, "phi", 1, flow_time_idxs, kx_dominant)
    print(f"kx={kx_val:.3f}  phase-residuals upar={r1:.2g} uparcos={r2:.2g} dphidx={r3:.2g}")
    return u, uc, -dphidx


flow_result = safe("flow quantities (upar/vE)", _load_flow)
HAS_FLOW = flow_result is not None
if HAS_FLOW:
    upar_t, uparcos_t, vE_t = flow_result
    valid = np.abs(vE_t) > 1e-4 * np.max(np.abs(vE_t))
    toroidal_coef = np.full_like(vE_t, np.nan)
    ps_coef = np.full_like(vE_t, np.nan)
    # --total's (or the no-dominant-kx fallback's) inputs are RMS amplitudes
    # (always >=0), so the ratio drops the leading minus sign that makes the
    # dominant-kx panel match the signed neoclassical-convention ratio --
    # it's a magnitude-only comparison instead.
    coef_sign = 1 if use_rms else -1
    toroidal_coef[valid] = coef_sign * 2 * (EPS / Q) * upar_t[valid] / vE_t[valid]
    ps_coef[valid] = coef_sign * 2 * (1 / Q) * uparcos_t[valid] / vE_t[valid]

# get_E_RH_t_kx/get_P_RH_coll_over_vnew_E_RH_t return a time array that's a
# truncated PREFIX of the main 't' array (missing the last couple of
# samples) -- not the same length. The distribution-function snapshots
# (g2_vs_zvpas/g2_vs_vpamus) and upar/vE/dphidx are indexed against the
# full main time array, so the common frame range is bounded by the
# shorter E_RH/P_RH arrays when both are available; falls back to the full
# main time array otherwise.
if HAS_ERH and HAS_PRH:
    n_frames = min(len(time_erh), len(time_pe))
    assert np.allclose(time_erh[:n_frames], time[:n_frames]), "time_erh is not a prefix of the main time array"
    assert np.allclose(time_pe[:n_frames], time[:n_frames]), "time_pe is not a prefix of the main time array"
elif HAS_ERH:
    n_frames = len(time_erh)
    assert np.allclose(time_erh[:n_frames], time[:n_frames]), "time_erh is not a prefix of the main time array"
else:
    n_frames = n_time
print(f"n_frames={n_frames} (main time array has {n_time})")

HAS_ZVPA = "g2_vs_zvpas" in run.ncdata.variables
HAS_VMU = "g2_vs_vpamus" in run.ncdata.variables
if not HAS_ZVPA:
    print("WARNING: 'g2_vs_zvpas' not in this run's output -- (zed,vpa) panel unavailable (rerun with write_g2_vs_zvpas=.true. to enable it).")
elif run.ncdata.variables["g2_vs_zvpas"].shape[0] < n_frames:
    print("WARNING: 'g2_vs_zvpas' shorter than the common frame range -- (zed,vpa) panel unavailable.")
    HAS_ZVPA = False
if not HAS_VMU:
    print("WARNING: 'g2_vs_vpamus' not in this run's output -- (vpa,mu) panel unavailable (rerun with write_g2_vs_vpamus=.true. to enable it).")
elif run.ncdata.variables["g2_vs_vpamus"].shape[0] < n_frames:
    print("WARNING: 'g2_vs_vpamus' shorter than the common frame range -- (vpa,mu) panel unavailable.")
    HAS_VMU = False

# --tmin/--tmax crop the DISPLAYED window (animated frames + static-panel
# xlim) to a sub-range of the n_frames available -- everything computed
# above (dominant kx, RMS, phase-rotation reference time) still used the
# full run, only what gets plotted/animated is restricted here.
i_min = 0 if args.tmin is None else int(np.searchsorted(time[:n_frames], args.tmin))
i_max = n_frames - 1 if args.tmax is None else int(np.searchsorted(time[:n_frames], args.tmax, side="right")) - 1
i_min = max(0, min(i_min, n_frames - 1))
i_max = max(i_min, min(i_max, n_frames - 1))
xlim = (time[i_min], time[i_max])
print(f"Time window: t in [{time[i_min]:.2f}, {time[i_max]:.2f}] (frame indices {i_min}-{i_max} of {n_frames})")

# Per-frame (not global-fixed) vmax: |g|^2 decays by ~3 orders of
# magnitude over the run, so a single global vmax with the usual 2-decade
# log floor leaves the whole second half of the movie black. The 1D E_RH
# panel already carries the absolute-amplitude information; letting each
# frame re-normalize to its own max keeps the (zed,vpa)/(vpa,mu) *shape*
# visible throughout instead.

# --- figure layout ---
# The two distribution-function panels share v_parallel as their x-axis, so
# stack them (zed,vpa) on top of (vpa,mu) instead of placing them
# side-by-side -- makes the vpa-structure directly comparable between the
# two by eye, and sharex locks their horizontal extent/ticks together.
fig = plt.figure(figsize=(15, 10))
# Outer split: generous whitespace between the distribution column and the
# time-series column. Inner (left) split: the colorbar sits right next to
# its own contour plot, not spaced out to match the outer gap -- a single
# GridSpec's uniform wspace can't give the two gaps different widths, hence
# the nested GridSpecFromSubplotSpec.
outer = GridSpec(1, 2, figure=fig, width_ratios=[1.08, 1.3], wspace=0.35)
left = GridSpecFromSubplotSpec(4, 2, subplot_spec=outer[0], width_ratios=[1, 0.04], wspace=0.08, hspace=0.15)
right = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[1], hspace=0.15)

ax_zvpa = fig.add_subplot(left[0:2, 0])
cax_zvpa = fig.add_subplot(left[0:2, 1])
ax_vmu = fig.add_subplot(left[2:4, 0], sharex=ax_zvpa)
cax_vmu = fig.add_subplot(left[2:4, 1])

ax_erh = fig.add_subplot(right[0, 0])
ax_prh = fig.add_subplot(right[1, 0], sharex=ax_erh)
ax_tor = fig.add_subplot(right[2, 0], sharex=ax_erh)
ax_ps = fig.add_subplot(right[3, 0], sharex=ax_erh)

total_ylim = (0, 2) if args.total else UPAR_YLIM
sign_str = "" if use_rms else "-"


def window_ylim(*arrays, log=False, pad_frac=0.15):
    """Padded y-limits from data restricted to [i_min, i_max] -- unlike
    matplotlib's default autoscale (based on the FULL curve regardless of
    xlim), this rescales the y-axis to whatever --tmin/--tmax actually
    shows, so zooming into a sub-window doesn't leave the axis dominated by
    a value outside the visible range."""
    vals = np.concatenate([np.asarray(a)[i_min:i_max + 1] for a in arrays])
    vals = vals[np.isfinite(vals)]
    lo, hi = vals.min(), vals.max()
    if log:
        lo = max(lo, 1e-300)
        pad = (hi / lo) ** pad_frac if hi > lo else 1.5
        return lo / pad, hi * pad
    pad = pad_frac * (hi - lo) if hi > lo else pad_frac * max(abs(hi), 1.0)
    return lo - pad, hi + pad


if HAS_ERH:
    erh_label = r"$E_{RH}(t)$ (total)" if args.total else r"$E_{RH}(t)/E_{RH}(0)$"
    ax_erh.semilogy(time_erh, E_RH_plot, color="C0")
    ax_erh.set_ylabel(erh_label)
    ax_erh.set_xlim(*xlim)
    ax_erh.set_ylim(*window_ylim(E_RH_plot, log=True))
    ax_erh.grid(alpha=0.3)
else:
    mark_unavailable(ax_erh, "E_RH", xlim)
ax_erh.set_title(f"{NAME}: overview" + (" -- total/RMS mode" if args.total else
                  (f" -- dominant kx={kx_dominant:.3g}" if kx_dominant is not None else "")))

if HAS_ERH and HAS_PRH:
    ax_prh.plot(time_pe, smoothed_p, color="C1", label=r"$-P_{RH}^{coll}/(\nu E_{RH})$")
    prh_window_arrays = [smoothed_p, smoothed_dlnE]
    if HAS_PRH_NL:
        ax_prh.plot(time_pe_nl, smoothed_nl_even, color="red", label=r"$-P_{RH}^{NL,even}/(\nu E_{RH})$")
        ax_prh.plot(time_pe_nl, smoothed_nl_odd, color="blue", label=r"$-P_{RH}^{NL,odd}/(\nu E_{RH})$")
        ax_prh.plot(time_pe_nl, smoothed_nl_total, color="purple", label=r"$-P_{RH}^{NL}/(\nu E_{RH})$")
        ax_prh.plot(time_pe_nl, smoothed_total, color="black", label=r"$-(P_{RH}^{coll}+P_{RH}^{NL})/(\nu E_{RH})$")
        prh_window_arrays += [smoothed_nl_even, smoothed_nl_odd, smoothed_nl_total, smoothed_total]
    # Grey (not the coll curve's own color) so it reads as a cross-check
    # reference line rather than one more physical channel, now that the
    # nonlinear breakdown adds several colored curves alongside it.
    ax_prh.plot(time_erh, smoothed_dlnE, color="gray", ls="--", label=r"$-d\ln(E_{RH})/dt/\nu$")
    ax_prh.axhline(0, color="gray", lw=1)
    ax_prh.set_ylabel(r"$-P_{RH}^{coll}/(\nu E_{RH})$" + (" (total)" if args.total else ""))
    ax_prh.set_ylim(*window_ylim(*prh_window_arrays))
    ax_prh.set_xlim(*xlim)
    ax_prh.legend(fontsize=7)
    ax_prh.grid(alpha=0.3)
elif HAS_ERH:
    # E_RH available but P_RH_coll itself failed -- still show the
    # numerically-differentiated rate, which only needs E_RH.
    ax_prh.plot(time_erh, smoothed_dlnE, color="gray", ls="--", label=r"$-d\ln(E_{RH})/dt/\nu$")
    ax_prh.axhline(0, color="gray", lw=1)
    ax_prh.set_ylabel(r"$-d\ln(E_{RH})/dt/\nu$")
    ax_prh.set_ylim(*window_ylim(smoothed_dlnE))
    ax_prh.set_xlim(*xlim)
    ax_prh.legend(fontsize=8)
    ax_prh.grid(alpha=0.3)
else:
    mark_unavailable(ax_prh, "P_RH_coll", xlim)

if HAS_FLOW:
    ax_tor.plot(time_flow[valid], toroidal_coef[valid], color="C2")
    ax_tor.set_ylabel(rf"${sign_str}2(\epsilon/q)\overline{{u_\parallel}}/\overline{{v_E}}$" if use_rms
                       else r"$-2(\epsilon/q)\langle u_\parallel\rangle/\langle v_E\rangle$", fontsize=11)
    ax_tor.set_xlim(*xlim); ax_tor.set_ylim(*total_ylim)
    ax_tor.grid(alpha=0.3)

    ax_ps.plot(time_flow[valid], ps_coef[valid], color="C3")
    ax_ps.set_ylabel(rf"${sign_str}2(1/q)\overline{{u_\parallel\cos\theta}}/\overline{{v_E}}$" if use_rms
                      else r"$-2(1/q)\langle u_\parallel\cos\theta\rangle/\langle v_E\rangle$", fontsize=11)
    ax_ps.set_xlabel(r"$t v_{Ti}/a$")
    ax_ps.set_xlim(*xlim); ax_ps.set_ylim(*total_ylim)
    ax_ps.grid(alpha=0.3)
else:
    mark_unavailable(ax_tor, "toroidal-rotation coefficient (upar/vE)", xlim)
    mark_unavailable(ax_ps, "Pfirsch-Schlueter-like coefficient (upar*cos/vE)", xlim)
    ax_ps.set_xlabel(r"$t v_{Ti}/a$")

# The four time-series panels share the x-axis (t) -- only the bottom one
# needs its ticklabels/label, the rest are redundant.
for ax in (ax_erh, ax_prh, ax_tor):
    plt.setp(ax.get_xticklabels(), visible=False)

vlines = [ax.axvline(time[i_min], color="k", ls="--", lw=1.5) for ax in (ax_erh, ax_prh, ax_tor, ax_ps)]

# tight_layout doesn't support the nested GridSpecFromSubplotSpec + sharex
# combination above (warns and is a no-op) -- spacing is fully controlled
# by the explicit wspace/hspace on outer/left/right instead.

if not HAS_ZVPA:
    mark_unavailable(ax_zvpa, "(zed,vpa) distribution (g2_vs_zvpas)", (None, None))
    cax_zvpa.set_xticks([]); cax_zvpa.set_yticks([])
if not HAS_VMU:
    mark_unavailable(ax_vmu, "(vpa,mu) distribution (g2_vs_vpamus)", (None, None))
    cax_vmu.set_xticks([]); cax_vmu.set_yticks([])

frame_idxs = np.arange(i_min, i_max + 1, TIME_IDX_STEP)
print(f"Rendering {len(frame_idxs)} frames (time_idx_step={TIME_IDX_STEP}, out of {n_frames} available)")


def update(k):
    i = frame_idxs[k]
    artists = []

    if HAS_ZVPA:
        ax_zvpa.clear()
        _, _, im1 = plot_contour_gzvs(run, fig=fig, ax=ax_zvpa, time_idx=i, logarithmic=True)
        ax_zvpa.set_title("")
        # Shares its x-axis (vpa) with ax_vmu below it -- drop the redundant
        # per-panel xlabel/ticklabels, the bottom panel's carry it.
        ax_zvpa.set_xlabel("")
        plt.setp(ax_zvpa.get_xticklabels(), visible=False)
        ax_zvpa.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax_zvpa.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        cax_zvpa.cla()
        fig.colorbar(im1, cax=cax_zvpa, label=r"$|g|^2(\zeta,v_\parallel)$ (all $k_x$)")

    if HAS_VMU:
        ax_vmu.clear()
        _, _, im2 = plot_contour_gvmu_vpa(run, fig=fig, ax=ax_vmu, time_idx=i, logarithmic=True, kx_min=KX_MIN, kx_max=KX_MAX)
        cax_vmu.cla()
        vmu_kx_label = "all $k_x$" if KX_MIN is None else rf"$k_x={kx_dominant:.2g}$"
        fig.colorbar(im2, cax=cax_vmu, label=rf"$|g|^2(v_\parallel,\mu)$ ({vmu_kx_label})")

    t_now = time[i]
    for vl in vlines:
        vl.set_xdata([t_now, t_now])

    if k % 25 == 0:
        print(f"frame {k}/{len(frame_idxs)} (time_idx {i}/{n_frames})", end="\r")
    return artists


ani = animation.FuncAnimation(fig, update, frames=len(frame_idxs), blit=False)
out_name = f"{NAME}_movie_total.mp4" if args.total else f"{NAME}_movie.mp4"
ani.save(out_name, writer="ffmpeg", fps=FPS, dpi=120)
print(f"\nWrote {out_name}")
