"""Generic stella restart-file / namelist I/O: parsing the input namelist,
reconstructing the physical (kx, zed, vpa, mu) grids and the kxkyz_lo MPI
domain decomposition from it, and reading g(vpa, mu) rows out of the
per-processor restart files.

Extracted from example_plots/plot_zonal_distribution.py -- the only script
in this codebase that reads restart files directly (no StellaRun/netCDF
output-file involvement), so this is genuinely new territory rather than a
refactor of existing code. Not integrated with stella_diagnostics.io.cache:
that cache's fingerprinting is StellaRun-shaped (keyed off
filename_base/netcdf_file), and a restart-file-aware fingerprint strategy
would need to be added there first; noted as a possible follow-up rather
than force-fitted here.
"""

import glob
import os
import re

import netCDF4
import numpy as np
from scipy.special import roots_laguerre

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
