"""Per-code (stella/GX/GS2) helpers.

NOTE: not every stella/GX/GS2 difference in this codebase is a pure
variable-name/path mapping -- some are genuine numerical differences
(e.g. GX applies a sqrt(2) factor to kx/ky, and has an old-vs-new
"GX_old_version" data layout). Those stay as explicit `if run.code ==
...` conditionals in grid.py/quantities rather than being folded into
a lookup table here, to avoid silently changing behavior.
"""
import re

import numpy as np


def get_nspecies(ncdata, default=1):
    """Number of species in the output, or `default` if the dimension is absent."""
    try:
        return len(ncdata.dimensions['species'])
    except Exception:
        return default


def get_species_label(ncdata, species_idx):
    """'i' for a positive-charge (ion) species, 'e' for negative-charge
    (electron) -- numbered ('i1', 'i2', ...) if there's more than one of
    either sign -- or 's<species_idx>' if charge information isn't
    available at all.
    """
    try:
        charge = ncdata.variables['charge'][:]
    except Exception:
        return "s%i" % species_idx

    if charge[species_idx] > 0:
        sign_idxs = list(np.where(charge > 0)[0])
        letter = "i"
    elif charge[species_idx] < 0:
        sign_idxs = list(np.where(charge < 0)[0])
        letter = "e"
    else:
        return "s%i" % species_idx

    if len(sign_idxs) == 1:
        return letter
    return "%s%i" % (letter, sign_idxs.index(species_idx) + 1)


_SPECIES_NAME_BY_LETTER = {"i": "Ions", "e": "Electrons"}


def get_species_name(ncdata, species_idx):
    """Human-readable species name for legend labels ('Ions'/'Electrons',
    numbered 'Ions 1'/'Ions 2' if there's more than one of either sign) --
    the readable counterpart to get_species_label's short math subscript,
    derived from it so the two never disagree.
    """
    match = re.match(r'^(i|e)(\d*)$', get_species_label(ncdata, species_idx))
    if not match:
        return "Species %i" % species_idx
    letter, num = match.groups()
    name = _SPECIES_NAME_BY_LETTER[letter]
    return "%s %s" % (name, num) if num else name


def get_reference_species_idx(ncdata):
    """Index of the species stella's reference values (v_ref, T_ref, ...)
    are conventionally tied to: the first ion (positive-charge) species
    if any is present, else the first electron species (an electron-only
    run with adiabatic ions), else species 0.
    """
    try:
        charge = ncdata.variables['charge'][:]
    except Exception:
        return 0

    ion_idxs = np.where(charge > 0)[0]
    if len(ion_idxs) > 0:
        return int(ion_idxs[0])
    electron_idxs = np.where(charge < 0)[0]
    if len(electron_idxs) > 0:
        return int(electron_idxs[0])
    return 0


def get_rho_label(ncdata, species_idx=None):
    """LaTeX math-mode gyroradius normalization label, e.g. r"\\rho_{i}",
    r"\\rho_{e}", r"\\rho_{i1}" -- species_idx=None (default) uses the
    reference species (get_reference_species_idx), so every plot that
    normalizes a length by "the" gyroradius does so consistently by
    default, instead of each call site separately hardcoding \\rho_i or
    dropping the species subscript entirely ($x/\\rho$). Always braces
    the subscript (matches get_vt_label's convention) so it's correct
    even for multi-digit species labels like "i1".
    """
    if species_idx is None:
        species_idx = get_reference_species_idx(ncdata)
    return r"\rho_{%s}" % get_species_label(ncdata, species_idx)


def get_vt_label(ncdata, species_idx=None):
    """LaTeX math-mode thermal-velocity normalization label, e.g.
    r"v_{Ti}", r"v_{Te}" -- same reference-species default and rationale
    as get_rho_label.
    """
    if species_idx is None:
        species_idx = get_reference_species_idx(ncdata)
    return r"v_{T%s}" % get_species_label(ncdata, species_idx)
