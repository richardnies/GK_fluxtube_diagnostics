"""Per-code (stella/GX/GS2) helpers.

NOTE: not every stella/GX/GS2 difference in this codebase is a pure
variable-name/path mapping -- some are genuine numerical differences
(e.g. GX applies a sqrt(2) factor to kx/ky, and has an old-vs-new
"GX_old_version" data layout). Those stay as explicit `if run.code ==
...` conditionals in grid.py/quantities rather than being folded into
a lookup table here, to avoid silently changing behavior.
"""


def get_nspecies(ncdata, default=1):
    """Number of species in the output, or `default` if the dimension is absent."""
    try:
        return len(ncdata.dimensions['species'])
    except Exception:
        return default
