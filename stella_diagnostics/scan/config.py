"""Loading scan-comparison definitions (which run directories, labels,
colors, axis limits) as small standalone Python files, kept separate from
the analysis/plotting code that consumes them.

This exists so that adding a new comparison means writing a short data-only
config file, not copying an entire plotting script -- and so that improving
the shared analysis code (in stella_diagnostics) automatically benefits every
existing config, instead of requiring the same fix to be re-applied to every
copy of a script scattered across scan directories.
"""

import importlib.util
import re
from glob import glob
from pathlib import Path


def load_scan_config(path, required=("dirnames",)):
    """Dynamically import the .py file at `path` as a module and return it.

    Callers read whatever attributes they need off the returned module
    (typically via ``getattr(config, "labels", None)`` for optional
    fields), so a config file only needs to define the fields relevant to
    the driver it's meant for. Raises ValueError up front, naming exactly
    which required attributes are missing, rather than an opaque
    AttributeError the first time a driver happens to touch a missing
    field.
    """
    path = Path(path)
    module_name = "stella_scan_config_" + re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ValueError(f"scan config {path} is missing required field(s): {', '.join(missing)}")

    return module


def discover_runs(base_dir, pattern="run_*", param_regex=None, exclude=None, sort=True):
    """Glob `base_dir` for subdirectories matching `pattern` (generalizes
    the inline glob in stella_diagnostics.scan.spectrum_scan).

    If `param_regex` is given (e.g. r"tprim-([0-9.eE+-]+)"), it must match
    every matched directory name and capture the scan-parameter value in
    its first group; returns a list of (dirname, param_value) tuples,
    sorted by param_value unless sort=False. A directory that matches
    `pattern` but not `param_regex` raises a clear ValueError naming it,
    rather than being silently skipped.

    If `param_regex` is None, returns a plain list of dirnames (sorted
    alphabetically unless sort=False).

    `exclude`: a substring, or list of substrings; any matched directory
    whose name contains one of them is dropped (matches
    spectrum_scan.py's rundir_str_exclude).
    """
    matches = sorted(glob(str(Path(base_dir) / pattern)))

    if exclude is not None:
        exclude_list = [exclude] if isinstance(exclude, str) else list(exclude)
        matches = [m for m in matches if not any(e in m for e in exclude_list)]

    if param_regex is None:
        return matches

    compiled = re.compile(param_regex)
    results = []
    for dirname in matches:
        name = Path(dirname).name
        m = compiled.search(name)
        if m is None:
            raise ValueError(
                f"directory {dirname!r} matches pattern {pattern!r} but its name doesn't "
                f"match param_regex {param_regex!r}"
            )
        results.append((dirname, float(m.group(1))))

    if sort:
        results.sort(key=lambda pair: pair[1])
    return results
