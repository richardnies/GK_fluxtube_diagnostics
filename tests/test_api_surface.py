"""Verifies the public API surface (method names, parameter names, order,
and defaults) on stellaDiagnostics/loadStellaScan is byte-for-byte
identical to dev/api_baseline_pre_refactor.txt, captured before the
restructure. This is the direct, data-free enforcement of the
requirement that any existing script calling these classes keeps
working unmodified.
"""
import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "dev" / "api_baseline_pre_refactor.txt"


def _dump(cls, name):
    lines = [f"=== {name} ==="]
    methods = [m for m in dir(cls) if not m.startswith("_") or m == "__init__"]
    for m in sorted(methods):
        attr = getattr(cls, m)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
            except (ValueError, TypeError):
                sig = "(unknown signature)"
            lines.append(f"{m}{sig}")
    return "\n".join(lines)


def test_stelladiagnostics_api_surface_unchanged():
    from stella_diagnostics.io.run import StellaRun

    baseline = BASELINE.read_text()
    old_block = baseline.split("=== loadStellaScan ===")[0].strip()
    new_block = _dump(StellaRun, "stellaDiagnostics").strip()
    assert new_block == old_block


def test_loadstellascan_api_surface_unchanged():
    from stella_diagnostics.scan.run_collection import RunCollection

    baseline = BASELINE.read_text()
    marker = "=== loadStellaScan ==="
    old_block = (marker + baseline.split(marker, 1)[1]).strip()
    new_block = _dump(RunCollection, "loadStellaScan").strip()
    assert new_block == old_block
