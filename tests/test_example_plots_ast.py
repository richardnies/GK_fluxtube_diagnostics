"""Statically verifies that every method call example_plots/*.py makes on
a stellaDiagnostics/loadStellaScan object resolves on the real classes,
without needing actual STELLA netCDF output to run the scripts."""
import ast
from pathlib import Path

import pytest

from stella_diagnostics.io.run import StellaRun
from stella_diagnostics.scan.run_collection import RunCollection

REPO = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO / "example_plots"

# Pre-existing bug (predates this restructure): plot_contour_phi_zed_t has
# never existed on stellaDiagnostics/StellaRun. Flagged, not fixed -- see
# README "Known issues".
KNOWN_BROKEN = {
    "plot_contour_phi_vs_t_zed.py": {"plot_contour_phi_zed_t"},
}


def _unresolved_attrs(path):
    tree = ast.parse(path.read_text())
    class_of_var = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute):
                if call.func.attr == "stellaDiagnostics":
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            class_of_var[t.id] = StellaRun
                elif call.func.attr == "loadStellaScan":
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            class_of_var[t.id] = RunCollection

    unresolved = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            cls = class_of_var.get(node.value.id)
            if cls is not None and not hasattr(cls, node.attr):
                unresolved.add(node.attr)
    return unresolved


@pytest.mark.parametrize("path", sorted(EXAMPLES_DIR.glob("*.py")), ids=lambda p: p.name)
def test_example_script_method_calls_resolve(path):
    unresolved = _unresolved_attrs(path)
    expected_broken = KNOWN_BROKEN.get(path.name, set())
    assert unresolved == expected_broken, (
        f"{path.name}: unresolved attribute accesses {unresolved} "
        f"do not match the expected known-broken set {expected_broken}"
    )
