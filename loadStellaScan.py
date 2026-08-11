"""Backward-compatible shim: re-exports stella_diagnostics.scan under the
legacy top-level name so existing scripts that do
``import loadStellaScan as lSS; lSS.loadStellaScan(filenames, labels)``
keep working unmodified. New code should import
stella_diagnostics.scan.run_collection.RunCollection directly.
"""
from stella_diagnostics.scan.run_collection import RunCollection as loadStellaScan
from stella_diagnostics.scan.spectrum_scan import get_alpha_spectrum

__all__ = ["loadStellaScan", "get_alpha_spectrum"]
