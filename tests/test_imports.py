"""Every module in the restructured package, plus both legacy
compatibility shims, must import cleanly with no ImportError/NameError."""
import importlib

import pytest

SUBMODULES = [
    "stella_diagnostics",
    "stella_diagnostics.grid",
    "stella_diagnostics.io.codes",
    "stella_diagnostics.io.run",
    "stella_diagnostics.quantities.labels",
    "stella_diagnostics.quantities.registry",
    "stella_diagnostics.quantities.realspace",
    "stella_diagnostics.physics.correlations",
    "stella_diagnostics.physics.fluxes",
    "stella_diagnostics.physics.rosenbluth_hinton",
    "stella_diagnostics.physics.velocity_space",
    "stella_diagnostics.physics.zonal_energy",
    "stella_diagnostics.spectral.fft",
    "stella_diagnostics.spectral.omega",
    "stella_diagnostics.spectral.stats",
    "stella_diagnostics.plotting.mpl_helpers",
    "stella_diagnostics.plotting.flux_plots",
    "stella_diagnostics.plotting.zed_plots",
    "stella_diagnostics.plotting.realspace_plots",
    "stella_diagnostics.plotting.kspace_plots",
    "stella_diagnostics.scan.run_collection",
    "stella_diagnostics.scan.omega_scan",
    "stella_diagnostics.scan.spectrum_scan",
]

LEGACY_SHIMS = ["stellaDiagnostics", "loadStellaScan", "setupStellaScan"]


@pytest.mark.parametrize("modname", SUBMODULES)
def test_submodule_imports(modname):
    importlib.import_module(modname)


@pytest.mark.parametrize("modname", LEGACY_SHIMS)
def test_legacy_shim_imports(modname):
    importlib.import_module(modname)


def test_legacy_names_point_at_new_classes():
    import stellaDiagnostics
    import loadStellaScan
    from stella_diagnostics.io.run import StellaRun
    from stella_diagnostics.scan.run_collection import RunCollection

    assert stellaDiagnostics.stellaDiagnostics is StellaRun
    assert loadStellaScan.loadStellaScan is RunCollection
