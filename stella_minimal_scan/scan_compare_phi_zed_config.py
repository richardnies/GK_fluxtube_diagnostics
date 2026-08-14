"""Config for plot_compare_phi_zed.py -- adapted for this scan's naming
("run_tprim-<val>" instead of "run_akyminmax-<aky>_nfield_periods-<nfp>").
akyminmax_vals is a single dummy value consumed by the unused %s slot;
nfield_periods_vals actually holds the tprim sweep here. Run with:
    python ../example_plots/plot_compare_phi_zed.py scan_compare_phi_zed_config.py

NOTE: fails here with FileNotFoundError -- this driver hardcodes
zed_times_nfield_periods=True, which requires a "<run>.vmec.geo" file;
this run only has Miller geometry ("<run>.geometry"). A data-availability
limitation of this minimal example (Miller, not VMEC), not a bug: the
same driver, called the same way, works fine against a VMEC-geometry run.
"""
akyminmax_vals = [""]
nfield_periods_vals = [4.2, 6.7]
filename_template = "run_tprim-%s%.4f/example"
label_template = r"$R/L_T=%.1f$"
figname = "fig_compare_phi_zed.png"
