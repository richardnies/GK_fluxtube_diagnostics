"""Config for plot_contour_phi_vs_t_zed.py -- run with:
    python ../example_plots/plot_contour_phi_vs_t_zed.py scan_contour_phi_vs_t_zed_config.py

NOTE: always fails with AttributeError -- confirmed pre-existing bug
(plot_contour_phi_zed_t has never existed on StellaRun), see README
"Known issues". Included for completeness/documentation, not because it
can succeed. akyminmax_vals holds two dummy directory-suffix strings
(not real aky values) purely to point at this scan's two real runs.
"""
base_dirs = ["."]
base_dir_titles = ["scan"]
akyminmax_vals = ["4.2000", "6.7000"]
tprim_val = ""
filename_template = "%s/run_tprim-%s%s/example"
