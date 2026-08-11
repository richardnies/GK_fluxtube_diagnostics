import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})

# Stella diagnostics directory
dir_stella_diagnostics = '/home/rnies/stella_diagnostics'
sys.path.append(dir_stella_diagnostics)
import loadStellaScan as lSS

akyminmax_vals      = [0.1]
#akyminmax_vals      = [0.3]
#akyminmax_vals      = [1.0]
#nfield_periods_vals = [20,50,70,100,150,200,250,300]
nfield_periods_vals = [200,250,300, 400]
#nfield_periods_vals = [20,50,70,100,150,200,250,300, 400,500,600,700,800]

filenames = []
labels    = []
for i_aky, aky_val in enumerate(akyminmax_vals):
    for i_nfp, nfp_val in enumerate(nfield_periods_vals):
        filenames.append("run_akyminmax-%.4f_" % (aky_val) + "nfield_periods-%.4f/precise_QA" % (nfp_val))
        labels.append(r"nfp = %.1f" % (nfp_val))
        #labels.append(r"$k_y \rho_i =$ %.1f, nfp = %.1f" % (aky_val, nfp_val))

Scan = lSS.loadStellaScan(filenames, labels)
Scan.plot_phi_vs_zed(zed_times_nfield_periods=True)
#plt.xlim([-200,200])
#plt.xlim([-600,600])
plt.xlabel(r"$\zeta$")
plt.grid()
plt.legend()
plt.savefig("fig_compare_phi_zed.png")

