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
#nfield_periods_vals = [70,100,150,200,250,300]
nfield_periods_vals = [20,50,70,100,150,200,250,300, 400,500,600,700,800]
#nfield_periods_vals = [70,100]#,150,200,250,300]

fig, axs = plt.subplots(nrows=2,ncols=1, figsize=(14,9))

for i_nfield_periods, nfield_periods_val in enumerate(nfield_periods_vals):
    filenames = []
    labels    = []
    for i_aky, aky_val in enumerate(akyminmax_vals):
        filenames.append("run_akyminmax-%.4f_" % (aky_val) + "nfield_periods-%.4f/precise_QA" % (nfield_periods_val))
        labels.append(None)
        #labels.append(r"$N_t= $ %.1f" % (nfield_periods_val))

    scanObj = lSS.loadStellaScan(filenames, labels)
    scanObj.plot_omega_ky(axs=axs, label=r"$Nfp = $%i" % (nfield_periods_val))

axs[0].legend()
plt.savefig("fig_comparison_growth_rates.png")
