import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})

import stellaDiagnostics as sD

base_dir = "theta-extent-2pi_kinetic_teprim-1"
filename  = base_dir+"/precise_QA_NL"
StellaObj = sD.stellaDiagnostics(filename)

fig, ax, im = StellaObj.plot_parallel_correlation_function()

plt.tight_layout()
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(im, cax=cbar_ax)
#plt.subplots_adjust(hspace=0)
plt.savefig("fig_correlation_func.png", dpi=800)
