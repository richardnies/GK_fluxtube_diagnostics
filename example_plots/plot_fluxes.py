import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})

import stellaDiagnostics as sD

filename_base  = "run_akyminmax-1.0000_nfield_periods-30.0000/precise_QA"

diagObj = sD.stellaDiagnostics(filename_base)
axs = diagObj.plot_flux_over_time()
axs[2].set_yscale('log')
plt.tight_layout()
plt.savefig("fig_fluxes.png")
