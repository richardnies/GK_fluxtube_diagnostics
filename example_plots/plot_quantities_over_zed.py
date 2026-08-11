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
import stellaDiagnostics as sD

filename = "run_akyminmax-1.0000_nfield_periods-100.0000/precise_QA"

dataObj = sD.stellaDiagnostics(filename)

kwargs = {  'plot_phi'       : True,
            'plot_B'         : True,
            'plot_Gamma0'    : True,
            'plot_omega_s_k' : True,
            'norm_factor_omega_s_k' : 1}

fig, ax = dataObj.plot_quantities_over_zed(**kwargs)

ax.legend()
#ax.set_xlim([-0.8*np.pi/2 * 139/234 * 20/10, -0.5*np.pi/2 * 139/234 * 20/10])
ax.set_ylim([-1,2.5])
ax.grid()

plt.savefig("fig_quantities_over_zed.png")
