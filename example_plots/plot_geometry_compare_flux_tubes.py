import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24, 
    "axes.titlepad": 15,
})

import loadStellaScan as lSS

filename_list = ["run_akyminmax-0.3000_nfield_periods-200.0000/precise_QA"]

label_list = [None]

data_obj = lSS.loadStellaScan(filename_list, label_list)

axs = data_obj.plot_comparison_flux_tube_geometry(zed_times_nfield_periods=True)

## Limit zeta to a few periods
#divide_zeta = 1
#for i in range(3):
#    for j in range(4):
#        axs[i][j].set_xlim([-np.pi/divide_zeta, np.pi/divide_zeta])

#for ax in axs:
#    print(ax)
#    ax.set_xlim([-np.pi/divide_zeta, np.pi/divide_zeta])

plt.savefig("fig_comparison_flux_tube_geometry.png")
