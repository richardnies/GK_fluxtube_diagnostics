"""Collisional Rosenbluth-Hinton P_RH(kx), normalized by nu_ii*E_RH.

Extracted from example_plots/plot_RH_P_C_kx_from_file.py -- reads
precomputed "data_ERH_mean_kx.dat"/"data_P_RH_coll_even_mean_kx.dat"/
"data_P_RH_coll_odd_mean_kx.dat" files (generated elsewhere) and combines/
normalizes them.
"""

import numpy as np


def get_P_RH_coll_normalized_vs_kx(run, vnew, eps=0.18):
    """(kx, P_RH_coll_mean_kx_norm) for one run, or (None, None) if the
    collisional P_RH is negligible everywhere (matches the original
    script's skip-if-below-threshold behavior).
    """
    dirname = run.filename_base.rsplit("/", 1)[0]
    E_RH_mean_kx = np.loadtxt(dirname + "/data_ERH_mean_kx.dat")
    P_RH_coll_even_mean_kx = np.loadtxt(dirname + "/data_P_RH_coll_even_mean_kx.dat")
    P_RH_coll_odd_mean_kx = np.loadtxt(dirname + "/data_P_RH_coll_odd_mean_kx.dat")

    P_RH_coll_mean_kx = P_RH_coll_even_mean_kx + P_RH_coll_odd_mean_kx
    if np.sum(np.abs(P_RH_coll_mean_kx)) < 1e-14:
        return None, None

    P_RH_coll_mean_kx_norm = P_RH_coll_mean_kx / (vnew * E_RH_mean_kx) * eps**2
    kx = run.ncdata["kx"][:]
    return kx, P_RH_coll_mean_kx_norm
