"""VMEC field-line-geometry setup for stella flux-tube scans.

Unrelated to the stella_diagnostics post-processing package: this is a
pre-processing tool that computes flux-tube placement parameters
(zeta_ctr, alpha0, nfield_periods) from a VMEC wout file, for
generating stella input decks -- it reads no stella output and shares
no code with stella_diagnostics.
"""
import numpy as np
import netCDF4 as nc4
import scipy.interpolate as interp
from scipy.optimize import root as root
from scipy.optimize import bisect as bisect


def get_zetactr0_alpha0_nfield_periods(tube_pos_val, Nturns_tube, vmec_wout, torflux, N_QS, M_QS, alpha_shift):

    ncdata = nc4.Dataset(vmec_wout,'r')
    Nfp  = ncdata['nfp'].getValue()
    ns   = ncdata['ns'].getValue()
    s_full = np.linspace(0, 1, ns, endpoint=True)
    s_half = 0.5*(s_full[1:] + s_full[:-1])
    mnmax = ncdata['mnmax'].getValue()
    mnmax_nyq = ncdata['mnmax_nyq'].getValue()

    iota_all = ncdata['iotaf']
    iota_interp = interp.interp1d(s_full, iota_all)
    iota = iota_interp(torflux)
    
    xm = ncdata['xm'][:]
    xn = ncdata['xn'][:]
    xm_nyq = ncdata['xm_nyq'][:]
    xn_nyq = ncdata['xn_nyq'][:]
    lmns_all = ncdata['lmns']
    bmnc_all = ncdata['bmnc']

    lmns = np.zeros(mnmax)
    for imn in range(mnmax):
        lmns_interp = interp.interp1d( s_half, lmns_all[1:,imn])
        lmns[imn] = lmns_interp(torflux)

    bmnc = np.zeros(mnmax_nyq)
    for imn in range(mnmax_nyq):
        bmnc_interp = interp.interp1d( s_half, bmnc_all[1:,imn])
        bmnc[imn] = bmnc_interp(torflux)

    # Get alpha corresponding to "tube_pos_val" (=1 for one field period shift)
    alpha_val = tube_pos_val*(2*np.pi/Nfp) * (N_QS/M_QS - iota)# + alpha_shift

    # Find global min and max of B
    Nr_theta = 500
    Nr_zeta  = 500
    thetas_tmp = np.linspace(0,2*np.pi,    Nr_theta)
    zetas_tmp  = np.linspace(0,2*np.pi/Nfp,Nr_zeta)
    thetas_2D, zetas_2D = np.meshgrid(thetas_tmp, zetas_tmp)
    B_global = 0
    for imn in range(mnmax_nyq):
        angles = xm_nyq[imn]*thetas_2D - xn_nyq[imn]*zetas_2D
        B_global = B_global + bmnc[imn]*np.cos(angles)

    B_global_min = B_global.min()
    B_global_max = B_global.max()

    # Function to minimise to find right zeta of B field extremum (dB/dtheta ~ 0)
    def get_B_der(zeta):

        # For given zeta, find theta_VMEC along field line
        theta_pest_target = alpha_val + iota*zeta

        # Find theta_VMEC corresponding to given theta_pest, from stella/geo/vmec_to_stella_geometry_interface.f90
        def residual_VMEC_PEST(theta_VMEC):
            fzero_residual = theta_VMEC - theta_pest_target
    
            for imn in range(mnmax):
                angle = xm[imn]*theta_VMEC - xn[imn]*zeta
                fzero_residual = fzero_residual + lmns[imn]*np.sin(angle)

            return fzero_residual

        theta_VMEC = root(residual_VMEC_PEST, x0=theta_pest_target).x[0]
#
        # Evaluate B-derivative
        B_der = 0
        for imn in range(mnmax_nyq):
            angle = xm_nyq[imn]*theta_VMEC - xn_nyq[imn]*zeta
            B_der = B_der + bmnc[imn]*np.sin(angle)*xm_nyq[imn]

        return B_der

#    # Function to minimise to find right zeta for desired B value
#    def residual_B_desired(zeta, B_desired):
#
#        # For given zeta, find theta_VMEC along field line
#        theta_pest_target = alpha_val + iota*zeta
#
#        # Find theta_VMEC corresponding to given theta_pest, from stella/geo/vmec_to_stella_geometry_interface.f90
#        def residual_VMEC_PEST(theta_VMEC):
#            fzero_residual = theta_VMEC - theta_pest_target
#    
#            for imn in range(mnmax):
#                angle = xm[imn]*theta_VMEC - xn[imn]*zeta
#                fzero_residual = fzero_residual + lmns[imn]*np.sin(angle)
#
#            return fzero_residual
#
#        #theta_VMEC = theta_pest
#        theta_VMEC = root(residual_VMEC_PEST, x0=theta_pest_target).x[0]
#        #theta_VMEC = bisect(residual_VMEC_PEST, a=theta_pest_target*0.5, b=theta_pest_target*1.5)
#
##        # Check
##        theta_pest_check = theta_VMEC
##        for imn in range(mnmax):
##            angle = xm[imn]*theta_VMEC - xn[imn]*zeta
##            theta_pest_check = theta_pest_check + lmns[imn]*np.sin(angle)
##
##        print("theta_pest(theta_VMEC) - theta_pest_target = %e" % (theta_pest_check-theta_pest_target))
##
#        # Evaluate B
#        B_val = 0
#        for imn in range(mnmax_nyq):
#            angle = xm_nyq[imn]*theta_VMEC - xn_nyq[imn]*zeta
#            B_val = B_val + bmnc[imn]*np.cos(angle)
#
#        return B_val - B_desired

    # Find zeta for given alpha that gives B at tube centre (=Bmin or Bmax depending on odd or even number of turns) and B at tube end
    if Nturns_tube % 2 == 1:
        #zeta_ctr = bisect(residual_B_desired, a=0.7*zeta_ctr_guess, b=1.3*zeta_ctr_guess, args=(B_global_min))

        zeta_beg_guess = (-Nturns_tube*np.pi-alpha_val)/(iota-N_QS/M_QS)
        #zeta_beg = root(residual_B_desired, zeta_beg_guess, args=(B_global_max)).x[0]

        zeta_end_guess = ( Nturns_tube*np.pi-alpha_val)/(iota-N_QS/M_QS)
        #zeta_end = root(residual_B_desired, zeta_end_guess, args=(B_global_max)).x[0]

        #print("residual B_desired at beg    = %e" % (residual_B_desired(zeta_beg, B_global_max)))
        #print("residual B_desired at centre = %e" % (residual_B_desired(zeta_ctr, B_global_min)))
        #print("residual B_desired at end    = %e" % (residual_B_desired(zeta_end, B_global_max)))

    else:
    #    zeta_ctr = root(residual_B_desired, zeta_ctr_guess, args=(B_global_max)).x[0]
        zeta_beg_guess = ((-Nturns_tube+1)*np.pi-alpha_val)/(iota-N_QS/M_QS)
        #zeta_beg = root(residual_B_desired, zeta_beg_guess, args=(B_global_max)).x[0]

        zeta_end_guess = (( Nturns_tube+1)*np.pi-alpha_val)/(iota-N_QS/M_QS)
        #zeta_end = root(residual_B_desired, zeta_end_guess, args=(B_global_max)).x[0]

    zeta_beg = bisect(get_B_der, a=0.9*zeta_beg_guess, b=1.1*zeta_beg_guess)
    zeta_end = bisect(get_B_der, a=0.9*zeta_end_guess, b=1.1*zeta_end_guess)


    #print("B max    = %e" % (B_global_max))
    #print("B min    = %e" % (B_global_min))

    zeta_ctr = 0.5*(zeta_beg + zeta_end)

    # Get length of tube from zeta_ctr, zeta_end, and desired number of turns
    Delta_zeta = np.abs(zeta_end - zeta_ctr)*2
    nfield_periods  = np.abs(Delta_zeta / (2*np.pi/Nfp))

    return zeta_ctr, alpha_val-alpha_shift, nfield_periods
