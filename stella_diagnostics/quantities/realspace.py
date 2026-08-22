"""Real-space (zed, x, y) quantity dispatch.

NOTE: get_quantity_zed_x_y independently re-implements the same
~40-branch quantity dispatch as quantities/registry.get_quantity_zed_kx_ky
instead of calling it and FFT-transforming the result (get_fft_real_space
already exists in spectral/fft.py for exactly this purpose but is not used
here). This duplication was intentionally NOT merged during the restructure
since doing so would change a live numerical code path; flagged here for a
future, carefully-tested cleanup.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.special as specialfunc
from scipy.interpolate import interp1d as interp
from scipy.interpolate import RegularGridInterpolator as interp2D
from scipy import integrate
from scipy.interpolate import interpn
from scipy.signal import argrelextrema
import seaborn as sns
from glob import glob
from os.path import exists
from stella_diagnostics.spectral.fft import get_fft_real_space
from stella_diagnostics.spectral.stats import dt_weighted_mean, dt_weights
from stella_diagnostics.grid import nearest_index
from stella_diagnostics.io.codes import get_nspecies


def get_quantity_zed_x_y(run, quantity, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, kx_lowpass_cutoff=np.inf, ky_lowpass_cutoff=np.inf, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1, abs_squared=False, quantity_mult=None):

    if remove_zonal and only_zonal:
        print("WARNING! Both only_zonal and remove_zonal were set to True, will thus return f_x_y = 0.")

    kx, ky, zed = run.get_kx_ky_zed()
    time_all    = run.get_time_array(GX_big=True)
    dl_over_B_avg = run.dl_over_B_avg()

    if time_val is not None:
        time_idx = np.argmin( np.abs(time_all-time_val) )

    if time_avg is not None and np.ndim(time_idx) == 0:
        # Full-width (time_avg) window centered on time_eval, SHIFTED (not
        # merely clamped/shrunk) when that would run past either edge of
        # the run's time range -- e.g. time_eval=time_all[-1] (the
        # implicit reference when no explicit time_val is given) shifts
        # to a full-width trailing window [time_all[-1]-time_avg,
        # time_all[-1]], matching the trailing-window convention used by
        # stella_diagnostics.scan.zonal_flow_scan.get_growth_rate_from_flux
        # and physics.velocity_space.plot_contour_gvmu_vpa. A naive clamp
        # (previous behavior) silently produced a half-width window here
        # instead -- confirmed to have affected every zonal_flow_scan
        # profile function's default (time_val_avg=None) case, which
        # reaches this same time_avg handling via get_quantity_x_y.
        #
        # Only resolved when time_idx is still a single reference point:
        # several composite quantities (e.g. "dyphi-T") recurse into this
        # same function with time_idx ALREADY set to an outer call's
        # resolved window array plus time_avg forwarded again -- in that
        # case there is no new reference point to resolve a window
        # around (time_all[time_idx] would already be an array), so skip
        # straight to using the given time_idx array as the window and
        # deriving its own dt_avg_vals below (confirmed pre-existing
        # crash risk in this recursive pattern, affecting ~40 composite
        # quantity branches, not something specific to any one of them).
        time_eval = time_all[time_idx]
        time_min = time_eval - time_avg/2
        time_max = time_eval + time_avg/2
        if time_min < 0:
            time_max += -time_min
            time_min = 0
        if time_max > time_all[-1]:
            time_min -= (time_max - time_all[-1])
            time_max = time_all[-1]
        time_min = max(0, time_min)
        time_idx_min = np.argmin( np.abs(time_all-time_min) )
        time_idx_max = np.argmin( np.abs(time_all-time_max) )
        time_idx = np.arange(time_idx_min,time_idx_max+1)

    time_eval = time_all[time_idx]

    # dt_avg_vals is needed whenever time_idx is array-valued (there's a
    # window to weight-average over), not only when time_avg is set in
    # THIS call -- composite quantities that recurse into this function
    # forwarding time_idx but not time_avg (e.g. Pi_RH_NL calling
    # Pi_RH_even/Pi_RH_odd) still have an array-valued time_idx here even
    # though time_avg is None for the recursive call itself.
    if np.ndim(time_idx) > 0:
        dt_avg_vals = dt_weights(np.atleast_1d(time_eval))
        
    if quantity=="phi":
        # phi_vs_t(t, tube, zed, theta0, ky, ri)
        if run.code == "stella":
            f_zed_kx_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
        elif run.code == "GS2":
            f_kx_ky_ri = np.transpose( run.ncdata.variables['phi_igomega_by_mode'][time_idx] , axes=[1,0,2] )
            f_zed_kx_ky_ri = np.tile(f_kx_ky_ri[None,:,:,:].T, len(zed)).T
        elif run.code == "GX":
            if run.GX_old_version:
                print("WARNING! You are loading Phi_z in GX, which had issues in early code versions.")
                f_zed_kx_ky_ri = np.transpose( run.ncdata['Special']['Phi_z'], axes=[2,1,0,3] )
            else:
                f_zed_kx_ky_ri = np.transpose( run.ncdata_big['Diagnostics']['Phi'][time_idx] , axes=[2,1,0,3] )

    elif quantity=="(1-Gamma0)phi":
        kperp2_zed_kx_ky = run.ncdata.variables['kperp2'][:][:,species_idx,:,:] 
        Gamma0 = specialfunc.iv(0, kperp2_zed_kx_ky/2) * np.exp(-kperp2_zed_kx_ky/2)
        f_zed_kx_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]*(1-Gamma0)[:,:,:,None]
 
    elif quantity[:3] == "RH_":
        if quantity=="RH_phi_I" or quantity=="RH_phi" or quantity == "RHnon_phi":
            f_zed_kx_ri = run.ncdata.variables['RH_phi_I'][time_idx,species_idx,0,:,:,:]
        elif quantity == "RH_fluxes_collisional":
            f_zed_kx_ri = run.ncdata.variables[quantity][time_idx,species_idx,0,:,:,:]
#            elif quantity=="RH_phi":
#                RH_phi_I_zed_kx_ri = run.ncdata.variables["RH_phi_I"][time_idx,species_idx,0,:,:,:]
#                RH_inertia_zed_kx = run.ncdata.variables['RH_inertia'][species_idx,0,:,:,0]
#                f_zed_kx_ri = RH_phi_I_zed_kx_ri / RH_inertia_zed_kx[:,:,None]
        else:
            # Quantity must be one of the nonlinear RH fluxes
            if species_idx == "sum":
                nspecies = get_nspecies(run.ncdata)
                RH_flux_zed_kx_ky_ri = run.ncdata.variables[quantity][time_idx,0,0,:,:,:,:]
                for i_spec in np.arange(nspecies-1):
                    RH_flux_zed_kx_ky_ri += run.ncdata.variables[quantity][time_idx,i_spec+1,0,:,:,:,:]

            else:
                RH_flux_zed_kx_ky_ri = run.ncdata.variables[quantity][time_idx,species_idx,0,:,:,:,:]
            # Sum over ky. axis_ky depends on whether time_idx (already
            # resolved above) is a scalar (single-snapshot shape) or an
            # array (extra leading time axis) -- NOT on time_avg, which
            # is only a proxy for that and was wrong for composite
            # quantities (e.g. Pi_RH_even/Pi_RH_odd/Pi_RH_NL) recursing
            # into this function with time_idx already array-valued from
            # an outer time_avg-engaged call but without forwarding
            # time_avg themselves.
            axis_ky = 2 if np.ndim(time_idx) == 0 else 3
            f_zed_kx_ri = np.sum(RH_flux_zed_kx_ky_ri, axis=axis_ky)


        ky = run.ncdata['ky'][:]

        # Same scalar-vs-array time_idx distinction as axis_ky above (not
        # time_avg -- see the comment there).
        if np.ndim(time_idx) == 0:
            f_zed_kx_ky_ri = np.zeros( (np.shape(f_zed_kx_ri)[0], np.shape(f_zed_kx_ri)[1], len(ky), 2) )
            f_zed_kx_ky_ri[:,:,0,:] = f_zed_kx_ri

            if quantity == "RH_phi" or quantity == "RHnon_phi":
                # Divide by RH inertia
                RH_inertia_zed_kx = run.ncdata.variables['RH_inertia'][species_idx,0,:,:,0]
                dl_over_B_avg = run.dl_over_B_avg()
                RH_inertia_kx = np.sum(dl_over_B_avg[:,None]*RH_inertia_zed_kx, axis=0)
                f_zed_kx_ky_ri = f_zed_kx_ky_ri / RH_inertia_kx[None,:,None,None]

                if quantity == "RHnon_phi":
                    phi_zed_kx_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
                    phi_zed_kx_ky_ri[:,:,1:,:] = 0
                    f_zed_kx_ky_ri = phi_zed_kx_ky_ri - f_zed_kx_ky_ri

        else:
            f_zed_kx_ky_ri = np.zeros( (np.shape(f_zed_kx_ri)[0], np.shape(f_zed_kx_ri)[1], np.shape(f_zed_kx_ri)[2], len(ky), 2) )
            f_zed_kx_ky_ri[:,:,:,0,:] = f_zed_kx_ri

            if quantity == "RH_phi":
                # Divide by RH inertia
                RH_inertia_zed_kx = run.ncdata.variables['RH_inertia'][species_idx,0,:,:,0]
                dl_over_B_avg = run.dl_over_B_avg()
                RH_inertia_kx = np.sum(dl_over_B_avg[:,None]*RH_inertia_zed_kx, axis=0)
                f_zed_kx_ky_ri = f_zed_kx_ky_ri / RH_inertia_kx[None,None,:,None,None]


    elif quantity=="density":
        # density(t, species, tube, zed, kx, ky, ri)
        f_zed_kx_ky_ri = run.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
    elif quantity=="upar":
        # upar(t, species, tube, zed, kx, ky, ri)
        f_zed_kx_ky_ri = run.ncdata.variables['upar'][time_idx,species_idx,0,:,:,:,:]

    elif quantity=="unonPS":
        costheta = run.get_zed_weight("cos", zed) / run.get_zed_weight(None, zed)

        # upar(t, species, tube, zed, kx, ky, ri)
        upar_zed_kx_ky_ri = run.ncdata.variables['upar'][time_idx,species_idx,0,:,:,:,:]

        # phi_vs_t(t, tube, zed, theta0, ky, ri)
        phi_zed_kx_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
        dxphi_zed_kx_ky_ri = np.zeros_like(phi_zed_kx_ky_ri)
        dxphi_zed_kx_ky_ri[:,:,:,0] = -kx[None,:,None]  * phi_zed_kx_ky_ri[:,:,:,1]
        dxphi_zed_kx_ky_ri[:,:,:,1] =  kx[None,:,None]  * phi_zed_kx_ky_ri[:,:,:,0]

        qinp   = run.safety_factor
        eps    = 0.5/2.778

        #FACTOR OF TWO
        f_zed_kx_ky_ri =  (upar_zed_kx_ky_ri + qinp*dxphi_zed_kx_ky_ri*costheta[:,None,None,None] + 0.8*qinp*np.sqrt(eps)*dxphi_zed_kx_ky_ri)*costheta[:,None,None,None]
        #f_zed_kx_ky_ri =  (upar_zed_kx_ky_ri + 2*qinp*dxphi_zed_kx_ky_ri*costheta[:,None,None,None] + 1.6*qinp*np.sqrt(eps)*dxphi_zed_kx_ky_ri)*costheta[:,None,None,None]


    elif quantity=="unonRH":
        costheta = run.get_zed_weight("cos", zed) / run.get_zed_weight(None, zed)

        # upar(t, species, tube, zed, kx, ky, ri)
        upar_zed_kx_ky_ri = run.ncdata.variables['upar'][time_idx,species_idx,0,:,:,:,:]

        # phi_vs_t(t, tube, zed, theta0, ky, ri)
        phi_zed_kx_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
        dxphi_zed_kx_ky_ri = np.zeros_like(phi_zed_kx_ky_ri)
        dxphi_zed_kx_ky_ri[:,:,:,0] = -kx[None,:,None]  * phi_zed_kx_ky_ri[:,:,:,1]
        dxphi_zed_kx_ky_ri[:,:,:,1] =  kx[None,:,None]  * phi_zed_kx_ky_ri[:,:,:,0]

        qinp   = run.safety_factor
        eps    = 0.5/2.778

        #FACTOR OF TWO
        f_zed_kx_ky_ri = upar_zed_kx_ky_ri + qinp*dxphi_zed_kx_ky_ri*costheta[:,None,None,None] + 0.8*qinp*np.sqrt(eps)*dxphi_zed_kx_ky_ri
        #f_zed_kx_ky_ri = upar_zed_kx_ky_ri + 2*qinp*dxphi_zed_kx_ky_ri*costheta[:,None,None,None] + 1.6*qinp*np.sqrt(eps)*dxphi_zed_kx_ky_ri

    elif quantity=="upar-over-B":
        _, _, _, _, _, bmag = run.get_FLR()
        f_zed_kx_ky_ri = run.ncdata.variables['upar'][time_idx,species_idx,0,:,:,:,:]/bmag[:,None,None,None]
    elif quantity=="pressure_par": #(xpa^2)
        P_zed_kx_ky_ri = run.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
        try:
            Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
        except:
            Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
        f_zed_kx_ky_ri = P_zed_kx_ky_ri-0.5*Pprp_zed_kx_ky_ri
    elif quantity=="pressure_perp": #(xperp^2)
        # pressure_perp(t, species, tube, zed, kx, ky, ri)
        try:
            f_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
        except:
            f_zed_kx_ky_ri = run.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
    elif quantity=="pressure": #(xpa^2+xprp^2/2)
        try:
            f_zed_kx_ky_ri = run.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
        except:
            n_zed_kx_ky_ri = run.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:] #1
            T_zed_kx_ky_ri = run.ncdata.variables['temperature'][time_idx,species_idx,0,:,:,:,:] #(xpa^2+xprp^2-3/2)/(3/2)
            Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:] # xprp^2
            f_zed_kx_ky_ri = 1.5*(T_zed_kx_ky_ri+n_zed_kx_ky_ri) - 0.5*Pprp_zed_kx_ky_ri

    elif quantity=="pressure-phi":
        f_zed_kx_ky_ri_1 = run.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
        f_zed_kx_ky_ri_2 = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
        f_zed_kx_ky_ri = f_zed_kx_ky_ri_1 + f_zed_kx_ky_ri_2
    elif quantity=="pressure_perp-phi":
        try:
            f_zed_kx_ky_ri_1 = run.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
        except:
            f_zed_kx_ky_ri_1 = run.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
        f_zed_kx_ky_ri_2 = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
        f_zed_kx_ky_ri = f_zed_kx_ky_ri_1 + f_zed_kx_ky_ri_2
    elif quantity=="dtP_GAM":
        fP_zed_kx_ky_ri = run.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
        try:
            fchi_zed_kx_ky_ri = run.ncdata.variables['chi'][time_idx,species_idx,0,:,:,:,:]
        except:
            print("chi not found in NETCDF! Using pressure instead.")
            fchi_zed_kx_ky_ri = fP_zed_kx_ky_ri
        fphi_zed_kx_ky_ri = run.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
        tau = 1
        f_zed_kx_ky_ri = fchi_zed_kx_ky_ri+fP_zed_kx_ky_ri/tau + fphi_zed_kx_ky_ri*(7/4+1/tau)
    elif quantity=="chi":
        try:
            f_zed_kx_ky_ri = run.ncdata.variables['chi'][time_idx,species_idx,0,:,:,:,:]
        except:
            print("chi not found in NETCDF! Using pressure instead.")
            f_zed_kx_ky_ri = run.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
    elif quantity=="temperature": #(xpa^2+xprp^2-1.5)/1.5
        if run.code == "stella":
            # temperature(t, species, tube, zed, kx, ky, ri)
            f_zed_kx_ky_ri = run.ncdata.variables['temperature'][time_idx,species_idx,0,:,:,:,:]
        elif run.code == "GX":
            Tpar_zed_kx_ky_ri = np.transpose( run.ncdata_big['Diagnostics']['Tpar'][time_idx, species_idx] , axes=[2,1,0,3] )
            Tprp_zed_kx_ky_ri = np.transpose( run.ncdata_big['Diagnostics']['Tperp'][time_idx, species_idx] , axes=[2,1,0,3] )
            f_zed_kx_ky_ri = Tpar_zed_kx_ky_ri + Tprp_zed_kx_ky_ri
    elif quantity=="temperature_par": #(xpa^2-1/2)
        # temperature(t, species, tube, zed, kx, ky, ri)
        P_zed_kx_ky_ri    = run.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
        try:
            Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
        except:
            Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
        n_zed_kx_ky_ri    = run.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
        f_zed_kx_ky_ri = P_zed_kx_ky_ri - 0.5*Pprp_zed_kx_ky_ri - 0.5*n_zed_kx_ky_ri
    elif quantity=="temperature_perp": #(xprp^2-1)
        # temperature(t, species, tube, zed, kx, ky, ri)
        try:
            Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
        except:
            Pprp_zed_kx_ky_ri = run.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
        n_zed_kx_ky_ri    = run.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
        f_zed_kx_ky_ri = Pprp_zed_kx_ky_ri - n_zed_kx_ky_ri
    elif quantity=="qpar":
        # qpar(t, species, tube, zed, kx, ky, ri)
        f_zed_kx_ky_ri = run.ncdata.variables['qpar'][time_idx,species_idx,0,:,:,:,:]
    elif quantity=="qperp":
        # qperp(t, species, tube, zed, kx, ky, ri)
        f_zed_kx_ky_ri = run.ncdata.variables['qperp'][time_idx,species_idx,0,:,:,:,:]
    elif quantity=="qpar-over-B":
        _, _, _, _, _, bmag = run.get_FLR()
        # qpar(t, species, tube, zed, kx, ky, ri)
        f_zed_kx_ky_ri = run.ncdata.variables['qpar'][time_idx,species_idx,0,:,:,:,:]/bmag[:,None,None,None]
    elif quantity=="vflx_pol_phi_slab_kxz":
        # vflx_pol_phi_slab_kxz(t, species, tube, zed, kx, ri)
        f_zed_kx_ky_ri = np.zeros(shape=(len(zed), len(kx), len(ky), 2))
        f_zed_kx_ky_ri[:,:,0,:] = run.ncdata.variables['vflx_pol_phi_slab_kxz'][time_idx,species_idx,0]
    elif quantity=="vflx_pol_phi_shear_kxz":
        f_zed_kx_ky_ri = np.zeros(shape=(len(zed), len(kx), len(ky), 2))
        f_zed_kx_ky_ri[:,:,0,:] = run.ncdata.variables['vflx_pol_phi_shear_kxz'][time_idx,species_idx,0]
    elif quantity=="vflx_pol_Tperp_slab_kxz":
        f_zed_kx_ky_ri = np.zeros(shape=(len(zed), len(kx), len(ky), 2))
        f_zed_kx_ky_ri[:,:,0,:] = run.ncdata.variables['vflx_pol_Tperp_slab_kxz'][time_idx,species_idx,0]
    elif quantity=="vflx_pol_Tperp_shear_kxz":
        f_zed_kx_ky_ri = np.zeros(shape=(len(zed), len(kx), len(ky), 2))
        f_zed_kx_ky_ri[:,:,0,:] = run.ncdata.variables['vflx_pol_Tperp_shear_kxz'][time_idx,species_idx,0]

    else:
        ### Composite quantities that can be evaluated directly in real space
        if quantity=="deltaphi_2":
            phi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=remove_zonal, only_zonal=only_zonal, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            phi_mean = np.mean(phi, axis=0)
            for i_zed in range(len(zed)):
                delta_phi = phi[i_zed] - phi_mean
            f_zed_x_y = delta_phi**2

        elif quantity=="deltaphi":
            phi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=remove_zonal, only_zonal=only_zonal, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            #phi[:] = np.mean(phi, axis=0)
            #delta_phi = phi
            zed_weight = run.dl_over_B_avg()
            delta_phi = phi - np.sum(phi*zed_weight[:,None,None], axis=0)
            f_zed_x_y = delta_phi

        elif quantity=="dyphi-qpar-over-B":
            _, _, _, _, _, bmag = run.get_FLR()
            # qpar(t, species, tube, zed, kx, ky, ri)
            dyphi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            qpar_over_B , _, _, _, _    = run.get_quantity_zed_x_y(quantity="qpar-over-B", time_idx=time_idx, species_idx=species_idx, time_val=time_val, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dyphi*qpar_over_B

        elif quantity=="dyphi-upar":
            dyphi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            upar, _, _, _, _ = run.get_quantity_zed_x_y(quantity="upar", time_idx=time_idx, species_idx=species_idx, time_val=time_val, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dyphi*upar


        elif quantity=="dyphi-dxphi":
            dyphi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dxphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dyphi*dxphi

        elif quantity=="dyphi-dyphi":
            dyphi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dyphi**2

        elif quantity=="dyPprp-dxphi":
            dyPprp, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="pressure_perp", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dxphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dyPprp*dxphi

        elif quantity=="dyphiPprp-dxphi":
            dyphi, zed, x, y, time_eval  = run.get_quantity_zed_x_y(quantity="phi",           time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dyPprp, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="pressure_perp", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dxphi, _, _, _, _            = run.get_quantity_zed_x_y(quantity="phi",           time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = (dyphi+dyPprp)*dxphi

        elif quantity=="dyphiPprp-dyphi":
            dyphi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dyPprp, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="pressure_perp", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = (dyphi+dyPprp)*dyphi

        elif quantity=="dyPprp-dyphi":
            dyPprp, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="pressure_perp", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dyphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dyPprp*dyphi

        elif quantity=="dyphi-P":
            pressure, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=0, time_avg=time_avg, nx=nx, ny=ny)
            dyphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = pressure*dyphi

        elif quantity=="dyphi-chi":
            chi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="chi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=0, time_avg=time_avg, nx=nx, ny=ny)
            dyphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = chi*dyphi

        elif quantity=="dxTZ_dyphi2":
            dxTZ, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="temperature", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=False, only_zonal=True, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dyphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dxTZ*dyphi**2

        elif quantity=="dxTZtot_dyphi2":
            tprim = run.ncdata.variables['tprim'][0]
            dxTZ, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="temperature", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=False, only_zonal=True, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dyphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = (dxTZ-tprim)*dyphi**2

        elif quantity=="dxPZ_dyphi2":
            dxPZ, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=False, only_zonal=True, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dyphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dxPZ*dyphi**2

        elif quantity=="dxdeltaphiZ_dyphi_dyP":
            dxdeltaphiZ, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="deltaphi", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=False, only_zonal=True, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dyphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi",      time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dyP, _, _, _, _   = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dxdeltaphiZ*dyphi*dyP

        elif quantity=="NL_heat_flux_transp":
            dyphi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dxphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
            dyP, _, _, _, _ = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dxP, _, _, _, _ = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
            f_zed_x_y = -dyphi*(dxphi*dyP - dyphi*dxP)

        elif quantity=="vMy_heat_flux_transp":
            dyphi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dy2phi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=2, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
            P, _, _, _, _ = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
            dy2P, _, _, _, _ = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=2, time_avg=time_avg, nx=nx, ny=ny)
            dychi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="chi",      time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
            tau = 1
            f_zed_x_y = -(dy2phi+dy2P)*P/tau - dyphi*(dychi + 7/4*dyphi)

        elif quantity=="vMx_heat_flux_transp":
            dyphi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dxphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
            dydxphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
            P, _, _, _, _ = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
            dydxP, _, _, _, _ = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dxchi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="chi",      time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
            tau = 1
            f_zed_x_y = -(dydxphi+dydxP)*P/tau - dyphi*(dxchi + 7/4*dxphi)

        elif quantity=="kappa_transp":
            tprim = run.ncdata.variables['tprim'][0]
            #fprim = run.ncdata.variables['fprim'][0]
            dyphi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            # Recall dxT0 < 0 as temperature is higher in core
            f_zed_x_y = dyphi**2 * (-tprim)

        elif quantity=="dyphi-dyupar-over-B":
            dyphi,  zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dyupar, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="upar-over-B", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dyphi*dyupar

        elif quantity=="d2yphi-dypressure":
            d2yphi,  zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=2, time_avg=time_avg, nx=nx, ny=ny)
            dypres, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = d2yphi*dypres

        elif quantity=="d2yphi-dxpressure":
            d2yphi,  zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=2, time_avg=time_avg, nx=nx, ny=ny)
            dxpres, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
            f_zed_x_y = d2yphi*dxpres

        elif quantity=="dyphi2":
            dyphi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dyphi**2

        elif quantity=="dxphiZ_dyphi_dyP":
            dxphiZ, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=False, only_zonal=True, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dyphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            dyP,   _, _, _, _ = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dxphiZ*dyphi*dyP

        elif quantity=="dyphi-T":
            # NOTE: found while verifying time_avg-engaged output against
            # real data -- these two recursive calls were missing
            # time_avg=time_avg (every other ~40 sibling branches in this
            # elif chain correctly forward it), leaving an un-collapsed,
            # still-time-indexed intermediate array whenever time_avg was
            # set, which broke the kx/ky filtering shape assumptions
            # further down. Confirmed pre-existing (predates this
            # session), never previously exercised with time_avg set.
            temp, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="temperature", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=0, time_avg=time_avg, nx=nx, ny=ny)
            dyphi, _, _, _, _ = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
            f_zed_x_y = dyphi*temp

        elif quantity=="dyphi-dyP":
            dyphi, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            dyP  , zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            f_zed_x_y = dyphi*dyP

        elif quantity[:8] == "Reynolds":
            _, _, _, gds21, gds22, bmag = run.get_FLR()

            dxphi_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("phi",           time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)
            dyphi_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("phi",           time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)
            RS_factor_zed_x_y_nablax2 = dxphi_zed_x_y*(gds22/bmag**2)[:,None,None]
            RS_factor_zed_x_y_nablaxy = dyphi_zed_x_y*(gds21/bmag**2)[:,None,None]
            f_zed_x_y = np.zeros_like(RS_factor_zed_x_y_nablax2)

            if quantity in ["Reynolds", "Reynolds_nablax2", "Reynolds_Pprp_nablax2"]:
                dyPprp_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)
                f_zed_x_y += RS_factor_zed_x_y_nablax2 * dyPprp_zed_x_y
            if quantity in ["Reynolds", "Reynolds_nablaxy", "Reynolds_Pprp_nablaxy"]:
                dyPprp_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)
                f_zed_x_y += RS_factor_zed_x_y_nablaxy * dyPprp_zed_x_y
            if quantity in ["Reynolds", "Reynolds_nablax2", "Reynolds_phi_nablax2"]:
                f_zed_x_y += RS_factor_zed_x_y_nablax2 * dyphi_zed_x_y
            if quantity in ["Reynolds", "Reynolds_nablaxy", "Reynolds_phi_nablaxy"]:
                f_zed_x_y += RS_factor_zed_x_y_nablaxy * dyphi_zed_x_y

        elif quantity == "dEZ_Reynolds":
            # An earlier version of this quantity was
            # dx2phi_zed_x_y*reynolds_zed_x_y (2nd x-derivative of phi times
            # the un-differentiated Reynolds stress); replaced at some point
            # by dxphi_zed_x_y*dx_reynolds_zed_x_y below (1st x-derivative
            # of each instead) -- the specific reason for the change wasn't
            # recorded, flagged here in case it matters for future work on
            # this quantity.
            dx_reynolds_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("Reynolds", kx_order=1, time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            dxphi_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("phi", kx_order=1, time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)
            f_zed_x_y = dxphi_zed_x_y*dx_reynolds_zed_x_y

        elif quantity == "dEZ_vdriftx":
            phiZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("phi", kx_order=0, time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)
            dxPZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("pressure", kx_order=1, time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)
            vdriftx = run.get_zed_weight("vdriftx")

            f_zed_x_y = phiZ_zed_x_y*dxPZ_zed_x_y*vdriftx[:,None,None]

        elif quantity == "qpar_mom_transport":

            dyphi_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("phi",           time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)
            qparNZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("qpar",           time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)

            f_zed_x_y = qparNZ_zed_x_y * dyphi_zed_x_y # To be multiplied by dx(qparZ) to get energy time derivative contribution

        elif quantity == "dEZ_qpar_mom_transport":
            qpar_mom_transport_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("qpar_mom_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            dxqparZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("qpar",           time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)
            f_zed_x_y = -2*dxqparZ_zed_x_y * qpar_mom_transport_zed_x_y

        elif quantity == "par_mom_transport":

            dyphi_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("phi",           time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)
            uparNZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("upar",           time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)

            f_zed_x_y = uparNZ_zed_x_y * dyphi_zed_x_y # To be multiplied by dx(uparZ) to get energy time derivative contribution

        elif quantity == "dEZ_par_mom_transport":
            #par_mom_transport_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("par_mom_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            #dxuparZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("upar",           time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)
            #f_zed_x_y = 2*dxuparZ_zed_x_y * par_mom_transport_zed_x_y

            dx_par_mom_transport_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("par_mom_transport",time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            uparZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("upar",           time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)
            f_zed_x_y = -2*uparZ_zed_x_y * dx_par_mom_transport_zed_x_y

        elif quantity == "duZ_par_mom_transport":
            par_mom_transport_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("par_mom_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            dxvEZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("phi",           time_idx=time_idx, species_idx=species_idx, kx_order=2, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)
            f_zed_x_y = -2*dxvEZ_zed_x_y * par_mom_transport_zed_x_y

        elif quantity == "temperature_transport":

            dyphi_zed_x_y, zed, x, y, time_eval      = run.get_quantity_zed_x_y("phi",      time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)
            tempNZ_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("temperature", time_idx=time_idx, species_idx=species_idx, ky_order=0, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)

            f_zed_x_y = tempNZ_zed_x_y * dyphi_zed_x_y # To be multiplied by dx(tempZ) to get energy time derivative contribution

        elif quantity == "dEZ_mean_temperature_transport":
            temperature_transp_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("temperature_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            dxtempZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("temperature",       time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)

            dl_over_B_avg = run.dl_over_B_avg()
            mean_dxtempZ_x_y = np.sum(dl_over_B_avg[:,None,None]*dxtempZ_zed_x_y, axis=0)
            mean_dxtempZ_zed_x_y = np.zeros_like(dxtempZ_zed_x_y)
            for i_zed in range(len(zed)):
                mean_dxtempZ_zed_x_y[i_zed] = mean_dxtempZ_x_y
            f_zed_x_y = -4/3*mean_dxtempZ_zed_x_y*temperature_transp_zed_x_y

        elif quantity == "dEZ_delt_temperature_transport":
            temperature_transp_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("temperature_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            dxtempZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("temperature",       time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)

            dl_over_B_avg = run.dl_over_B_avg()
            mean_dxtempZ_x_y = np.sum(dl_over_B_avg[:,None,None]*dxtempZ_zed_x_y, axis=0)
            mean_dxtempZ_zed_x_y = np.zeros_like(dxtempZ_zed_x_y)
            for i_zed in range(len(zed)):
                mean_dxtempZ_zed_x_y[i_zed] = mean_dxtempZ_x_y
            delt_dxtempZ_zed_x_y = dxtempZ_zed_x_y - mean_dxtempZ_zed_x_y
            f_zed_x_y = -4/3*delt_dxtempZ_zed_x_y*temperature_transp_zed_x_y

        elif quantity == "pressure_transport":

            dyphi_zed_x_y, zed, x, y, time_eval      = run.get_quantity_zed_x_y("phi",      time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)
            presNZ_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("pressure", time_idx=time_idx, species_idx=species_idx, ky_order=0, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, remove_zonal=True)

            f_zed_x_y = presNZ_zed_x_y * dyphi_zed_x_y # To be multiplied by dx(presZ) to get energy time derivative contribution

        elif quantity == "dEZ_mean_pressure_transport":
            pressure_transp_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("pressure_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            dxpresZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("pressure",       time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)

            dl_over_B_avg = run.dl_over_B_avg()
            mean_dxpresZ_x_y = np.sum(dl_over_B_avg[:,None,None]*dxpresZ_zed_x_y, axis=0)
            mean_dxpresZ_zed_x_y = np.zeros_like(dxpresZ_zed_x_y)
            for i_zed in range(len(zed)):
                mean_dxpresZ_zed_x_y[i_zed] = mean_dxpresZ_x_y
            f_zed_x_y = -4/3*mean_dxpresZ_zed_x_y*pressure_transp_zed_x_y

        elif quantity == "dEZ_delt_pressure_transport":
            pressure_transp_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("pressure_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            dxpresZ_zed_x_y, zed, x, y, time_eval  = run.get_quantity_zed_x_y("pressure",       time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)

            dl_over_B_avg = run.dl_over_B_avg()
            mean_dxpresZ_x_y = np.sum(dl_over_B_avg[:,None,None]*dxpresZ_zed_x_y, axis=0)
            mean_dxpresZ_zed_x_y = np.zeros_like(dxpresZ_zed_x_y)
            for i_zed in range(len(zed)):
                mean_dxpresZ_zed_x_y[i_zed] = mean_dxpresZ_x_y
            delt_dxpresZ_zed_x_y = dxpresZ_zed_x_y - mean_dxpresZ_zed_x_y
            f_zed_x_y = -4/3*delt_dxpresZ_zed_x_y*pressure_transp_zed_x_y

        elif quantity == "P_RH_coll":
            #RH_flux_coll_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_collisional",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=0)
            #dx_phi_RH_zed_x_y, zed, x, y, time_eval    = run.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True, kx_order=1)
            #f_zed_x_y = RH_flux_coll_zed_x_y*dx_phi_RH_zed_x_y

            dx_phi_RH_zed_x_y, zed, x, y, time_eval    = run.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True, kx_order=1)
            try:
                RH_flux_coll_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_collisional",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=0)
                f_zed_x_y = RH_flux_coll_zed_x_y*dx_phi_RH_zed_x_y
            except:
                f_zed_x_y = np.zeros_like(dx_phi_RH_zed_x_y)

        elif quantity == "Pi_RH_even":
            try:
                Pi_RH_phi_even_passing_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_even_passing",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=-1)
                Pi_RH_phi_even_trapped_zed_x_y, _, _, _, _   = run.get_quantity_zed_x_y("RH_fluxes_phi_even_trapped",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=-1)
                f_zed_x_y = Pi_RH_phi_even_passing_zed_x_y + Pi_RH_phi_even_trapped_zed_x_y
            except:
                f_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_even",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=-1)


        elif quantity == "Pi_RH_odd":
            try:
                Pi_RH_phi_odd_passing_zed_x_y, zed, x, y, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_odd_passing",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=-1)
                Pi_RH_phi_odd_trapped_zed_x_y, _, _, _, _   = run.get_quantity_zed_x_y("RH_fluxes_phi_odd_trapped",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=-1)
                f_zed_x_y = Pi_RH_phi_odd_passing_zed_x_y + Pi_RH_phi_odd_trapped_zed_x_y
            except:
                f_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_odd",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=-1)

        elif quantity == "Pi_RH_NL":
            Pi_RH_even_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("Pi_RH_even", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff) 
            Pi_RH_odd_zed_x_y,    _, _, _,        _  = run.get_quantity_zed_x_y("Pi_RH_odd",  time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff) 
            f_zed_x_y = Pi_RH_even_zed_x_y + Pi_RH_odd_zed_x_y

        elif quantity == "P_RH_even":
            try:
                #dx_RH_flux_phi_even_passing_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_even_passing",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=1)
                #dx_RH_flux_phi_even_trapped_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_even_trapped",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=1)
                #dx_RH_flux_phi_even_zed_x_y = dx_RH_flux_phi_even_passing_zed_x_y + dx_RH_flux_phi_even_trapped_zed_x_y
                RH_flux_phi_even_passing_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_even_passing",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=0)
                RH_flux_phi_even_trapped_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_even_trapped",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=0)
                RH_flux_phi_even_zed_x_y = RH_flux_phi_even_passing_zed_x_y + RH_flux_phi_even_trapped_zed_x_y
            except:
                #dx_RH_flux_phi_even_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_even",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=1)
                RH_flux_phi_even_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_even",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=0)


            #phi_RH_zed_x_y, zed, x, y, time_eval    = run.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)
            dx_phi_RH_zed_x_y, zed, x, y, time_eval    = run.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True, kx_order=1)

            #f_zed_x_y = dx_RH_flux_phi_even_zed_x_y*phi_RH_zed_x_y
            f_zed_x_y = RH_flux_phi_even_zed_x_y*dx_phi_RH_zed_x_y

        elif quantity == "P_RH_odd":

            try:
                #dx_RH_flux_phi_odd_passing_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_odd_passing",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=1)
                #dx_RH_flux_phi_odd_trapped_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_odd_trapped",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=1)
                #dx_RH_flux_phi_odd_zed_x_y = dx_RH_flux_phi_odd_passing_zed_x_y + dx_RH_flux_phi_odd_trapped_zed_x_y
                RH_flux_phi_odd_passing_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_odd_passing",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=0)
                RH_flux_phi_odd_trapped_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_odd_trapped",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=0)
                RH_flux_phi_odd_zed_x_y = RH_flux_phi_odd_passing_zed_x_y + RH_flux_phi_odd_trapped_zed_x_y
            except:
                #dx_RH_flux_phi_odd_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_odd",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=1)
                RH_flux_phi_odd_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_odd",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=0)

            #phi_RH_zed_x_y, zed, x, y, time_eval    = run.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True)
            dx_phi_RH_zed_x_y, zed, x, y, time_eval    = run.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True, kx_order=1)
            #f_zed_x_y = -dx_RH_flux_phi_odd_zed_x_y*phi_RH_zed_x_y
            f_zed_x_y = RH_flux_phi_odd_zed_x_y*dx_phi_RH_zed_x_y

        elif quantity in ("P_RH_apar_even", "P_RH_apar_odd", "P_RH_bpar_even", "P_RH_bpar_odd"):
            # Electromagnetic (apar/bpar) counterparts of P_RH_even/P_RH_odd
            # above (which are phi-only) -- same passing+trapped-with-
            # fallback-to-combined naming, plus an outer fallback to zero
            # for runs that don't have this field at all (e.g.
            # include_apar/include_bpar=.false.), matching
            # physics.rosenbluth_hinton.get_RH_fluxes's same fallback.
            _, field, parity = quantity.split("_")[1:]
            dx_phi_RH_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("RH_phi", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True, kx_order=1)
            try:
                try:
                    passing_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_%s_%s_passing" % (field, parity), time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=0)
                    trapped_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_%s_%s_trapped" % (field, parity), time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=0)
                    RH_flux_zed_x_y = passing_zed_x_y + trapped_zed_x_y
                except:
                    RH_flux_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_%s_%s" % (field, parity), time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, kx_order=0)
                f_zed_x_y = RH_flux_zed_x_y*dx_phi_RH_zed_x_y
            except:
                f_zed_x_y = np.zeros_like(dx_phi_RH_zed_x_y)

        elif quantity in ("P_RH_full_even", "P_RH_full_odd"):
            # phi+apar+bpar for one parity -- the full physical P_RH^even/
            # P_RH^odd (P_RH_even/P_RH_odd above are phi-only).
            parity = quantity.split("_")[-1]
            f_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("P_RH_%s" % parity, time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            for field in ("apar", "bpar"):
                f_field_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("P_RH_%s_%s" % (field, parity), time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
                f_zed_x_y = f_zed_x_y + f_field_zed_x_y

        elif quantity == "P_RH_tot":
            P_RH_NL_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("P_RH_NL", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff) 
            P_RH_coll_zed_x_y,    _, _, _,        _  = run.get_quantity_zed_x_y("P_RH_coll",  time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff) 
            f_zed_x_y = P_RH_NL_zed_x_y + P_RH_coll_zed_x_y

        elif quantity == "P_RH_NL":
            P_RH_even_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y("P_RH_even", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff) 
            P_RH_odd_zed_x_y,    _, _, _,        _  = run.get_quantity_zed_x_y("P_RH_odd",  time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff) 
            f_zed_x_y = P_RH_even_zed_x_y + P_RH_odd_zed_x_y
            #RH_flux_phi_odd_zed_x_y, _, _, _, _ = run.get_quantity_zed_x_y("RH_fluxes_phi_odd", time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)
            #dx_phi_RH_zed_x_y, zed, x, y, time_eval    = run.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, only_zonal=True, kx_order=1)
            #f_zed_x_y = RH_flux_phi_odd_zed_x_y*dx_phi_RH_zed_x_y


        else:
            print("Did not enter valid quantity to plot (" + str(quantity) + "). Returning")
            return

        # Take x-derivative with finite difference if needed
        dx = x[1]-x[0]
        for i in range(kx_order):
            # Use periodicity
            f_zed_x_y_copy = np.copy(f_zed_x_y)
            for i_x in range(len(x)-1):
                if i_x == 0:
                    f_zed_x_y[:,0] = 0.5*(f_zed_x_y_copy[:,1]-f_zed_x_y_copy[:,-1])/dx
                else:
                    f_zed_x_y[:,i_x] = 0.5*(f_zed_x_y_copy[:,i_x+1]-f_zed_x_y_copy[:,i_x-1])/dx
            f_zed_x_y[:,-1] = 0.5*(f_zed_x_y_copy[:,0]-f_zed_x_y_copy[:,-2])/dx

        if only_zonal or remove_zonal:
            Ny = len(f_zed_x_y[0,0])
            fzonal_zed_x = np.sum(f_zed_x_y, axis=2)/Ny
            for i_y in range(Ny):
                if only_zonal:
                    f_zed_x_y[:,:,i_y] = fzonal_zed_x
                else:
                    f_zed_x_y[:,:,i_y] = f_zed_x_y[:,:,i_y] - fzonal_zed_x

        return f_zed_x_y, zed, x, y, time_eval

    # If absolute value squared of real part, we need to first transform to x-y, then abs^2, then time average (if desired). SLOWER!
    if abs_squared:
        # Insert a dummy leading time axis only if f_zed_kx_ky_ri doesn't
        # already have one -- i.e. based on time_idx's actual shape, not
        # time_avg (see the dt_avg_vals comment above for why those
        # differ for recursing composite quantities).
        if np.ndim(time_idx) == 0:
            f_t_zed_kx_ky_ri = f_zed_kx_ky_ri[None,:,:,:,:]
        else:
            f_t_zed_kx_ky_ri = f_zed_kx_ky_ri

        # Complex variable
        f_t_zed_kx_ky = f_t_zed_kx_ky_ri[:,:,:,:,0] + 1j*f_t_zed_kx_ky_ri[:,:,:,:,1]

        # Filter out kx's
        f_t_zed_kx_ky[:,:,np.abs(kx)>kx_lowpass_cutoff,:] = 0
        f_t_zed_kx_ky[:,:,np.abs(kx)<kx_highpass_cutoff,:] = 0

        # Filter out ky's
        f_t_zed_kx_ky[:,:,:,ky>ky_lowpass_cutoff] = 0
        f_t_zed_kx_ky[:,:,:,ky<ky_highpass_cutoff] = 0

        # x-derivatives
        f_t_zed_kx_ky[:,:,1:,:] = f_t_zed_kx_ky[:,:,1:,:] * (1j*kx[None,None,1:,None])**kx_order

        # y-derivatives
        f_t_zed_kx_ky = f_t_zed_kx_ky * (1j*ky[None,None,None,:])**ky_order

        # Filter zonal if requested
        if remove_zonal:
            f_t_zed_kx_ky[:,:,:,0]= 0
        if only_zonal:
            f_t_zed_kx_ky[:,:,:,1:]= 0

        # Fourier transform to real space
        f_t_zed_x_y = []
        for i_t in range(np.shape(f_t_zed_kx_ky)[0]):
            f_zed_x_y = []
            for i_zed in range(len(zed)):
                tmp, x, y = get_fft_real_space(f_t_zed_kx_ky[i_t, i_zed], kx, ky, nx=nx, ny=ny)
                f_zed_x_y.append(tmp)
            f_t_zed_x_y.append(f_zed_x_y)

        f_t_zed_x_y = np.array(f_t_zed_x_y)

        # Abs-squared
        f_t_zed_x_y = f_t_zed_x_y**2

        # Time average -- collapse the leading time axis whenever time_idx
        # was array-valued (not gated on time_avg, see the dt_avg_vals
        # comment above).
        if np.ndim(time_idx) > 0:
            f_zed_x_y = dt_weighted_mean(f_t_zed_x_y, weights=dt_avg_vals, axis=0)
        else:
            f_zed_x_y = f_t_zed_x_y[0,:,:,:]

    else:


        # Time average -- same time_idx-shape-based condition as above.
        if np.ndim(time_idx) > 0:
            f_zed_kx_ky_ri = dt_weighted_mean(f_zed_kx_ky_ri, weights=dt_avg_vals, axis=0)

        # Filter out kx's
        f_zed_kx_ky_ri[:,np.abs(kx)>kx_lowpass_cutoff,:,:] = 0
        f_zed_kx_ky_ri[:,np.abs(kx)<kx_highpass_cutoff,:,:] = 0

        # Filter out ky's
        f_zed_kx_ky_ri[:,:,ky>ky_lowpass_cutoff,:] = 0
        f_zed_kx_ky_ri[:,:,ky<ky_highpass_cutoff,:] = 0

        f_zed_kx_ky = f_zed_kx_ky_ri[:,:,:,0] + 1j*f_zed_kx_ky_ri[:,:,:,1]

        # x-derivatives
        f_zed_kx_ky = f_zed_kx_ky * (1j*kx[None,:,None])**kx_order

        # y-derivatives
        f_zed_kx_ky = f_zed_kx_ky * (1j*ky[None,None,:])**ky_order

        # Filter zonal if requested
        if remove_zonal:
            f_zed_kx_ky[:,:,0]= 0
        if only_zonal:
            f_zed_kx_ky[:,:,1:]= 0
    #f_zed_kx_ky[:,0,0] = 0

        if run.code == "GX":
            # Ensure kx is in FFT form
            idx_kx_0 = nearest_index(kx)
            kx_copy = np.copy(kx)
            kx[:idx_kx_0+1] = kx_copy[idx_kx_0:]
            kx[idx_kx_0+1:] = kx_copy[:idx_kx_0]
            f_zed_kx_ky_copy = np.copy(f_zed_kx_ky)
            f_zed_kx_ky[:,:idx_kx_0+1,:] = f_zed_kx_ky_copy[:,idx_kx_0:,:]
            f_zed_kx_ky[:,idx_kx_0+1:,:] = f_zed_kx_ky_copy[:,:idx_kx_0,:]

        # Fourier transform to real space
        f_zed_x_y = []
        for i_zed in range(len(zed)):
            tmp, x, y = get_fft_real_space(f_zed_kx_ky[i_zed], kx, ky, nx=nx, ny=ny)
            f_zed_x_y.append(tmp)

        f_zed_x_y = np.array(f_zed_x_y)

    return f_zed_x_y, zed, x, y, time_eval


def get_quantity_x_y(run, quantity, zed_val = None, zed_idx=None, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, mult_zed=None, kx_lowpass_cutoff=1e10, ky_lowpass_cutoff=1e10, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1, par_der_order=0, abs_squared=False):

    x_der_taken = False
    f_zed_x_y, zed, x, y, time_eval = run.get_quantity_zed_x_y(quantity=quantity, time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, abs_squared=abs_squared)
    if quantity not in ["Reynolds", "Reynolds_phi_nablax2", "Reynolds_phi_nablaxy", "Reynolds_Pprp_nablax2", "Reynolds_Pprp_nablaxy"]:
        x_der_taken = True

    # Take derivatives by finite differences if needed
    if not x_der_taken:
        dx = x[1]-x[0]
        for i in range(kx_order):
            # Use periodicity
            f_zed_x_y_copy = np.copy(f_zed_x_y)
            for i_x in range(len(x)-1):
                if i_x == 0:
                    f_zed_x_y[:,0] = 0.5*(f_zed_x_y_copy[:,1]-f_zed_x_y_copy[:,-1])/dx
                else:
                    f_zed_x_y[:,i_x] = 0.5*(f_zed_x_y_copy[:,i_x+1]-f_zed_x_y_copy[:,i_x-1])/dx
            f_zed_x_y[:,-1] = 0.5*(f_zed_x_y_copy[:,0]-f_zed_x_y_copy[:,-2])/dx

    # Take zed derivatives if needed
    for i in range(par_der_order):
        gradpar  = run.ncdata.variables['gradpar'][:]
        # Use periodicity
        _, _, _, _, gds22, bmag = run.get_FLR()
        f_zed_x_y = np.gradient(f_zed_x_y*gradpar[:,None,None], zed, axis=0)

    # if zed_val is not None, find zed_idx matching zed_val most closely
    if zed_val is not None:
        zed_idx = np.argmin( np.abs(zed - zed_val) )

    if zed_idx is None:
        zed_weight = run.get_zed_weight(mult_zed=mult_zed, zed=zed)
        f_x_y = np.sum(f_zed_x_y*zed_weight[:,None,None], axis=0)
 
    else:
        f_x_y = f_zed_x_y[zed_idx]

    return f_x_y, x, y, time_eval


def get_quantity_x(run, quantity="phi", species_idx=0, zed_idx=None, time_idx=-1, normalise=False, time_avg=None, nx=None, mult_zed=None, kx_order=0, kx_lowpass_cutoff=1e5, mult=1, plot_factor=1):
    """No-plot equivalent of stella_diagnostics.plotting.realspace_plots.
    plot_quantity_x's computation (same get_quantity_x_y call, 1D
    extraction, and normalisation), extracted so it can be called from
    inside a @cached function (stella_diagnostics.scan.quantities_x_scan),
    which must not call plotting code. plot_quantity_x's own body is left
    untouched. Returns (norm_val, x, f_Z, time_eval) -- matching the
    (norm_val, x, f_Z) plot_quantity_x returns alongside its fig/ax."""
    f_Z, x, _, time_eval = run.get_quantity_x_y(quantity=quantity, species_idx=species_idx, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=False, only_zonal=True, kx_order=kx_order, time_avg=time_avg, nx=nx, mult_zed=mult_zed, kx_lowpass_cutoff=kx_lowpass_cutoff)

    f_Z = mult * f_Z[:, 0]

    if normalise:
        norm_val = 1 / np.abs(f_Z).max()
    else:
        norm_val = 1

    return norm_val, x, f_Z, time_eval
