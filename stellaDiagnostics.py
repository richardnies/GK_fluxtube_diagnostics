import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.special as specialfunc
import warnings
from scipy.interpolate import interp1d as interp
from scipy.interpolate import RegularGridInterpolator as interp2D
from scipy import integrate
from tftb.processing import WignerVilleDistribution
import seaborn as sns
from scipy.interpolate import interpn
from glob import glob
from os.path import exists
from scipy.signal import argrelextrema

class stellaDiagnostics:

    def __init__(self, filename_base, code="stella", debug=False):

        self.code  = code
        self.debug = debug

        warnings.filterwarnings('ignore', category=UserWarning)
        self.filename_base = filename_base

        self.input_file   = filename_base+".in"
        #print("Diagnostics for code " + self.code + ", reading " + filename_base)
#        if self.code == "GX":
#            self.netcdf_file  = filename_base+".nc"
#        elif self.code == "GS2":
#            self.netcdf_file  = filename_base+".out_copy.nc"
#        else:
#            self.netcdf_file  = filename_base+".out.nc"

        self.netcdf_file  = filename_base+".out.nc"
        if not exists(self.netcdf_file):
            #print("Could not find " + self.netcdf_file)
            self.netcdf_file  = filename_base+".nc"

        self.omega_file   = filename_base+".omega"
        if code == "stella":
            self.geo_file_alt = filename_base+".geometry"
            self.geo_file     = filename_base+".vmec.geo"
        elif code == "GX":
            char_dir = filename_base.rfind("/")
            if char_dir == -1:
                dir_data = "."
            else:
                dir_data = filename_base[:char_dir]
            self.geo_file     = dir_data + "/eik.out"
        self.fluxes_file  = filename_base+".fluxes"
        self.ncdata       = nc4.Dataset(self.netcdf_file,'r')
        if code=="GX":
            self.GX_old_version = False
            try:
                self.netcdf_big = filename_base+".big.nc"
                self.ncdata_big = nc4.Dataset(self.netcdf_big,'r')
            except:
                print("WARNING! Old GX version.")
                self.GX_old_version = True
      
        try:
            geom_factors = np.loadtxt(self.geo_file, max_rows=1)
            self.safety_factor = geom_factors[1] #qinp
            self.aspect_ratio  = geom_factors[0] * geom_factors[3] # aref*rhotor
            self.aspect_ratio_a= geom_factors[0] # aref
            geom_factors = np.loadtxt(self.geo_file, skiprows=2)
            self.alpha0        = geom_factors[0][0]
        except:
            try:
                # For Miller
                inputdata = open(self.geo_file_alt, 'r').read().strip()
                inputdata1 = inputdata.split("\n")[1][5:].split("   ")
                self.safety_factor = float(inputdata1[1]) #qinp
                self.aspect_ratio  = 2.8#float(inputdata1[0]) / float(inputdata1[6]) # rhoc/dxdXcoord
                #self.aspect_ratio  = 1/5.55#float(inputdata1[0]) / float(inputdata1[6]) # rhoc/dxdXcoord
            
            except Exception as e:
                #print("Warning:", type(e).__name__) 
                print("Warning! Geometry file for " + self.filename_base + " do not exist?")
                #print("Suppressing this warning for rest of python call")
                #warnings.filterwarnings('ignore', category=FileNotFoundError)


    def read_basic_params(self):
        dict_params = dict()

        dict_params['theta0'] = self.ncdata['theta0'][0]
        dict_params['kx'] = self.ncdata['kx'][:]
        dict_params['ky'] = self.ncdata['ky'][:]

        return dict_params

    def get_kx_ky_zed(self):
        if self.code == "stella":
            kx     = self.ncdata.variables['kx'][:]
            ky     = self.ncdata.variables['ky'][:] 
            zed    = self.ncdata.variables['zed'][:]
        elif self.code == "GX":
            if self.GX_old_version:
                kx     = np.array(self.ncdata.variables['kx'][:]) * np.sqrt(2) # sqrt(2) factor as we care about kx*rho_i !
                ky     = np.array(self.ncdata.variables['ky'][:]) * np.sqrt(2) 
                zed    = self.ncdata.variables['theta'][:]
            else:
                kx     = np.array(self.ncdata['Grids']['kx'][:])   * np.sqrt(2)
                ky     = np.array(self.ncdata['Grids']['ky'][:])   * np.sqrt(2)
                zed    = self.ncdata['Grids']['theta'][:]

        elif self.code == "GS2":
            kx     = self.ncdata.variables['kx'][:]
            ky     = self.ncdata.variables['ky'][:] 
            zed    = self.ncdata.variables['zed'][:]
        return kx, ky, zed


    def get_time_array(self, GX_big=False):
        if self.code == "GX":
            if self.GX_old_version:
                time   = self.ncdata.variables['time'][:]
            elif GX_big:
                time = self.ncdata_big['Grids']['time'][:]
            else:
                time = self.ncdata['Grids']['time'][:]
        else:
            time   = self.ncdata.variables['t'][:]
        return time

    def get_time_idx(self, time_val):
        time = self.get_time_array()
        return np.argmin(np.abs(time-time_val))
        

    ###########################################################################
    #######  Functions loading data                                     #######
    ###########################################################################
    def get_zed_weight(self, mult_zed, zed=None):

        zed_weight  = self.dl_over_B_avg()
        if self.code=="stella":
            shat   = self.ncdata.variables['shat'].getValue()
        elif self.code=="GX":
            shat   = self.ncdata['Geometry']['shat']

        if mult_zed == 1:
            zed_weight[:] = 1
        elif mult_zed is None:
            zed_weight = zed_weight*1
        elif mult_zed == "nablax-nablax":
            _, _, _, _, gds22, bmag = self.get_FLR()
            zed_weight = zed_weight*gds22/bmag**2
        elif mult_zed == "nablax2-vdriftx":
            _, _, _, _, gds22, bmag = self.get_FLR()
            zed_weight = zed_weight * self.ncdata.variables['gbdrift0'][:,0]/(2*shat)*gds22/bmag**2
        elif mult_zed == "nablax-nablay":
            _, _, _, gds21, _, bmag = self.get_FLR()
            zed_weight = zed_weight*gds21/bmag**2
        elif mult_zed == "nablaxy-vdriftx":
            _, _, _, gds21, _, bmag = self.get_FLR()
            zed_weight = zed_weight * self.ncdata.variables['gbdrift0'][:,0]/(2*shat)*gds21/bmag**2
        elif mult_zed == "vdrifty":
            zed_weight = zed_weight * self.ncdata.variables['gbdrift'][:,0]
        elif mult_zed == "vdriftx-vdrifty":
            zed_weight = zed_weight * self.ncdata.variables['gbdrift'][:,0]* self.ncdata.variables['gbdrift0'][:,0]/(2*shat)
        elif mult_zed == "vdriftx":
            zed_weight = zed_weight * self.ncdata.variables['gbdrift0'][:,0]/(2*shat)
            if self.debug:
                print("Note: sum(vdriftx)/sum(|vdriftx|) = %e" % (np.sum(zed_weight)/np.sum(np.abs(zed_weight))))
        elif mult_zed == "vdriftx-B":
            _, _, _, _, gds22, bmag = self.get_FLR()
            zed_weight = zed_weight*self.ncdata.variables['gbdrift0'][:,0]/(2*shat)*bmag
        elif mult_zed == "B":
            _, _, _, _, _, bmag = self.get_FLR()
            zed_weight = zed_weight*bmag
        elif mult_zed == "vdriftx-abs":
            zed_weight = zed_weight*np.abs(self.ncdata.variables['gbdrift0'][:,0]/(2*shat))
        elif mult_zed == "vdriftx2":
            zed_weight = zed_weight * (self.ncdata.variables['gbdrift0'][:,0]/(2*shat))**2
        elif mult_zed == "pos":
            zed_weight[zed<0] = 0
        elif mult_zed == "neg":
            zed_weight[zed>0] = 0
        elif mult_zed == "sin":
            zed_weight = zed_weight * np.sin(zed)
        elif mult_zed == "cos":
            zed_weight = zed_weight * np.cos(zed)
        elif mult_zed == "hfs":
            zed_weight[np.abs(zed)<np.pi/2] = 0
        elif mult_zed == "lfs":
            zed_weight[np.abs(zed)>np.pi/2] = 0
        else:
            print("WARNING! The indicated mult_zed is not in the list of options.")

        return zed_weight


    # Outer scale k_y rho_i, (13) in Barnes et al. 2011
    def read_avg_ky_rhoi(self, time_idx_jump=1, avg_qflx=False, normal_mean=False, take_max=False):
        time   = self.ncdata.variables['t'][::time_idx_jump]
        Ntime = len(time)
        ky     = self.ncdata.variables['ky'][1:]

        if avg_qflx:
            # Average over qflx
    	    # qflx_kxky(t, species, tube, zed, kx, ky)
            qflx_t_zed_kx_ky = self.ncdata.variables['qflx_kxky'][::time_idx_jump,0, 0, :, :, 1:]
            dl_over_B_avg = self.dl_over_B_avg()
            phi2_vs_kxky = np.sum( qflx_t_zed_kx_ky*dl_over_B_avg[None,:,None,None], axis=1)
            
        else:
            # Average over phi^2
            # phi2_vs_kxky(t, kx, ky)
            phi2_vs_kxky = self.ncdata.variables['phi2_vs_kxky'][::time_idx_jump,:,1:]

        ky_rhoi_O = np.zeros(Ntime) 

        for i_time in range(Ntime):
            if take_max:
                phi2_ky = np.sum(phi2_vs_kxky[i_time], axis=0)
                ky_rhoi_O[i_time] = ky[np.argmax(phi2_ky)]
            else:
                denominator = np.sum(phi2_vs_kxky[i_time])
                if normal_mean:
                    numerator   = np.sum(phi2_vs_kxky[i_time]*ky[None,:])
                    ky_rhoi_O[i_time] = numerator/denominator
                else:
                    numerator   = np.sum(phi2_vs_kxky[i_time]/ky[None,:])
                    ky_rhoi_O[i_time] = 1. / (numerator/denominator)

        return ky_rhoi_O, np.asarray(time)

    # Outer scale k_x rho_i
    def read_avg_kx_rhoi(self, time_idx_jump=1, avg_qflx=False, normal_mean=False, take_max=False, only_zonal=False, remove_zonal=True):
        time   = self.ncdata.variables['t'][::time_idx_jump]
        Ntime = len(time)
        kx     = self.ncdata.variables['kx'][:]

        if avg_qflx and not only_zonal:
            # Average over qflx
    	    # qflx_kxky(t, species, tube, zed, kx, ky)
            qflx_t_zed_kx_ky = self.ncdata.variables['qflx_kxky'][::time_idx_jump,0, 0, :, :, 1:]
            dl_over_B_avg = self.dl_over_B_avg()
            phi2_vs_kxky = np.sum( qflx_t_zed_kx_ky*dl_over_B_avg[None,:,None,None], axis=1)
            
        else:
            # Average over phi^2
            # phi2_vs_kxky(t, kx, ky)
            phi2_vs_kxky = self.ncdata.variables['phi2_vs_kxky'][::time_idx_jump]
            if only_zonal:
                phi2_vs_kxky[:,:,1:] = 0
            elif remove_zonal:
                phi2_vs_kxky[:,:,0] = 0

        kx_rhoi_O = np.zeros(Ntime) 

        for i_time in range(Ntime):
            if take_max:
                phi2_kx = np.sum(phi2_vs_kxky[i_time], axis=1)
                kx_rhoi_O[i_time] = np.abs(kx[np.argmax(phi2_kx)])
            else:
                denominator = np.sum(phi2_vs_kxky[i_time])
                if normal_mean:
                    numerator   = np.sum(phi2_vs_kxky[i_time]*np.abs(kx[:,None]))
                    kx_rhoi_O[i_time] = numerator/denominator
                else:
                    numerator   = np.sum(phi2_vs_kxky[i_time]/np.abs(kx[:,None]))
                    kx_rhoi_O[i_time] = 1. / (numerator/denominator)

        return kx_rhoi_O, np.asarray(time)

#    # Outer scale k_x rho_i
#    def read_avg_kx_rhoi(self):
#        time   = self.ncdata.variables['t'] 
#        Ntime = len(time)
#        kx     = self.ncdata.variables['kx'] 
#        # phi2_vs_kxky(t, kx, ky)
#        phi2_vs_kxky = self.ncdata.variables['phi2_vs_kxky']
#
#        kx_rhoi_O = np.zeros(Ntime) 
#
#        for i_time in range(Ntime):
#            numerator = denominator = 0
#            for i_kx in range(1,len(kx)):
#                numerator   = numerator   + np.sum(phi2_vs_kxky[i_time], axis=1)[i_kx] / np.abs(kx[i_kx])
#                denominator = denominator + np.sum(phi2_vs_kxky[i_time], axis=1)[i_kx]
#
#            kx_rhoi_O[i_time] = 1 / (numerator/denominator)
#
#        return kx_rhoi_O, np.asarray(time)

    # Return flux-tube average factor
    def dl_over_B_avg(self):

        if self.code == "stella" or self.code == "GS2":
            gradpar = self.ncdata.variables['gradpar'][:]
            bmag    = self.ncdata.variables['bmag'][:]
        elif self.code == "GX":
            gradpar  = self.ncdata['Geometry']['gradpar']
            bmag     = self.ncdata['Geometry']['bmag'][:]
    
        dl_over_b = np.squeeze(1/(gradpar*bmag.T))
        #dl_over_b[ 0] = 0.5*dl_over_b[ 0] # First and last points of tube are connected
        #dl_over_b[-1] = 0.5*dl_over_b[-1] # First and last points of tube are connected
        dl_over_b_avg = np.squeeze(dl_over_b/np.sum(dl_over_b))

        return dl_over_b_avg

    # Factor used in normalisation of fluxes
    # see stella_diagnostics.f90
    def flux_norm(self):
        grho      = self.ncdata.variables['grho'][:,0]
        dl_over_b = self.dl_over_B_avg()

        flux_norm = np.sum(grho*dl_over_b)
        print("\n"+self.filename_base+": flux_norm = %e" % (flux_norm))

        return flux_norm
        
    # Evaluate net radial drift (assumes tube is centered on magnetic well, B_bounce goes between 0 and 1 (max along field line))
    def evaluate_net_radial_drift(self, B_bounce=0.9):
        cvdrift0 = self.ncdata.variables['cvdrift0'][:,0] # drift * grad(x) * shat
        bmag     = self.ncdata.variables['bmag'][:,0]
        dl_over_B_avg = self.dl_over_B_avg()
        zed     = self.ncdata.variables['zed'][:]
    
        bmag_norm = (bmag - bmag.min())/(bmag.max() - bmag.min())

        func_integrand = dl_over_B_avg*cvdrift0

        bmag_norm_signed = bmag_norm*np.sign(zed)
        idx_bnc_pls  = np.argmin( np.abs(bmag_norm_signed - B_bounce) )
        idx_bnc_min  = np.argmin( np.abs(bmag_norm_signed + B_bounce) )

        dB_dzed_bnc_pls = (bmag_norm[idx_bnc_pls]-bmag_norm[idx_bnc_pls-1])/(zed[idx_bnc_pls]-zed[idx_bnc_pls-1])
        dB_dzed_bnc_min = (bmag_norm[idx_bnc_min]-bmag_norm[idx_bnc_min+1])/(zed[idx_bnc_min]-zed[idx_bnc_min+1])


        endpoint_contr_pls = 2*func_integrand[idx_bnc_pls]*np.sqrt(np.abs( (zed[idx_bnc_pls]-zed[idx_bnc_min]) *B_bounce/dB_dzed_bnc_pls) )
        endpoint_contr_min = 2*func_integrand[idx_bnc_min]*np.sqrt(np.abs( (zed[idx_bnc_pls]-zed[idx_bnc_min]) *B_bounce/dB_dzed_bnc_min) )

        net_radial_drift      =        endpoint_contr_pls         + endpoint_contr_min
        net_radial_drift_norm = np.abs(endpoint_contr_pls) + np.abs(endpoint_contr_min)
        integrand_tmp = np.zeros(len(zed))

#        net_radial_drift      = 0
#        net_radial_drift_norm = 0

#        for i_zed in range(len(bmag)):
#            if bmag_norm[i_zed] < B_bounce:
#                net_radial_drift      +=        func_integrand[i_zed]  / np.sqrt(1-bmag_norm[i_zed]/B_bounce)
#                net_radial_drift_norm += np.abs(func_integrand[i_zed]) / np.sqrt(1-bmag_norm[i_zed]/B_bounce)
#                #net_radial_drift      +=        func_integrand[i_zed]  / np.sqrt(1-bmag_norm[i_zed])
#                #net_radial_drift_norm += np.abs(func_integrand[i_zed]) / np.sqrt(1-bmag_norm[i_zed])

        dzed = zed[1]-zed[0]
        for i_zed in range(idx_bnc_min+1, idx_bnc_pls):
            integrand =  func_integrand[i_zed]      /np.sqrt(1-bmag_norm[i_zed]/B_bounce) \
                      - func_integrand[idx_bnc_pls]/np.sqrt(np.abs(dB_dzed_bnc_pls/B_bounce*(zed[i_zed]-zed[idx_bnc_pls]))) \
                      - func_integrand[idx_bnc_min]/np.sqrt(np.abs(dB_dzed_bnc_min/B_bounce*(zed[i_zed]-zed[idx_bnc_min])))
            net_radial_drift      +=        integrand  *dzed
            net_radial_drift_norm += np.abs(integrand) *dzed
            integrand_tmp[i_zed] = integrand

#        plt.loglog(bmag_norm[idx_bnc_min+1:idx_bnc_pls], np.abs(integrand_tmp[idx_bnc_min+1:idx_bnc_pls]))
#        plt.plot(bmag_norm[idx_bnc_min+1:idx_bnc_pls], integrand_tmp[idx_bnc_min+1:idx_bnc_pls])
#        plt.savefig("tmp.pdf")

#        from scipy.interpolate import interp1d as interp
#        integrand_interp = interp( bmag_norm_signed[bmag_norm<=B_bounce], func_integrand[bmag_norm<=B_bounce]/np.sqrt(1-bmag_norm[bmag_norm<=B_bounce]/B_bounce), fill_value="extrapolate")
#
#
#        B_val_max = max( bmag_norm[bmag_norm<=B_bounce])
#        
#        x = np.linspace(-B_val_max, B_val_max, 10000)
#        y = integrand_interp(x)
#        plt.plot(np.abs(x), np.abs(y))
#        plt.savefig("tmp.pdf")
#        print(B_val_max)
#        Nr_B_points = int(1e3)
#
#        #for B_val in np.linspace(0, (1-eps)*B_bounce, Nr_B_points):
#        for B_val in np.linspace(0, B_val_max, Nr_B_points):
##            print(B_val)
#            net_radial_drift      += integrand_interp(B_val) + integrand_interp(-B_val)
#            net_radial_drift_norm += np.abs(integrand_interp(B_val)) + np.abs(integrand_interp(-B_val))
 
        return net_radial_drift/net_radial_drift_norm

    # Plot net radial drift as a function of zed_b
    def plot_net_radial_drift(self, fig=None, ax=None, label=None, ls=None, color=None):
        zed     = self.ncdata.variables['zed'][:]
        zed_pos = zed[zed>=0]

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        # Evaluate net drift
        net_radial_drift = np.zeros(len(zed_pos))
        for i_b, zed_b in enumerate(zed_pos):
            net_radial_drift[i_b] = self.evaluate_net_radial_drift(zed_b=zed_b)

        # Plot
        ax.plot(zed_pos, net_radial_drift, ls=ls, label=label, color=color)

        ax.set_xlabel(r"$\zeta_B$")
        ax.set_ylabel(r"$\Delta \psi$ (a.u.)")
        ax.set_xlim([0,np.pi])

        return fig, ax
        

    # Energy-averaged kperp_rho_i, to estimate FLR effect strength in NL sims
    def read_avg_kperp_rhoi(self, exclude_zonal=True, only_zonal=False, time_idx_jump=1):

        print("\n"+self.filename_base+":")

        # phi_vs_t(t, tube, zed, theta0, ky, ri)
        phi2_vs_t = np.abs( self.ncdata.variables['phi_vs_t'][::time_idx_jump,0,:,:,:,0] + 1j*self.ncdata.variables['phi_vs_t'][::time_idx_jump,0,:,:,:,1])**2
        time      = self.ncdata.variables['t'][::time_idx_jump] 
        Ntime     = len(time)
        zed       = self.ncdata.variables['zed'][:]
        # kperp2(zed, alpha, kx, ky)
        kperp2    = self.ncdata.variables['kperp2'][:,0,:,:]

#        ky        = self.ncdata.variables['ky'] 
#        kx        = self.ncdata.variables['kx'] 

        dl_over_B_avg = self.dl_over_B_avg()

        shat   = self.ncdata.variables['shat'].getValue()
#        gds2   = np.asarray(self.ncdata.variables['gds2'][:] )
#        gds21  = np.asarray(self.ncdata.variables['gds21'][:])
#        gds22  = np.asarray(self.ncdata.variables['gds22'][:])
  
        if exclude_zonal:
            phi2_vs_t[:,:,:,0] = 0
        if only_zonal:
            phi2_vs_t[:,:,:,1:] = 0

        # Avoid division by zero
        phi2_vs_t[:,:,0,0] = 0
        kperp2[:,0,0] = 1e16

#        # Get kperp along tube for all mode numbers
#        kperp_rhoi = np.zeros(shape=(len(zed),len(kx),len(ky))) 
#        for i_zed in range(len(zed)):
#            print("zed  index %i/%i" % (i_zed+1, len(zed)), end="\r")
#            for i_kx in range(len(kx)):
#                for i_ky in range(len(ky)):
#                    if i_ky == 0 and (exclude_zonal or i_kx == 0):
#                        kperp_rhoi[i_zed, i_kx, i_ky] = np.infty
#                    else:
#                        kperp_rhoi[i_zed, i_kx, i_ky] = np.sqrt( gds22[i_zed]/(shat*shat)*kx[i_kx]**2 + gds21[i_zed]/shat*kx[i_kx]*ky[i_ky] + gds2[i_zed]*ky[i_ky]**2 )

        # For all times, obtain energy-averaged kperp
        kperp2_O        = np.zeros(Ntime)
        kperp2_O_stddev = np.zeros(Ntime)
        for i_time in range(Ntime):
            print("Time index %i/%i" % (i_time+1, Ntime), end="\r")
            numerator   = np.sum(phi2_vs_t[i_time]/kperp2, axis=(1,2))
            denominator = np.sum(phi2_vs_t[i_time],        axis=(1,2))

            numerator_stddev   = np.sum(phi2_vs_t[i_time]**2 * (1/kperp2 - (numerator/denominator)[:,None,None])**2, axis=(1,2))

            # Tube-average
            kperp2_O[i_time]        = np.sum(               denominator/numerator * dl_over_B_avg)
            kperp2_O_stddev[i_time] = np.sum( np.sqrt(numerator_stddev)/numerator * dl_over_B_avg)

        return kperp2_O, kperp2_O_stddev, np.asarray(time)


    #######  Read real and imaginary frequency as a function of time and mode number
    def read_data_omega_k(self, timestep=-1, om_avg=True, check_convergence=True, nonconverged_to_none=True, delta_t_avg=None, t_val=None):
        kx   = self.ncdata.variables['kx'][:]
        ky   = self.ncdata.variables['ky'][:]
        dim_kx = len(kx)
        dim_ky = len(ky)

        # omega is in format [time ky kx Re[om] Im[om] Re[omavg] Im[omavg]]
        # omega_data has dim (N_time)*(N_ky)*(N_kx)*(7)
        if self.code == "stella":
            omega_data = np.loadtxt(self.omega_file, dtype='float').reshape(-1, dim_ky, dim_kx, 7)
            time_all = omega_data[:,0,0,0]

        elif self.code == "GX":
            #omega_v_time(time, ky, kx, ri)
            omega_v_time = self.ncdata['Special']['omega_v_time']

            time_all = self.get_time_array()
            kx, ky, zed = self.get_kx_ky_zed()

            omega_data = np.zeros( (len(time_all), len(ky), len(kx), 7) )
            omega_data[:,:,:,0] = time_all[:,None,None]
            omega_data[:,:,:,1] = ky[None,:,None]
            omega_data[:,:,:,2] = kx[None,None,:]
            omega_data[:,:,:,3] = omega_v_time[:,:,:,0]
            omega_data[:,:,:,4] = omega_v_time[:,:,:,1]

            om_avg = False

        # Make sure t_val is smaller or equal to the maximal time
        if t_val is not None:
            t_val = min(t_val, time_all[-1])

        if delta_t_avg is None:
            if t_val is None:
                omega_slice = omega_data[timestep]
            else:
                omega_slice = omega_data[np.argmin( np.abs(time_all - t_val) )]
        else:
            if t_val is None:
                omega_slice = np.mean(omega_data[ np.logical_and(time_all > time_all[timestep]-delta_t_avg, time_all <= time_all[timestep])], axis=0)
            else:
                omega_slice = np.mean(omega_data[ np.logical_and(time_all > t_val-delta_t_avg, time_all <= t_val)], axis=0)

 
        time     = omega_slice[:,:,0]
        ky       = omega_slice[:,:,1]
        kx       = omega_slice[:,:,2]
        if om_avg:
            omega_r = omega_slice[:,:,5]
            omega_i = omega_slice[:,:,6]
        else:
            omega_r = omega_slice[:,:,3]
            omega_i = omega_slice[:,:,4]


        if delta_t_avg is None and check_convergence and len(kx) == len(ky) == 1:
            omega_r_prev = omega_data[timestep-1][0][0][5]
            omega_i_prev = omega_data[timestep-1][0][0][6]

            diff_omega_r = np.abs( (omega_r - omega_r_prev)/omega_r )
            diff_omega_i = np.abs( (omega_i - omega_i_prev)/omega_i )

            threshold = 1e-1

            if diff_omega_r > threshold:
                print(self.filename_base + ": average omega_r = %e evolved by %.3f > threshold = %.3f in last step." % (omega_r, diff_omega_r, threshold))
                if nonconverged_to_none:
                    omega_r[:] = np.nan

            if diff_omega_i > threshold:
                print(self.filename_base + ": average omega_i = %e evolved by %.3f > threshold = %.3f in last step." % (omega_i, diff_omega_i, threshold))
                if nonconverged_to_none:
                    omega_i[:] = np.nan

        return time, ky, kx, omega_r, omega_i

    def read_omega_t(self, delta_t_avg=None):
        omega_data = np.loadtxt(self.omega_file)
        Nr_timesteps = len(omega_data)

        time    = np.zeros(Nr_timesteps)
        omega_r = np.zeros(Nr_timesteps)
        omega_i = np.zeros(Nr_timesteps)

        for i in range(Nr_timesteps):

            time[i], _, _, omega_r[i], omega_i[i] = self.read_data_omega_k(timestep=i, check_convergence=False, delta_t_avg=delta_t_avg)

        return time, omega_r, omega_i


    #######  Read electrostatic potential over the flux tube
    def read_phi_vs_zed(self, time_avg=None, time_idx=-1, normalise_phi=True, kx_idx=0, ky_idx=0, eval_real=True, squared=False, remove_zonal=False):


        # phi_vs_t(t, tube, zed, theta0, ky, ri)
        if time_avg is None:
            phi_vs_zed_theta0_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx,0]
        else:
            phi_vs_t_zed_theta0_ky_ri = self.ncdata.variables['phi_vs_t'][:,0]

            time   = self.ncdata.variables['t'][:]
            time_max = time[time_idx]
            phi_vs_zed_theta0_ky_ri = np.mean( phi_vs_t_zed_theta0_ky_ri[time > time_max-time_avg], axis=0)

        if eval_real:
            phi_vs_zed_theta0_ky = phi_vs_zed_theta0_ky_ri[:,:,:,0]
        else:
            phi_vs_zed_theta0_ky = np.abs(phi_vs_zed_theta0_ky_ri[:,:,:,0] + 1j*phi_vs_zed_theta0_ky_ri[:,:,:,1])

        if squared:
            phi_vs_zed_theta0_ky = phi_vs_zed_theta0_ky**2

        if remove_zonal:
            phi_vs_zed_theta0_ky[:,:,0] = 0
    
        if ky_idx is not None:
            phi_vs_zed_theta0 = phi_vs_zed_theta0_ky[:,:,ky_idx]
        else:
            phi_vs_zed_theta0 = np.sum(phi_vs_zed_theta0_ky, axis=2)
        
        if kx_idx is not None:
            phi_vs_zed = phi_vs_zed_theta0[:,kx_idx]
        else:
            phi_vs_zed = np.sum(phi_vs_zed_theta0, axis=1)

        if normalise_phi:
            max_phi = np.max(phi_vs_zed)
            min_phi = np.min(phi_vs_zed)
            if np.abs(max_phi) > np.abs(min_phi):
                phi_vs_zed = phi_vs_zed / max_phi
            else:
                phi_vs_zed = phi_vs_zed / min_phi
            
        zed      = self.ncdata.variables['zed'][:]

        return phi_vs_zed, zed

    #######  Read (non-adiabatic?) distribution function in flux tube, averaged over other directions
    def read_g_vs_zed(self, time_idx=-1, species_idx=0, vpa_index=None, normalise=True):

        # gzvs(t, species, vpa, zed, tube) ;
        gzvs = self.ncdata.variables['gzvs'][time_idx,species_idx,:,:,0]

        # vpa?
        vpa = self.ncdata.variables['vpa']
        #print(vpa[vpa_index])
        if vpa_index is None:
            gz = np.sum(np.abs(gzvs), axis=0)
        else:
            gz = gzvs[vpa_index,:]
        #gz = np.sum(gzvs, axis=0)

        if normalise:
            max_g = np.max(gz)
            min_g = np.min(gz)
            if np.abs(max_g) > np.abs(min_g):
                gz = gz / max_g
            else:
                gz = gz / min_g

        zed      = self.ncdata.variables['zed'][:]

        return gz, zed

    #######  Read fluxes as a function of mode numbers and time
    def read_flux_spectra(self, species_idx=0, tube=0):#, zed_slice=None, kx_slice=None, ky_slice=None, t_slice=None):

    	# qflx_kxky(t, species, tube, zed, kx, ky)
        if self.code == "stella":
            qflx_t_zed_kx_ky = self.ncdata.variables['qflx_kxky'][:,species_idx, tube, :, :, :]
        elif self.code == "GX":
            qflx_t_zed_kx_ky = np.transpose(self.ncdata['Diagnostics']['HeatFlux_kxkyzst'][:,species_idx, :, :, :], axes=(0,1,3,2))

        time = self.get_time_array()
        kx, ky, zed = self.get_kx_ky_zed()

        # Check shape
        assert len(time) == np.shape(qflx_t_zed_kx_ky)[0]
        assert len(zed)  == np.shape(qflx_t_zed_kx_ky)[1]
        assert len(kx)   == np.shape(qflx_t_zed_kx_ky)[2]
        assert len(ky)   == np.shape(qflx_t_zed_kx_ky)[3]

        return qflx_t_zed_kx_ky, time, zed, kx, ky

    #######  Read electrostatic potential as a function of mode numbers and time
    def read_phi2_spectra(self, time_min=0, time_max=1e10, time_idx_skip=1):

        time = self.get_time_array()
        time_idx_min = self.get_time_idx(time_min)
        time_idx_max = self.get_time_idx(time_max)
        time = time[time_idx_min:time_idx_max:time_idx_skip]
        kx, ky, zed = self.get_kx_ky_zed()

        if self.code == "stella":
            # phi2_vs_kxky(t, kx, ky)
            phi2_vs_kxky = self.ncdata.variables['phi2_vs_kxky'][time_idx_min:time_idx_max:time_idx_skip]

        elif self.code == "GS2":
            phi2_vs_kxky = np.transpose(self.ncdata['phi2_by_mode'][time_idx_min:time_idx_max:time_idx_skip], axes=(0,2,1))
    
        elif self.code == "GX":
            if self.GX_old_version:
                phi2_vs_kxky = np.transpose(self.ncdata['Spectra']['Akxkyst'][time_idx_min:time_idx_max:time_idx_skip,:,:], axes=(0,2,1))
            else:
                phi2_vs_kxky = np.transpose(self.ncdata['Diagnostics']['Wphi_kxkyst'][time_idx_min:time_idx_max:time_idx_skip,0,:,:], axes=(0,2,1))

            # divide zonal by (1-Gamma0(kx(GX)**2))
            Gamma0 = specialfunc.iv(0, kx**2/2) * np.exp(-kx**2/2)
            #Gamma0 = specialfunc.iv(0, kx**2/4) * np.exp(-kx**2/4)
            phi2_vs_kxky[:,:,0] = phi2_vs_kxky[:,:,0]/(1-Gamma0)[None,:]

            # Factor from vT definition
            #phi2_vs_kxky = phi2_vs_kxky/4

        # Check shape
        assert len(time) == np.shape(phi2_vs_kxky)[0]
        assert len(kx)   == np.shape(phi2_vs_kxky)[1]
        assert len(ky)   == np.shape(phi2_vs_kxky)[2]

        # Make (0,0) mode NaN
#        for i_t in range(len(time)):
#            phi2_vs_kxky[i_t, 0, 0] = np.nan

        return phi2_vs_kxky, time, kx, ky

    #######  Read energy as a function of mode numbers and time
    def read_W_spectra(self, time_min=0, time_max=1e10, time_idx_skip=1):

        kx, ky, _ = self.get_kx_ky_zed()
        if self.code == "stella":
            # Consider only temperature and parallel flow energy
            time = self.get_time_array()
            time_idx_min = self.get_time_idx(time_min)
            time_idx_max = self.get_time_idx(time_max)
            time = time[time_idx_min:time_idx_max:time_idx_skip]

            # temperature(t, species, tube, zed, kx, ky, ri)
            #temperature_t_s_zed_kx_ky_ri = self.ncdata.variables['temperature'][time_idx_min:time_idx_max:time_idx_skip,:,0,:,:,:,:]
            #temperature_t_s_zed_kx_ky = temperature_t_s_zed_kx_ky_ri[:,:,:,:,:,0]+1j*temperature_t_s_zed_kx_ky_ri[:,:,:,:,:,1]
            upar_t_s_zed_kx_ky_ri = self.ncdata.variables['upar'][time_idx_min:time_idx_max:time_idx_skip,:,0,:,:,:,:]
            upar_t_s_zed_kx_ky = upar_t_s_zed_kx_ky_ri[:,:,:,:,:,0]+1j*upar_t_s_zed_kx_ky_ri[:,:,:,:,:,1]
            dl_over_B_avg = self.dl_over_B_avg()
            W_vs_kxky = np.sum( np.abs(upar_t_s_zed_kx_ky)**2 * dl_over_B_avg[None,None,:,None,None], axis=(1,2))
            #W_vs_kxky = np.sum( (np.abs(temperature_t_s_zed_kx_ky)**2+np.abs(upar_t_s_zed_kx_ky)**2) * dl_over_B_avg[None,None,:,None,None], axis=(1,2))
    
        elif self.code == "GX":
            #if get_kx:
            #    W_vs_kx = self.ncdata['Spectra']['Wkxst'][:,0,:]
            #    W_vs_kxky = W_vs_kx[:,:,None]
            #else:
            #    W_vs_ky = self.ncdata['Spectra']['Wkyst'][:,0,:]
            #    W_vs_kxky = W_vs_ky[:,None,:]

            time = self.get_time_array()
            time_idx_min = self.get_time_idx(time_min)
            time_idx_max = self.get_time_idx(time_max)
            time = time[time_idx_min:time_idx_max:time_idx_skip]
            if self.GX_old_version:
                W_vs_kxky = np.transpose(self.ncdata['Spectra']['Wkxkyst'][time_idx_min:time_idx_max:time_idx_skip,0,:,:], axes=(0,2,1))
            else:
                W_vs_kxky = np.transpose(self.ncdata['Diagnostics']['Wg_kxkyst'][time_idx_min:time_idx_max:time_idx_skip,0,:,:], axes=(0,2,1))

        # Check shape
        #assert len(time) == np.shape(W_vs_kxky)[0]
        #assert len(kx)   == np.shape(W_vs_kxky)[1]
        #assert len(ky)   == np.shape(W_vs_kxky)[2]

        # Make (0,0) mode NaN
#        for i_t in range(len(time)):
#            phi2_vs_kxky[i_t, 0, 0] = np.nan

        return W_vs_kxky, time, kx, ky

    #######  Read flux-tube averaged zonal potential and derivatives as a function of mode numbers and time
    def read_phi_zonal_spectra(self):

        # phi_vs_t(t, tube, zed, theta0, ky, ri)
        phiZF_vs_t  = self.ncdata.variables['phi_vs_t'][:,0,:,:,0,:] #(t, zed, kx, ri)
        time        = self.ncdata.variables['t'] 
        kx          = self.ncdata.variables['kx'][:] 
        gds22       = self.ncdata.variables['gds22'][:,0] # |nabla(x)|^2
        nablax      = np.sqrt(gds22)

        # Zonal flow derivatives
        phiZF_prime_vs_t  = 1j*kx[None,None,:,None]*nablax[None,:,None,None]*phiZF_vs_t
        phiZF_dprime_vs_t = - (kx[None,None,:,None]*nablax[None,:,None,None])**2 *phiZF_vs_t

        # Tube average and absolute value
        dl_over_B_avg = self.dl_over_B_avg()
        phiZF_vs_t        = np.sum( dl_over_B_avg[None,:,None] * np.abs(phiZF_vs_t[:,:,:,0]       +1j*phiZF_vs_t[:,:,:,1])       , axis=1)
        phiZF_prime_vs_t  = np.sum( dl_over_B_avg[None,:,None] * np.abs(phiZF_prime_vs_t[:,:,:,0] +1j*phiZF_prime_vs_t[:,:,:,1]) , axis=1)
        phiZF_dprime_vs_t = np.sum( dl_over_B_avg[None,:,None] * np.abs(phiZF_dprime_vs_t[:,:,:,0]+1j*phiZF_dprime_vs_t[:,:,:,1]), axis=1)

        return phiZF_vs_t, phiZF_prime_vs_t, phiZF_dprime_vs_t, time, kx

    #######  Read phi squared as a function of length along tube and time
    def read_phi2_vs_t_zed(self, tube=0, remove_zonal=False, only_zonal=False, kx_zonal=True, time_min=0, time_max=1e6):

        time = self.get_time_array()
        if time_min < 0:
            time_min = time[-1] - np.abs(time_min)
        time_idx_min = self.get_time_idx(time_min)
        time_idx_max = self.get_time_idx(time_max)
        time = time[time_idx_min:time_idx_max]
        # phi_vs_t(t, tube, zed, theta0, ky, ri) (ri=real,imaginary)
        phi_vs_t = self.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max,0,:,:,:]
        zed  = self.ncdata.variables['zed']
        kx   = self.ncdata.variables['kx'][:]

        if remove_zonal:
            phi_vs_t[:,:,:,0,:] = 0
        if only_zonal:
            phi_vs_t[:,:,:,1:,:] = 0
            if kx_zonal:
                phi_vs_t = phi_vs_t*kx[None,None,:,None,None]

        phi2 = phi_vs_t[:,:,:, :,0]**2 + phi_vs_t[:,:,:, :,1]**2

        phi2_vs_t_zed = np.sum(phi2, axis=(2,3))

        return phi2_vs_t_zed, time, zed

    #######  Read flux-surface averaged phi squared as a function of time
    def read_phi2_vs_t(self, tube=0):

        # phi2(t)
        phi2_t = self.ncdata.variables['phi2'][:]
        time = self.ncdata.variables['t'][:]

        return time, phi2_t

    #######  Get average kperp2 in tube
    def get_avg_kperp2(self, ky_idx=0, kx_idx=0):
        kperp2 = self.ncdata.variables['kperp2'][:][:,0,kx_idx,ky_idx] 
        dl_over_B_avg = self.dl_over_B_avg()

        return np.sum(kperp2*dl_over_B_avg)

    #######  Get Rosenbluth-Hinton inertia terms
    def get_RH_inertia(self, species_idx="sum", kx_max=1e5, idxs_kx=None):

        # kx indices
        kx_all, _, zed = self.get_kx_ky_zed()
        if idxs_kx is None:
            idxs_kx = np.arange(len(kx_all))
        idxs_kx = idxs_kx[ np.abs(kx_all[idxs_kx]) <= kx_max ]
        kx_vals = kx_all[idxs_kx]

        if species_idx == "sum":
            # Sum over all species to get total RH inertia
            try:
                nspecies = len(self.ncdata.dimensions['species'])
            except:
                nspecies = 1

            RH_inertia_zed_kx_ri = self.ncdata.variables['RH_inertia'][0,0,:,idxs_kx,:]

            for i_spec in np.arange(nspecies-1):
                RH_inertia_zed_kx_ri += self.ncdata.variables['RH_inertia'][i_spec+1,0,:,idxs_kx,:]

        else:
            # Evaluate RH inertia from one species only
            RH_inertia_zed_kx_ri = self.ncdata.variables['RH_inertia'][species_idx,0,:,idxs_kx,:]

        # Convert to complex
        RH_inertia_zed_kx = RH_inertia_zed_kx_ri[:,:,0] + 1j*RH_inertia_zed_kx_ri[:,:,1]

        return RH_inertia_zed_kx, zed, kx_vals

    #######  Get Rosenbluth-Hinton fluxes (even and odd in vparallel)
    def get_RH_fluxes(self, species_idx="sum", passing_trapped="both", time_min=0, time_max=1e10, time_idx_skip=1, kx_max=1e5, idxs_kx=None, fphi=1, fapar=1, fbpar=1, fcoll=1):

        # Determine time indices
        time_idx_min = self.get_time_idx(time_min)
        time_idx_max = self.get_time_idx(time_max)
        time_all = self.get_time_array()
        time = time_all[time_idx_min:time_idx_max-1:time_idx_skip]

        # kx indices
        kx_all, ky, zed = self.get_kx_ky_zed()
        if idxs_kx is None:
            idxs_kx = np.arange(len(kx_all))
        idxs_kx = idxs_kx[ np.abs(kx_all[idxs_kx]) <= kx_max ]
        kx_vals = kx_all[idxs_kx]

        try:
            nspecies = len(self.ncdata.dimensions['species'])
        except:
            nspecies = 1

        # Get shape of array
        try:
            RH_fluxes_phi_even_t_zed_kx_ky_ri = self.ncdata.variables['RH_fluxes_phi_even'][time_idx_min:time_idx_max-1:time_idx_skip,0,0,:,idxs_kx,:,:]
        except:
            # For backwards compatibility (<1st April 2026)
            RH_fluxes_phi_even_t_zed_kx_ky_ri = self.ncdata.variables['RH_fluxes_phi_even_passing'][time_idx_min:time_idx_max-1:time_idx_skip,0,0,:,idxs_kx,:,:]
        
        # Start with zero-filled arrays
        RH_fluxes_phi_even_t_zed_kx_ky_ri[:] = 0
        RH_fluxes_phi_odd_t_zed_kx_ky_ri   = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)
        RH_fluxes_apar_even_t_zed_kx_ky_ri = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)
        RH_fluxes_apar_odd_t_zed_kx_ky_ri  = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)
        RH_fluxes_bpar_even_t_zed_kx_ky_ri = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)
        RH_fluxes_bpar_odd_t_zed_kx_ky_ri  = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)
        RH_fluxes_coll_t_zed_kx_ky_ri      = np.zeros_like(RH_fluxes_phi_even_t_zed_kx_ky_ri)

        # Sum over species
        for i_spec in np.arange(nspecies):

            if species_idx == "sum" or species_idx == i_spec:

                try:

                    if passing_trapped == "passing" or passing_trapped == "both":
                        RH_fluxes_phi_even_t_zed_kx_ky_ri += self.ncdata.variables['RH_fluxes_phi_even_passing'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        RH_fluxes_phi_odd_t_zed_kx_ky_ri  += self.ncdata.variables['RH_fluxes_phi_odd_passing'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        try:
                            RH_fluxes_apar_even_t_zed_kx_ky_ri += self.ncdata.variables['RH_fluxes_apar_even_passing'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                            RH_fluxes_apar_odd_t_zed_kx_ky_ri  += self.ncdata.variables['RH_fluxes_apar_odd_passing'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        except:
                            pass

                        try:
                            RH_fluxes_bpar_even_t_zed_kx_ky_ri += self.ncdata.variables['RH_fluxes_bpar_even_passing'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                            RH_fluxes_bpar_odd_t_zed_kx_ky_ri  += self.ncdata.variables['RH_fluxes_bpar_odd_passing'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        except:
                            pass


                    if passing_trapped == "trapped" or passing_trapped == "both":
                        RH_fluxes_phi_even_t_zed_kx_ky_ri += self.ncdata.variables['RH_fluxes_phi_even_trapped'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        RH_fluxes_phi_odd_t_zed_kx_ky_ri  += self.ncdata.variables['RH_fluxes_phi_odd_trapped'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        try:
                            RH_fluxes_apar_even_t_zed_kx_ky_ri += self.ncdata.variables['RH_fluxes_apar_even_trapped'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                            RH_fluxes_apar_odd_t_zed_kx_ky_ri  += self.ncdata.variables['RH_fluxes_apar_odd_trapped'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        except:
                            pass

                        try:
                            RH_fluxes_bpar_even_t_zed_kx_ky_ri += self.ncdata.variables['RH_fluxes_bpar_even_trapped'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                            RH_fluxes_bpar_odd_t_zed_kx_ky_ri  += self.ncdata.variables['RH_fluxes_bpar_odd_trapped'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        except:
                            pass


                # For backwards compatibility (<1st April 2026)
                except:
                    RH_fluxes_phi_even_t_zed_kx_ky_ri += self.ncdata.variables['RH_fluxes_phi_even'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    RH_fluxes_phi_odd_t_zed_kx_ky_ri  += self.ncdata.variables['RH_fluxes_phi_odd'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    try:
                        RH_fluxes_apar_even_t_zed_kx_ky_ri += self.ncdata.variables['RH_fluxes_apar_even'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        RH_fluxes_apar_odd_t_zed_kx_ky_ri  += self.ncdata.variables['RH_fluxes_apar_odd'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    except:
                        pass

                    try:
                        RH_fluxes_bpar_even_t_zed_kx_ky_ri += self.ncdata.variables['RH_fluxes_bpar_even'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                        RH_fluxes_bpar_odd_t_zed_kx_ky_ri  += self.ncdata.variables['RH_fluxes_bpar_odd'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:,:]
                    except:
                        pass

                # Collisional flux
                try:
                    tmp  = self.ncdata.variables['RH_fluxes_collisional'][ time_idx_min:time_idx_max-1:time_idx_skip,i_spec,0,:,idxs_kx,:]

                    RH_fluxes_coll_t_zed_kx_ky_ri[:,:,:,0,:]  += tmp

#                    # Normalise by -1/(i*kx) to get P_RH contribution in same way as NL fluxes
#                    RH_fluxes_coll_t_zed_kx_ky_ri[:,:,:,0,0]  += -tmp[:,:,:,1]/kx_vals[None,None,:]
#                    RH_fluxes_coll_t_zed_kx_ky_ri[:,:,:,0,1]  +=  tmp[:,:,:,0]/kx_vals[None,None,:]
                
                except Exception as e:
                    print(e)
                    pass

        # Add up contributions from {phi, Apar, Bpar}
        RH_fluxes_even_t_zed_kx_ky_ri = fphi*RH_fluxes_phi_even_t_zed_kx_ky_ri + fapar*RH_fluxes_apar_even_t_zed_kx_ky_ri + fbpar*RH_fluxes_bpar_even_t_zed_kx_ky_ri + fcoll*RH_fluxes_coll_t_zed_kx_ky_ri
        RH_fluxes_odd_t_zed_kx_ky_ri  = fphi*RH_fluxes_phi_odd_t_zed_kx_ky_ri  + fapar*RH_fluxes_apar_odd_t_zed_kx_ky_ri  + fbpar*RH_fluxes_bpar_odd_t_zed_kx_ky_ri

        RH_fluxes_even_t_zed_kx_ky = RH_fluxes_even_t_zed_kx_ky_ri[:,:,:,:,0] + 1j*RH_fluxes_even_t_zed_kx_ky_ri[:,:,:,:,1]
        RH_fluxes_odd_t_zed_kx_ky = RH_fluxes_odd_t_zed_kx_ky_ri[:,:,:,:,0] + 1j*RH_fluxes_odd_t_zed_kx_ky_ri[:,:,:,:,1]

        return RH_fluxes_even_t_zed_kx_ky, RH_fluxes_odd_t_zed_kx_ky, time, zed, kx_vals, ky
               

    #######  Get Rosenbluth-Hinton phi*I
    def get_RH_phi_I(self, species_idx="sum", time_min=0, time_max=1e10, time_idx_skip=1, kx_max=1e5, idxs_kx=None):

        # Determine time indices
        time_idx_min = self.get_time_idx(time_min)
        time_idx_max = self.get_time_idx(time_max)
        time_all = self.get_time_array()
        time = time_all[time_idx_min:time_idx_max-1:time_idx_skip]

        # kx indices
        kx_all, ky, zed = self.get_kx_ky_zed()
        if idxs_kx is None:
            idxs_kx = np.arange(len(kx_all))

        idxs_kx = idxs_kx[ np.abs(kx_all[idxs_kx]) <= kx_max ]
        kx_vals = kx_all[idxs_kx]

        # Load RH_phi_I
        if species_idx == "sum":
            # Sum over all species to get total RH phi*I
            try:
                nspecies = len(self.ncdata.dimensions['species'])
            except:
                nspecies = 1

            RH_phi_I_t_zed_kx_ri = self.ncdata.variables['RH_phi_I'][time_idx_min:time_idx_max-1:time_idx_skip,0,0,:,idxs_kx,:]

            for i_spec in np.arange(nspecies-1):
                RH_phi_I_t_zed_kx_ri += self.ncdata.variables['RH_phi_I'][time_idx_min:time_idx_max-1:time_idx_skip,i_spec+1,0,:,idxs_kx,:]

        else:
            # Evaluate RH phi*I from one species only
            RH_phi_I_t_zed_kx_ri = self.ncdata.variables['RH_phi_I'][time_idx_min:time_idx_max-1:time_idx_skip,species_idx,0,:,idxs_kx,:]

        RH_phi_I_t_zed_kx = RH_phi_I_t_zed_kx_ri[:,:,:,0] + 1j*RH_phi_I_t_zed_kx_ri[:,:,:,1]

        return RH_phi_I_t_zed_kx, time, zed, kx_vals

    #######  Get RH fluxes in t and kx only
    def get_RH_fluxes_t_kx(self, species_idx="sum", passing_trapped="both", time_min=0, time_max=1e10, time_idx_skip=1, kx_max=1e5, idxs_kx=None, fphi=1, fapar=1, fbpar=1, fcoll=1):

        RH_fluxes_even_t_zed_kx_ky, RH_fluxes_odd_t_zed_kx_ky,\
                time, zed, kx_vals, ky_vals \
                = self.get_RH_fluxes(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll)

        dl_over_B_avg = self.dl_over_B_avg()

        RH_fluxes_even_t_kx = np.sum(RH_fluxes_even_t_zed_kx_ky*dl_over_B_avg[None,:,None,None], axis=(1,3))
        RH_fluxes_odd_t_kx  = np.sum(RH_fluxes_odd_t_zed_kx_ky* dl_over_B_avg[None,:,None,None], axis=(1,3))

        return RH_fluxes_even_t_kx, RH_fluxes_odd_t_kx, time, kx_vals

    #######  Get RH phi_I in t and kx only
    def get_RH_phi_I_t_kx(self, species_idx="sum", time_min=0, time_max=1e10, time_idx_skip=1, kx_max=1e5, idxs_kx=None):

        RH_phi_I_t_zed_kx, time, zed, kx_vals \
                = self.get_RH_phi_I(species_idx=species_idx, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx)

        dl_over_B_avg = self.dl_over_B_avg()

        RH_phi_I_t_kx = np.sum(RH_phi_I_t_zed_kx*dl_over_B_avg[None,:,None], axis=1)

        return RH_phi_I_t_kx, time, kx_vals

    #######  Get RH energy in t and kx only
    def get_E_RH_t_kx(self, species_idx="sum", time_min=0, time_max=1e10, kx_max=1e5, idxs_kx=None):

        RH_phi_I_t_zed_kx, time, zed, kx_vals \
                = self.get_RH_phi_I(species_idx=species_idx, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)

        dl_over_B_avg = self.dl_over_B_avg()

        RH_phi_I_t_kx = np.sum(RH_phi_I_t_zed_kx*dl_over_B_avg[None,:,None], axis=1)

        RH_inertia_zed_kx, zed, kx = self.get_RH_inertia(species_idx=species_idx, kx_max=kx_max, idxs_kx=idxs_kx)
        RH_inertia_kx = np.sum(dl_over_B_avg[:,None]*RH_inertia_zed_kx, axis=0)

        # Evaluate 1-Gamma0 (single species!)
        Gamma0_vals = np.zeros_like(kx)
        for i_kx, kx_val in enumerate(kx):
            shat   = self.ncdata.variables['shat'].getValue()
            gds22  = self.ncdata.variables['gds22'][:,0]/shat**2 # |nabla(x)|^2
            bmag   = self.ncdata.variables['bmag'][:,0]
            kperp2 = (kx_val/bmag)**2 * gds22
            Gamma0_vals[i_kx] = np.sum(dl_over_B_avg * specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2))

        E_RH_t_kx =  np.abs(RH_phi_I_t_kx)**2 / (2*np.abs(RH_inertia_kx[None,:])**2) * (1-Gamma0_vals)[None, :]

        return E_RH_t_kx, time, kx_vals

    #######  Get predicted time derivative of E_RH due to Rosenbluth-Hinton fluxes (even and odd in vparallel)
    def get_P_RH(self, species_idx="sum", passing_trapped="both", time_min=0, time_max=1e10, time_idx_skip=1, kx_max=1e5, idxs_kx=None, fphi=1, fapar=1, fbpar=1, fcoll=1):

        RH_fluxes_even_t_kx, RH_fluxes_odd_t_kx, time, kx = \
                self.get_RH_fluxes_t_kx(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll)

        dl_over_B_avg = self.dl_over_B_avg()

        # Evaluate RH inertia
        RH_inertia_zed_kx, zed, kx = self.get_RH_inertia(species_idx=species_idx, kx_max=kx_max, idxs_kx=idxs_kx)
        RH_inertia_kx = np.sum(dl_over_B_avg[:,None]*RH_inertia_zed_kx, axis=0)

        # Obtain phi_RH*I_RH from simulation
        RH_phi_I_t_kx, time, kx_vals = self.get_RH_phi_I_t_kx(species_idx=species_idx, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx)

        # Evaluate 1-Gamma0 (single species!)
        Gamma0_vals = np.zeros_like(kx)
        for i_kx, kx_val in enumerate(kx):
            shat   = self.ncdata.variables['shat'].getValue()
            gds22  = self.ncdata.variables['gds22'][:,0]/shat**2 # |nabla(x)|^2
            bmag   = self.ncdata.variables['bmag'][:,0]
            kperp2 = (kx_val/bmag)**2 * gds22
            Gamma0_vals[i_kx] = np.sum(dl_over_B_avg * specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2))

        P_RH_even_t_kx = -np.real(1j*kx[None,:]*RH_fluxes_even_t_kx*np.conj(RH_phi_I_t_kx)) / np.abs(RH_inertia_kx[None,:])**2 * (1-Gamma0_vals)[None,:]
        P_RH_odd_t_kx  = -np.real(1j*kx[None,:]*RH_fluxes_odd_t_kx *np.conj(RH_phi_I_t_kx)) / np.abs(RH_inertia_kx[None,:])**2 * (1-Gamma0_vals)[None,:]

        return P_RH_even_t_kx, P_RH_odd_t_kx, time, kx

    #######  Plot Rosenbluth-Hinton energy and compare to ZF energy
    def plot_E_RH(self, fig=None, ax=None, time_min=0, time_max=1e10, idxs_kx=None, kx_max=1e5, colors=None):

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(9,8))

        E_RH_t_kx, time, kx_vals = self.get_E_RH_t_kx(time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)

        # kx indices
        kx_all, ky, zed = self.get_kx_ky_zed()
        if idxs_kx is None:
            idxs_kx = np.arange(len(kx_vals))
        idxs_kx = idxs_kx[ np.abs(kx_all[idxs_kx]) <= kx_max ]
        kx_vals = kx_all[idxs_kx]

        # Evaluate phiZ
        time_idx_min = self.get_time_idx(time_min)
        time_idx_max = self.get_time_idx(time_max)
        phiZ_t_zed_kx_ri = self.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max-1,0,:,idxs_kx,0,:]
        phiZ_t_zed_kx = phiZ_t_zed_kx_ri[:,:,:,0]+1j*phiZ_t_zed_kx_ri[:,:,:,1]

        # Evaluate RH inertia
        RH_inertia_zed_kx, zed, kx = self.get_RH_inertia(species_idx="sum", kx_max=kx_max, idxs_kx=idxs_kx)

        dl_over_B_avg = self.dl_over_B_avg()

        phiZ_IRH_t_kx = np.sum((RH_inertia_zed_kx*dl_over_B_avg[:,None])[None,:,:]*phiZ_t_zed_kx, axis=1)
        RH_inertia_kx = np.sum(dl_over_B_avg[:,None]*RH_inertia_zed_kx, axis=0)

        # Evaluate 1-Gamma0 (single species!)
        Gamma0_vals = np.zeros_like(kx)
        for i_kx, kx_val in enumerate(kx):
            shat   = self.ncdata.variables['shat'].getValue()
            gds22  = self.ncdata.variables['gds22'][:,0]/shat**2 # |nabla(x)|^2
            bmag   = self.ncdata.variables['bmag'][:,0]
            kperp2 = (kx_val/bmag)**2 * gds22
            Gamma0_vals[i_kx] = np.sum(dl_over_B_avg * specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2))

        E_Z_t_kx = np.abs(phiZ_IRH_t_kx)**2 / (2*np.abs(RH_inertia_kx[None,:])**2) * (1-Gamma0_vals)

        if colors is None:
            colors = sns.color_palette("coolwarm", len(kx_vals))

        for i_kx, kx in enumerate(kx_vals):

            if kx <= 0:
                continue

            ax.plot(time, E_RH_t_kx[:,i_kx], 
                    label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx], lw=2)
            ax.plot(time, E_Z_t_kx[:,i_kx], 
                                                           c=colors[i_kx], ls='--')

            #diff = np.abs(E_RH_t_kx[:,i_kx] - E_Z_t_kx[:,i_kx])/E_RH_t_kx[:,i_kx]
            #axs[1].plot(time, diff, c=colors[i_kx])

        ax.set_ylabel(r"$E_{RH}$ (solid), $E^Z_\varphi$ (dashed)")
        #axs[1].set_ylabel(r"$|E_{RH}-E^Z_\varphi|/E_{RH}$")
        ax.set_xlabel(r"$t v_{T}/a$")

        #for ax in axs:
        ax.grid(True)

        return fig, ax, E_RH_t_kx, time, kx


    #######  Plot Rosenbluth-Hinton phi_I and compare to phi*I
    def plot_RH_phi_I(self, fig=None, axs=None, time_min=0, time_max=1e10, idxs_kx=None, kx_max=1e5, colors=None, colors_sim=None):

        if axs is None:
            fig, axs = plt.subplots(nrows=3,ncols=1, figsize=(6,16))
            #fig, axs = plt.subplots(nrows=3,ncols=1, figsize=(9,16))

        RH_phi_I_t_kx, time, kx_vals = self.get_RH_phi_I_t_kx(time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)

        if colors is None:
            if len(idxs_kx) > 1:
                colors = sns.color_palette("coolwarm", len(kx_vals))
            else:
                colors = ["crimson"]
        if colors_sim is None:
            if len(idxs_kx) > 1:
                colors_sim = colors
            else:
                colors_sim = ["mediumblue"]

        # Evaluate RH inertia
        RH_inertia_zed_kx, zed, kx_vals = self.get_RH_inertia(species_idx="sum", kx_max=kx_max, idxs_kx=idxs_kx)

        dl_over_B_avg = self.dl_over_B_avg()

        # Evaluate phiZ
        if idxs_kx is None:
            idxs_kx = np.arange(len(kx_vals))

        time_idx_min = self.get_time_idx(time_min)
        time_idx_max = self.get_time_idx(time_max)
        phiZ_t_zed_kx_ri = self.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max-1,0,:,idxs_kx,0,:]
        phiZ_t_zed_kx_all = phiZ_t_zed_kx_ri[:,:,:,0]+1j*phiZ_t_zed_kx_ri[:,:,:,1]
        kx_all, _, _ = self.get_kx_ky_zed()
        kx_all = kx_all[idxs_kx]

        phiZ_t_zed_kx = phiZ_t_zed_kx_all[:,:,np.abs(kx_all)<=kx_max]
        kx = kx_all[np.abs(kx_all)<=kx_max]

        phiZ_IRH_t_kx = np.sum((RH_inertia_zed_kx*dl_over_B_avg[:,None])[None,:,:]*phiZ_t_zed_kx, axis=1)

        rel_diff_t_kx = (RH_phi_I_t_kx - phiZ_IRH_t_kx)/np.abs(RH_phi_I_t_kx)

        for i_kx, kx in enumerate(kx_vals):

            axs[0].plot(time, np.real(RH_phi_I_t_kx[:,i_kx]), 
                        label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
            axs[0].plot(time, np.imag(RH_phi_I_t_kx[:,i_kx]), 
                        ls="--", c=colors[i_kx])

            axs[1].plot(time, np.real(phiZ_IRH_t_kx[:,i_kx]), 
                        label=r"$k_x \rho = %.3f$" % (kx), c=colors_sim[i_kx])
            axs[1].plot(time, np.imag(phiZ_IRH_t_kx[:,i_kx]), 
                        ls="--", c=colors_sim[i_kx])

            axs[2].plot(time, np.abs(RH_phi_I_t_kx[:,i_kx]), 
                        label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
            axs[2].plot(time, np.abs(phiZ_IRH_t_kx[:,i_kx]), 
                        label=r"$k_x \rho = %.3f$" % (kx), c=colors_sim[i_kx], alpha=0.5)

            #axs[2].plot(time, np.real(rel_diff_t_kx[:,i_kx]), 
            #            label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
            #axs[2].plot(time, np.imag(rel_diff_t_kx[:,i_kx]), 
            #            ls="--", c=colors[i_kx])

        axs[0].set_ylabel(r"$\varphi_\mathrm{RH} I_\mathrm{RH}$")
        axs[1].set_ylabel(r"$\varphi I_{RH}$")
        axs[2].set_ylabel(r"Both")
        #axs[2].set_ylabel(r"Relative diff")
        axs[2].set_xlabel(r"$t v_{T}/a$")

        for ax in axs:
            ax.grid(True)

        return fig, axs, RH_phi_I_t_kx, time, kx


    #######  Plot Rosenbluth-Hinton fluxes (even and odd in vparallel)
    def plot_RH_fluxes(self, fig=None, axs=None, time_min=0, time_max=1e10, species_idx="sum", passing_trapped="both", idxs_kx=None, kx_max=1e5, colors=None, fphi=1, fapar=1, fbpar=1, fcoll=1):

        if axs is None:
            fig, axs = plt.subplots(nrows=3,ncols=1, figsize=(9,16))

        RH_fluxes_even_t_kx, RH_fluxes_odd_t_kx, time, kx_vals = \
                self.get_RH_fluxes_t_kx(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll)

        if colors is None:
            colors = sns.color_palette("coolwarm", len(kx_vals))

        for i_kx, kx in enumerate(kx_vals):

            axs[0].plot(time, np.real(RH_fluxes_even_t_kx[:,i_kx]), 
                        label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
            axs[0].plot(time, np.imag(RH_fluxes_even_t_kx[:,i_kx]), 
                        ls="--", c=colors[i_kx])
            axs[1].plot(time, np.real(RH_fluxes_odd_t_kx[:,i_kx]),  
                        label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
            axs[1].plot(time, np.imag(RH_fluxes_odd_t_kx[:,i_kx]), 
                        ls="--", c=colors[i_kx])
            axs[2].plot(time, np.real(RH_fluxes_even_t_kx[:,i_kx]+RH_fluxes_odd_t_kx[:,i_kx]),  
                        label=r"$k_x \rho = %.3f$" % (kx), c=colors[i_kx])
            axs[2].plot(time, np.imag(RH_fluxes_even_t_kx[:,i_kx]+RH_fluxes_odd_t_kx[:,i_kx]), 
                        ls="--", c=colors[i_kx])

        axs[0].set_ylabel(r"$F_\mathrm{RH}^+$")
        axs[1].set_ylabel(r"$F_\mathrm{RH}^-$")
        axs[2].set_ylabel(r"$F_\mathrm{RH}^+ + F_\mathrm{RH}^-$")
        axs[2].set_xlabel(r"$t v_{T}/a$")

        for ax in axs:
            ax.grid(True)

        return fig, axs, RH_fluxes_even_t_kx, RH_fluxes_odd_t_kx, time, kx


    #######  Plot evolution of RH energy
    def plot_P_RH(self, fig=None, axs=None, time_min=0, time_max=1e10, species_idx="sum", passing_trapped="both", idxs_kx=None, kx_max=1e5, colors=None, fphi=1, fapar=1, fbpar=1, fcoll=1, D_hyper=None):

        if axs is None:
            fig, axs = plt.subplots(nrows=3,ncols=1, figsize=(9,16))

        # Evaluate from NL fluxes
        P_RH_phi_even_t_kx, P_RH_phi_odd_t_kx, time, kx_vals = \
                    self.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=0, fbpar=0, fcoll=0)
        P_RH_phi_t_kx  = P_RH_phi_even_t_kx + P_RH_phi_odd_t_kx
        P_RH_even_t_kx = np.copy(P_RH_phi_even_t_kx )
        P_RH_odd_t_kx  = np.copy(P_RH_phi_odd_t_kx  )

        if fapar != 0:
            P_RH_apar_even_t_kx, P_RH_apar_odd_t_kx, time, kx_vals = \
                    self.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx, fphi=0, fapar=fapar, fbpar=0, fcoll=0)
            P_RH_apar_t_kx  = P_RH_apar_even_t_kx + P_RH_apar_odd_t_kx
            P_RH_even_t_kx += P_RH_apar_even_t_kx
            P_RH_odd_t_kx  += P_RH_apar_odd_t_kx
        else:
            P_RH_apar_even_t_kx = np.zeros_like(P_RH_phi_t_kx) #None
            P_RH_apar_odd_t_kx  = np.zeros_like(P_RH_phi_t_kx) #None
            P_RH_apar_t_kx      = np.zeros_like(P_RH_phi_t_kx) #None

        if fbpar != 0:
            P_RH_bpar_even_t_kx, P_RH_bpar_odd_t_kx, time, kx_vals = \
                    self.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx, fphi=0, fapar=0, fbpar=fbpar, fcoll=0)
            P_RH_bpar_t_kx  = P_RH_bpar_even_t_kx + P_RH_bpar_odd_t_kx
            P_RH_even_t_kx += P_RH_bpar_even_t_kx
            P_RH_odd_t_kx  += P_RH_bpar_odd_t_kx
        else:
            P_RH_bpar_even_t_kx = np.zeros_like(P_RH_phi_t_kx) #None
            P_RH_bpar_odd_t_kx  = np.zeros_like(P_RH_phi_t_kx) #None
            P_RH_bpar_t_kx      = np.zeros_like(P_RH_phi_t_kx) #None

        if fcoll != 0:
            P_RH_coll_even_t_kx, P_RH_coll_odd_t_kx, time, kx_vals = \
                    self.get_P_RH(species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx, fphi=0, fapar=0, fbpar=0, fcoll=fcoll)
            P_RH_coll_t_kx  = P_RH_coll_even_t_kx + P_RH_coll_odd_t_kx
            P_RH_even_t_kx += P_RH_coll_even_t_kx
            P_RH_odd_t_kx  += P_RH_coll_odd_t_kx
        else:
            P_RH_coll_even_t_kx = np.zeros_like(P_RH_phi_t_kx) #None
            P_RH_coll_odd_t_kx  = np.zeros_like(P_RH_phi_t_kx) #None
            P_RH_coll_t_kx      = np.zeros_like(P_RH_phi_t_kx) #None


        # Evaluate numerically from time trace
        E_RH_t_kx, time, kx_vals = self.get_E_RH_t_kx(species_idx=species_idx, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)
        P_RH_t_kx_num = np.gradient(E_RH_t_kx, time, axis=0)
        #P_RH_t_kx_num = np.gradient(E_RH_t_kx, axis=0)/np.gradient(time)[:,None]

        # Evaluate hyperdissipation contribution if desired
        if D_hyper is not None:
            kperp2 = self.ncdata.variables['kperp2'][:][:,0,:,:]
            kmax = np.sqrt( kperp2.max() )
            P_RH_hyper_t_kx = -2*D_hyper * E_RH_t_kx * (kx_vals[None,:]/kmax)**4
        else:
            P_RH_hyper_t_kx = None

        for i_kx, kx in enumerate(kx_vals):

            if kx <= 0:
                continue

            if colors is None:
                #colors = sns.color_palette("coolwarm", len(kx_vals))
                c_num   = '0.5'
                c_tot   = 'k'
                c_phi   = 'mediumblue'
                c_apar  = 'crimson'
                c_bpar  = 'forestgreen'
                c_coll  = 'orange'
            else:
                c_num   = colors[i_kx] 
                c_tot   = colors[i_kx] 
                c_phi   = colors[i_kx] 
                c_apar  = colors[i_kx] 
                c_bpar  = colors[i_kx] 
                c_coll  = colors[i_kx] 

            axs[0].plot(time, P_RH_even_t_kx[:,i_kx], 
                        label=r"$k_x \rho = %.3f$" % (kx), c=c_tot, lw=2)
            axs[1].plot(time, P_RH_odd_t_kx[:,i_kx],  
                        label=r"$k_x \rho = %.3f$" % (kx), c=c_tot, lw=2)
            P_RH_tot_t_kx = P_RH_even_t_kx + P_RH_odd_t_kx
            ylabel = r"$P_\mathrm{RH}^+ + P_\mathrm{RH}^-$"
            if D_hyper is not None:
                P_RH_tot_t_kx += P_RH_hyper_t_kx
                ylabel += r"$+ P_\mathrm{RH}^\mathrm{hyper}$"

            axs[2].plot(time, P_RH_tot_t_kx[:,i_kx], c=c_tot, lw=2)
            axs[2].plot(time, P_RH_t_kx_num[:,i_kx], ls=(0, (3, 5, 1, 5, 1, 5)), lw=2,c=c_num)

            if fphi != 0:
                axs[0].plot(time, P_RH_phi_even_t_kx[:,i_kx], c=c_phi, ls='--')
                axs[1].plot(time, P_RH_phi_odd_t_kx[ :,i_kx], c=c_phi, ls='--')
                axs[2].plot(time, P_RH_phi_t_kx[     :,i_kx], c=c_phi, ls='--')

            if fapar != 0:
                axs[0].plot(time, P_RH_apar_even_t_kx[:,i_kx], c=c_apar, ls='-.')
                axs[1].plot(time, P_RH_apar_odd_t_kx[ :,i_kx], c=c_apar, ls='-.')
                axs[2].plot(time, P_RH_apar_t_kx[     :,i_kx], c=c_apar, ls='-.')

            if fbpar != 0:
                axs[0].plot(time, P_RH_bpar_even_t_kx[:,i_kx], c=c_bpar, ls=':')
                axs[1].plot(time, P_RH_bpar_odd_t_kx[ :,i_kx], c=c_bpar, ls=':')
                axs[2].plot(time, P_RH_bpar_t_kx[     :,i_kx], c=c_bpar, ls=':')

            if fcoll != 0:
                axs[0].plot(time, P_RH_coll_even_t_kx[:,i_kx], c=c_coll, ls=':')
                axs[1].plot(time, P_RH_coll_odd_t_kx[ :,i_kx], c=c_coll, ls=':')
                axs[2].plot(time, P_RH_coll_t_kx[     :,i_kx], c=c_coll, ls=':')

        axs[0].set_ylabel(r"$P_\mathrm{RH}^+$")
        axs[1].set_ylabel(r"$P_\mathrm{RH}^-$")
        axs[2].set_ylabel(ylabel)
        axs[2].set_xlabel(r"$t v_{T}/a$")

        for ax in axs:
            ax.grid(True)

        return fig, axs, P_RH_even_t_kx,      P_RH_odd_t_kx, \
                         P_RH_phi_even_t_kx,  P_RH_phi_odd_t_kx, \
                         P_RH_apar_even_t_kx, P_RH_apar_odd_t_kx,\
                         P_RH_bpar_even_t_kx, P_RH_bpar_odd_t_kx,\
                         P_RH_coll_even_t_kx, P_RH_coll_odd_t_kx,\
                         P_RH_hyper_t_kx, time, kx
        #return fig, axs, P_RH_even_t_kx, P_RH_odd_t_kx, P_RH_hyper_t_kx, time, kx

    #######  Get Rosenbluth-Hinton velocity-space integrand terms
    def get_RH_integrand_mu_vpa_zed_kx(self, species_idx=0):
        RH_integrand_even_mu_vpa_zed_kx_ri = self.ncdata.variables['RH_integrand_even'][:,:,species_idx,0,:,:,:]
        RH_integrand_even_mu_vpa_zed_kx = RH_integrand_even_mu_vpa_zed_kx_ri[:,:,:,:,0] + 1j*RH_integrand_even_mu_vpa_zed_kx_ri[:,:,:,:,1]

        RH_integrand_odd_mu_vpa_zed_kx_ri = self.ncdata.variables['RH_integrand_odd'][:,:,species_idx,0,:,:,:]
        RH_integrand_odd_mu_vpa_zed_kx = RH_integrand_odd_mu_vpa_zed_kx_ri[:,:,:,:,0] + 1j*RH_integrand_odd_mu_vpa_zed_kx_ri[:,:,:,:,1]

        kx, _, zed = self.get_kx_ky_zed()

        vpa    = self.ncdata.variables['vpa']
        mu     = self.ncdata.variables['mu']

        return RH_integrand_even_mu_vpa_zed_kx, RH_integrand_odd_mu_vpa_zed_kx, mu, vpa, zed, kx

    #######  Get strength of FLR effects
    def get_FLR(self, ky_idx=0, kx_idx=0):
        if self.code in ["stella", "GS2"]:
            kperp2 = self.ncdata.variables['kperp2'][:][:,0,kx_idx,ky_idx] 
            Gamma0 = specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2)
            gds2   = self.ncdata.variables['gds2'][:,0]  # |nabla(y)|^2
            gds21  = self.ncdata.variables['gds21'][:,0] # nabla(x)*nabla(y)
            gds22  = self.ncdata.variables['gds22'][:,0] # |nabla(x)|^2
            shat   = self.ncdata.variables['shat'].getValue()
            bmag   = self.ncdata.variables['bmag'][:,0]
            gds21  = gds21/shat
            gds22  = gds22/shat**2

        elif self.code == "GX":
            gds2   = self.ncdata['Geometry']['gds2'][:]  # |nabla(y)|^2
            gds21  = self.ncdata['Geometry']['gds21'][:] # nabla(x)*nabla(y)
            gds22  = self.ncdata['Geometry']['gds22'][:] # |nabla(x)|^2
            shat   = self.ncdata['Geometry']['shat'].getValue()
            bmag   = self.ncdata['Geometry']['bmag'][:]
            gds21  = gds21/shat
            gds22  = gds22/shat**2

            kx, ky, _ = self.get_kx_ky_zed()
            kperp2 = (kx[kx_idx]**2*gds22 + 2*kx[kx_idx]*ky[ky_idx]*gds21 + ky[ky_idx]**2*gds2)/bmag**2
            Gamma0 = specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2)

        return kperp2, Gamma0, gds2, gds21, gds22, bmag

    #######  Get ratio of diamagnetic to curvature drift frequencies along the tube
    def get_omega_s_k(self, ky_idx=0, kx_idx=0):

        ky     = self.ncdata.variables['ky'][ky_idx]
        theta0 = self.ncdata.variables['theta0'][kx_idx,ky_idx]
        shat   = self.ncdata.variables['shat']
        iota   = 1/self.ncdata.variables['q'].getValue()
        tprim  = self.ncdata.variables['tprim'][0]
        cvdrift  = np.squeeze(self.ncdata.variables['cvdrift'][:] ) # drift * grad(y)
        cvdrift0 = np.squeeze(self.ncdata.variables['cvdrift0'][:]) # drift * grad(x) * shat

        omega_sT = ky*tprim #= ky*rho * a/L_T -> norm = vT/a

        #geom_quantities = np.loadtxt(self.geo_file, skiprows=2).T
        #zeta = geom_quantities[1]
        #theta = zeta*iota
        #Kx = np.squeeze( ky*shat*(theta0 - theta) )
        #omega_k = ky*cvdrift + Kx*cvdrift0
        omega_k = ky*(cvdrift + theta0*cvdrift0)

        omega_s_k = omega_sT / omega_k

        return omega_s_k, omega_sT, omega_k

    #######  Get strength of FLR effects
    def get_Gamma0(self, ky_idx=0, kx_idx=0):
        kperp2 = self.ncdata.variables['kperp2'][:][:,0,kx_idx,ky_idx] 
        Gamma0 = specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2)
        return Gamma0

    ###########################################################################
    #######  Functions plotting data                                    #######
    ###########################################################################

    #######  Plot electrostatic potential over the flux tube
    def plot_phi_vs_zed(self, ax=None, label=None, ls=None, color=None, zed_times_nfield_periods=False, time_idx=-1, normalise_phi=True):

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        phi_vs_t, zed = self.read_phi_vs_zed(time_idx=time_idx, normalise_phi=normalise_phi)

        time_eval   = self.ncdata.variables['t'][time_idx]

        set_xlim = True
        if zed_times_nfield_periods:
            geom_quantities = np.loadtxt(self.geo_file, skiprows=2).T
            zed = geom_quantities[1]
            set_xlim = False

        plot_y_over_zed(ax, zed, phi_vs_t, ylabel=r"$\phi$", label=label, ls=ls, color=color, set_xlim=set_xlim)

        return ax, time_eval

    #######  Plot electrostatic potential over the flux tube def plot_phi2_vs_t_zed(self, tube=0, ax=None, label=None, zed_times_nfield_periods=False, remove_zonal=False):

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        phi2_vs_t_zed, time, zed = self.read_phi2_vs_t_zed(tube, remove_zonal=remove_zonal)

        set_xlim = True
        if zed_times_nfield_periods:
            geom_quantities = np.loadtxt(self.geo_file, skiprows=2).T
            zed = geom_quantities[1]
            set_xlim = False

        #X, Y = np.meshgrid(zed, time)
        X, Y = np.meshgrid(time, zed)
        Z = phi2_vs_t_zed.T

        eps_rel = 3e-2
        #im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='inferno', vmax=10)
        im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=max(Z.min(), eps_rel*Z.max()), vmax=Z.max()), shading='auto', cmap='inferno')

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\zeta$")
        ax.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

        return fig, ax, im


    #######  Plot flux tube geometrical factors
    def plot_flux_tube_geometry(self, fig=None, axs=None, label=None, plot_phi=True, zed_times_nfield_periods=False, load_from_nc=True, normalise_bmag=False, color=None, ls="-", xlim=None, norm_gradpar=False):

        if axs is None:
            fig, axs = plt.subplots(nrows=3,ncols=4, figsize=(20,12))
            plt.subplots_adjust(hspace=0,left=0.08,right=0.95,top=0.95,bottom=0.05,wspace=0.5)

        _, _, zed = self.get_kx_ky_zed()

        set_xlim = True
        if zed_times_nfield_periods:
            geom_quantities = np.loadtxt(self.geo_file, skiprows=2).T
            zed = geom_quantities[1]
            set_xlim = False

        if load_from_nc:
            if self.code == "stella":
                bmag     = self.ncdata.variables['bmag'][:]
                i = 0
                #np.savetxt("data_bmag.dat", bmag)
                #np.savetxt("data_zed.dat", zed)

                gradpar  = self.ncdata.variables['gradpar'][:]
                kperp2   = np.zeros_like(gradpar)#self.ncdata.variables['kperp2'][:][:,0,0,0] 
                #kperp2   = self.ncdata.variables['kperp2'][:][:,0,1,0] 
                jacob    = self.ncdata.variables['jacob'][:]
                gbdrift  = self.ncdata.variables['gbdrift'][:]  # drift * grad(y)
                gbdrift0 = self.ncdata.variables['gbdrift0'][:] # drift * grad(x) * shat
                cvdrift  = self.ncdata.variables['cvdrift'][:]  # drift * grad(y)
                cvdrift0 = self.ncdata.variables['cvdrift0'][:] # drift * grad(x) * shat
                gds2     = self.ncdata.variables['gds2'][:]
                gds21    = self.ncdata.variables['gds21'][:]
                gds22    = self.ncdata.variables['gds22'][:]
                grho     = self.ncdata.variables['grho'][:]

            elif self.code == "GX":
                i = 0
                bmag     = self.ncdata['Geometry']['bmag'][:]
                gradpar  = np.array(self.ncdata['Geometry']['gradpar']    )* np.ones_like(bmag)
                gbdrift  = 2*np.array(self.ncdata['Geometry']['gbdrift'][:] ) # drift * grad(y)
                gbdrift0 = 2*np.array(self.ncdata['Geometry']['gbdrift0'][:]) # drift * grad(x) * shat
                cvdrift  = 2*np.array(self.ncdata['Geometry']['cvdrift'][:] ) # drift * grad(y)
                cvdrift0 = 2*np.array(self.ncdata['Geometry']['cvdrift0'][:]) # drift * grad(x) * shat
                gds2     = self.ncdata['Geometry']['gds2'][:]
                gds21    = self.ncdata['Geometry']['gds21'][:]
                gds22    = self.ncdata['Geometry']['gds22'][:]
                grho     = self.ncdata['Geometry']['grho'][:]
                kperp2   = np.zeros_like(gradpar)#self.ncdata.variables['kperp2'][:][:,0,0,0] 
                jacob    = np.zeros_like(gradpar)#self.ncdata.variables['kperp2'][:][:,0,0,0] 

        else:
            geom_quantities = np.loadtxt(self.geo_file, skiprows=2).T

            # alpha zeta bmag gradpar grad_alpha2 gd_alph_psi grad_psi2 gds23 gds24 gbdriftalph gbdrift0psi cvdriftalph cvdrift0psi theta_vmec B_sub_theta B_sub_zeta
            # 0     1    2    3       4           5           6         7     8     9           10          11          12          13         14          15
#            bmag     = geom_quantities[2]
#            gradpar  = geom_quantities[3]
#            kperp2   = self.ncdata.variables['kperp2'][:][:,0,0,0] 
#            jacob    = self.ncdata.variables['jacob'][:]
#            gbdrift  = geom_quantities[9]
#            gbdrift0 = geom_quantities[10]
#            cvdrift  = geom_quantities[11]
#            cvdrift0 = geom_quantities[12]
#            gds2     = geom_quantities[4]
#            gds21    = geom_quantities[5]
#            gds22    = geom_quantities[6]
#            grho     = self.ncdata.variables['grho'][:]


            # alpha zed zeta bmag bdot_grad_z gds2 gds21 gds22 gds23 gds24 gbdrift cvdrift gbdrift0 bmag_psi0 btor
            # 0     1   2    3    4           5    6     7     8     9     10      11      12       13
            bmag     = geom_quantities[3]
            gradpar  = geom_quantities[4]
            kperp2   = self.ncdata.variables['kperp2'][:][:,0,0,0] 
            jacob    = self.ncdata.variables['jacob'][:]
            gbdrift  = geom_quantities[10]
            gbdrift0 = geom_quantities[12]
            cvdrift  = geom_quantities[11]
            cvdrift0 = geom_quantities[12]
            gds2     = geom_quantities[5]
            gds21    = geom_quantities[6]
            gds22    = geom_quantities[7]
            grho     = self.ncdata.variables['grho'][:]

#            # alpha zeta bmag gradpar bdot_grad_z grad_alpha2 gd_alph_psi grad_psi2 gds23 gds24 gbdriftalph gbdrift0psi cvdriftalph cvdrift0psi theta_vmec B_sub_theta B_sub_zeta
#            # 0     1    2    3       4           5           6           7         8     9     10          11          12          13          14         15          16
#            bmag     = geom_quantities[2]
#            gradpar  = geom_quantities[3]
#            kperp2   = self.ncdata.variables['kperp2'][:][:,0,0,0] 
#            jacob    = self.ncdata.variables['jacob'][:]
#            gbdrift  = geom_quantities[10]
#            gbdrift0 = geom_quantities[11]
#            cvdrift  = geom_quantities[12]
#            cvdrift0 = geom_quantities[13]
#            gds2     = geom_quantities[5]
#            gds21    = geom_quantities[6]
#            gds22    = geom_quantities[7]
#            grho     = self.ncdata.variables['grho'][:]
#

        if self.code == "stella":
            shat   = self.ncdata.variables['shat'].getValue()
        else:
            shat   = self.ncdata['Geometry']['shat'].getValue()

        if not plot_phi:
            gradpar_or_phi       = gradpar
            label_gradpar_or_phi = r"$\nabla_\parallel \zeta$"
        else:
            gradpar_or_phi, _   = self.read_phi_vs_zed()
            label_gradpar_or_phi = r"$\phi$"

        if norm_gradpar:
            norm = gradpar
        else:
            norm = 1

        # Normalise Bmag to go from 0.5 to 1.5
        if normalise_bmag:
            bmag = 0.5 * ( 2 + (bmag - (min(bmag)+max(bmag))/2 ) / ((max(bmag)-min(bmag))/2) )

        # Plot
        cvdrift = cvdrift/2
        plot_y_over_zed(axs[0,0], zed, cvdrift, ylabel=r"$B^{-2} \mathbf{B}\times\mathbf{\kappa}\cdot\nabla y$", no_xticks=True, set_xlim=set_xlim, label=label, color=color, ls=ls, xlim=xlim)
        cvdrift0 = cvdrift0 / (2*shat)
        plot_y_over_zed(axs[1,0], zed, cvdrift0, ylabel=r"$B^{-2} \mathbf{B}\times\mathbf{\kappa}\cdot\nabla x$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
        plot_y_over_zed(axs[2,0], zed, gradpar_or_phi, ylabel=label_gradpar_or_phi, no_xticks=False, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)

        plot_y_over_zed(axs[0,1], zed, bmag, ylabel=r"$B$", no_xticks=True, label=label, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
        gbdrift = gbdrift/2
        plot_y_over_zed(axs[1,1], zed, gbdrift/norm, ylabel=r"$v_{My}$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
        #plot_y_over_zed(axs[1,1], zed, gbdrift, ylabel=r"$B^{-3} \mathbf{B}\times\nabla B\cdot\nabla y$", no_xticks=True, set_xlim=set_xlim, color=color, xlim=xlim)
        gbdrift0 = gbdrift0 / (2*shat)
        plot_y_over_zed(axs[2,1], zed, gbdrift0/norm, ylabel=r"$v_{Mx}$", no_xticks=False, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
        #plot_y_over_zed(axs[2,1], zed, gbdrift0, ylabel=r"$B^{-3} \mathbf{B}\times\nabla B\cdot\nabla x$", no_xticks=False, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)


        plot_y_over_zed(axs[0,2], zed, gds2, ylabel=r"$|\nabla y|^2$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
        axs[0,2].set_ylim(ymin=0)
        gds21 = gds21/shat
        plot_y_over_zed(axs[1,2], zed, gds21, ylabel=r"$\nabla y \cdot \nabla x$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
        gds22 = gds22/shat**2
        plot_y_over_zed(axs[2,2], zed, gds22, ylabel=r"$|\nabla x|^2$", no_xticks=False, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
        axs[2,2].set_ylim(ymin=0)

        plot_y_over_zed(axs[0,3], zed, kperp2, ylabel=r"$(\rho_i k_\perp)^2$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
        plot_y_over_zed(axs[1,3], zed, jacob, ylabel=r"$\sqrt{g}$", no_xticks=True, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)
        plot_y_over_zed(axs[2,3], zed, grho, ylabel=r"$|\nabla \rho|$", no_xticks=False, set_xlim=set_xlim, color=color, ls=ls, xlim=xlim)

        if label is not None:
            axs[0,0].legend()

        # Evaluate flux-surface avg of FLR related quantities
        dl_over_B_avg = self.dl_over_B_avg()
        nablax2_avg = np.sum(dl_over_B_avg * gds22[:])
        nablay2_avg = np.sum(dl_over_B_avg * gds2[:])
        nablaxy_avg = np.sum(dl_over_B_avg * gds21[:])
        kperp2_avg  = np.sum(dl_over_B_avg * kperp2)

        print("\n"+self.filename_base+":")
        print("shat = %e" % (shat))
        print("Bmin, Bmax = %.2e, %.2e" % (bmag.min(), bmag.max()))
        print("Max(vMx) = %.2e" % (gbdrift0.max()))
        print("Avg of |nabla x|^2 = %e" % (nablax2_avg))
        print("Avg of |nabla y|^2 = %e" % (nablay2_avg))
        print("Avg of nabla x * nabla y = %e" % (nablaxy_avg))
        print("Avg of |kperp|^2  = %e" % (kperp2_avg))
        print("gradpar(theta(0)) = %.2e" % (gradpar[0]))
        print("(kperp*rho)^2 in [%e, %e]" % (kperp2.min(), kperp2.max()) )
        print(" min, max of |grad-y| = %e, %e" % (np.sqrt(gds2).min(), np.sqrt(gds2).max()))
        print(" min, max of |grad-x| = %e, %e" % (np.sqrt(gds22).min()/shat, np.sqrt(gds22).max()/shat))

        return fig, axs

    #######  Get mean zonal shearing rate (kx)
    def get_zonal_shearing_kx(self, time_min=0, time_max=1e5):

        time_idx_min = self.get_time_idx(time_min)
        time_idx_max = self.get_time_idx(time_max)
        time_all = self.get_time_array()
        time = time_all[time_idx_min:time_idx_max-1]
        dt = time_all[time_idx_min+1:time_idx_max]-time_all[time_idx_min:time_idx_max-1]
        kx, _, _ = self.get_kx_ky_zed()
        Gamma0 = specialfunc.iv(0, kx**2/2) * np.exp(-kx**2/2)

        if self.code == "stella":
            # phi_vs_t(t, tube, zed, theta0, ky, ri)
            #phiZ_t_kx_ri = self.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max-1,0,0,:,0,:]
            phiZ_t_kx = np.sqrt(self.ncdata.variables['phi2_vs_kxky'][time_idx_min:time_idx_max-1,:,0])
            phiZ_t_kx_ri = np.zeros( (np.shape(phiZ_t_kx)[0], np.shape(phiZ_t_kx)[1], 2) )
            phiZ_t_kx_ri[:,:,0] = phiZ_t_kx            

        elif self.code == "GX":
            if self.GX_old_version:
                # phi_vs_t(t, tube, zed, theta0, ky, ri)
                #phiZ_t_kx_ri = self.ncdata['Special']['Phi_z'][0,:,0,:]
                #phiZ_t_kx_ri = phiZ_t_kx_ri[None,:,:]

                phiZ_t_kx = np.sqrt(self.ncdata['Spectra']['Akxkyst'][time_idx_min:time_idx_max-1,0,:]/(1-Gamma0)[None,:])

            else:
                phiZ_t_kx = np.sqrt( self.ncdata['Diagnostics']['Wphi_kxkyst'][time_idx_min:time_idx_max-1,0,0,:] / (1-Gamma0)[None,:] )

            phiZ_t_kx_ri = np.zeros( (np.shape(phiZ_t_kx)[0], np.shape(phiZ_t_kx)[1], 2) )
            phiZ_t_kx_ri[:,:,0] = phiZ_t_kx*np.sqrt(2) #/(2*np.pi) # undo theta avg

        dx2phiZ_t_kx = -(1-Gamma0) * (phiZ_t_kx_ri[:,:,0]+1j*phiZ_t_kx_ri[:,:,1])

        dx2phiZ_stationary_kx_C = np.sum(dx2phiZ_t_kx*dt[:,None], axis=0)/np.sum(dt)

        gammaE2_stationary_kx = np.abs(dx2phiZ_stationary_kx_C)**2

        gammaE2_timevar_kx = np.sum(np.abs(dx2phiZ_t_kx-dx2phiZ_stationary_kx_C[None,:])**2 *dt[:,None], axis=0)/np.sum(dt)

        gammaE2_tot_kx = np.sum(np.abs(dx2phiZ_t_kx)**2 *dt[:,None], axis=0)/np.sum(dt)

        return gammaE2_tot_kx[kx>0], gammaE2_stationary_kx[kx>0], gammaE2_timevar_kx[kx>0], kx[kx>0]

    #######  Get total fluxes over time
    # Units:
    def get_fluxes_over_time(self, species_idx=0, norm=True, configuration=None, delta_t=None, load_from_nc=False):

        time = self.get_time_array()
        if self.code == "stella":
            if load_from_nc:
                pflx = self.ncdata['pflux_vs_s'][:,species_idx]
                vflx = self.ncdata['vflux_vs_s'][:,species_idx]
                qflx = self.ncdata['qflux_vs_s'][:,species_idx]

            else:
                fluxes   = np.loadtxt(self.fluxes_file)
                try:
                    nspecies = len(self.ncdata.dimensions['species'])
                except:
                    nspecies = 1
    
                print(nspecies)
                # fluxes is in format [ #time pflx*ns vflx*ns qflx*ns ]
                time = fluxes[:,0]
                pflx = fluxes[:,1           +species_idx]
                vflx = fluxes[:,1+  nspecies+species_idx]
                qflx = fluxes[:,1+2*nspecies+species_idx]
        #np.nan_to_num(vflx)

        elif self.code == "GS2":
            pflx = self.ncdata['es_part_flux'][:,species_idx]
            vflx = self.ncdata['es_mom_flux'][:,species_idx]
            qflx = self.ncdata['es_heat_flux'][:,species_idx]
            norm = False

        elif self.code == "GX":
            if delta_t is not None:
                time_idx_min = np.argmin(np.abs(time-(time[-1]-delta_t)))
            else:
                time_idx_min = 0

            time = time[time_idx_min:]# * 2**(1/2)

            pflx = 0 #self.ncdata['Fluxes']['pflux'][:,species_idx]
            vflx = 0
            if self.GX_old_version:
                qflx = self.ncdata['Fluxes']['qflux'][time_idx_min:,species_idx] / (2**(3/2))
            else:
                qflx = self.ncdata['Diagnostics']['HeatFlux_st'][time_idx_min:,species_idx] / (2**(3/2))
            norm = False

        if norm:
            flux_norm = self.flux_norm()
            if configuration is not None:
                flux_norm = flux_norm / get_true_flux_norm(configuration)

            pflx = pflx/flux_norm
            vflx = vflx/flux_norm
            qflx = qflx/flux_norm

        return pflx, vflx, qflx, time

    #######  Get contributions to energy evolution equation
    def get_dt_par_mom_pressure_transport(self, time_min=0, time_max=1e10, time_idx_skip=1, nx=None, ny=None, kxmin_filter=np.infty, kymin_filter=np.infty, kxmax_filter=-1, kymax_filter=-1):

        time = self.get_time_array()
        time_max = min(time[-1], time_max)
        if time_min < 0:
            time_min = time_max - np.abs(time_min)
        time_idx_min = np.argmin(np.abs(time-time_min))
        time_idx_max = np.argmin(np.abs(time-time_max))
        time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)

        dE_par_mom_tr  = np.zeros(len(time_idx_eval))
        dE_meanP_tr    = np.zeros(len(time_idx_eval))
        dE_deltP_tr    = np.zeros(len(time_idx_eval))

        for i_time_idx, time_idx in enumerate(time_idx_eval):
            print("Evaluating par mom transport: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")

            x, dE_par_mom_tr_x, dE_meanP_tr_x, dE_deltP_tr_x = self.get_dt_par_mom_pressure_transport_x(time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
            dx = x[1]-x[0]
            dE_par_mom_tr[i_time_idx] = np.sum(dE_par_mom_tr_x)*dx
            dE_meanP_tr[i_time_idx]   = np.sum(dE_meanP_tr_x)*dx
            dE_deltP_tr[i_time_idx]   = np.sum(dE_deltP_tr_x)*dx

        return time[time_idx_eval], dE_par_mom_tr, dE_meanP_tr, dE_deltP_tr


    #######  Get contributions to energy evolution equation
    def get_dt_par_mom_pressure_transport_x(self, time_idx=-1, nx=None, ny=None, kxmin_filter=np.infty, kymin_filter=np.infty, kxmax_filter=-1, kymax_filter=-1):

        uparZ_zed_x_y, zed, x, y, _  = self.get_quantity_zed_x_y("upar",           time_idx=time_idx, kx_order=0, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True, nx=nx, ny=ny)
        par_mom_transp_zed_x_y, zed, x, y, _ = self.get_quantity_zed_x_y("par_mom_transport", time_idx=time_idx, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, nx=nx, ny=ny)

        presZ_zed_x_y, zed, x, y, _  = self.get_quantity_zed_x_y("pressure",       time_idx=time_idx, kx_order=0, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True, nx=nx, ny=ny)
        densZ_zed_x_y, zed, x, y, _  = self.get_quantity_zed_x_y("density",        time_idx=time_idx, kx_order=0, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True, nx=nx, ny=ny)
        tempZ_zed_x_y = presZ_zed_x_y-densZ_zed_x_y
        pressure_transp_zed_x_y, zed, x, y, _ = self.get_quantity_zed_x_y("pressure_transport", time_idx=time_idx, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, nx=nx, ny=ny)

        dl_over_B_avg = self.dl_over_B_avg()
        mean_tempZ_x_y = np.sum(dl_over_B_avg[:,None,None]*tempZ_zed_x_y)
        mean_tempZ_zed_x_y = np.zeros_like(tempZ_zed_x_y)
        for i_zed in range(len(zed)):
            mean_tempZ_zed_x_y[i_zed] = mean_tempZ_x_y
        delt_tempZ_zed_x_y = tempZ_zed_x_y - mean_tempZ_zed_x_y

        dy = y[1]-y[0]
        dE_par_mom_tr_x  = np.sum(par_mom_transp_zed_x_y*uparZ_zed_x_y*dl_over_B_avg[:,None,None], axis=(0,2))*dy  * 2
        dE_mean_pressure_tr_x = np.sum(pressure_transp_zed_x_y*mean_tempZ_zed_x_y*dl_over_B_avg[:,None,None], axis=(0,2))*dy * 4/3#4/7 *2
        dE_delt_pressure_tr_x = np.sum(pressure_transp_zed_x_y*delt_tempZ_zed_x_y*dl_over_B_avg[:,None,None], axis=(0,2))*dy * 4/3#4/7 *2
        return x, dE_par_mom_tr_x, dE_mean_pressure_tr_x, dE_delt_pressure_tr_x

    #######  Get constributions to zonal flow energy evolution equation
    def get_dt_zonal_energy_contributions(self, time_min=0, time_max=1e10, time_idx_skip=1, nx=None, ny=None, kxmin_filter=np.infty, kymin_filter=np.infty, kxmax_filter=-1, kymax_filter=-1, separate_Reynolds=True):

        time = self.get_time_array()
        time_max = min(time[-1], time_max)
        if time_min < 0:
            time_min = time_max - np.abs(time_min)
        time_idx_min = np.argmin(np.abs(time-time_min))
        time_idx_max = np.argmin(np.abs(time-time_max))
        time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)

        EZ_t                        = np.zeros(len(time_idx_eval))
        EZ_t_deltaphi2              = np.zeros(len(time_idx_eval))
        dEZ_reynolds_phi_nablax2_t  = np.zeros(len(time_idx_eval))
        dEZ_reynolds_Pprp_nablax2_t = np.zeros(len(time_idx_eval))
        dEZ_reynolds_phi_nablaxy_t  = np.zeros(len(time_idx_eval))
        dEZ_reynolds_Pprp_nablaxy_t = np.zeros(len(time_idx_eval))
        dEZ_vDx_t                   = np.zeros(len(time_idx_eval))
        dEZ_upar_t                  = np.zeros(len(time_idx_eval))

        for i_time_idx, time_idx in enumerate(time_idx_eval):
            print("Evaluating zonal energy contributions: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")

            # Get energies
            x, EZ_x, EZ_deltaphi2_x, dEZ_reynolds_phi_nablax2_x, dEZ_reynolds_Pprp_nablax2_x, dEZ_reynolds_phi_nablaxy_x, dEZ_reynolds_Pprp_nablaxy_x, dEZ_vDx_x, dEZ_upar_x = self.get_dt_zonal_energy_contributions_x(time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
            dx = x[1]-x[0]

            EZ_t[i_time_idx]                        = np.sum(EZ_x               )*dx
            EZ_t_deltaphi2[i_time_idx]              = np.sum(EZ_deltaphi2_x     )*dx
            dEZ_reynolds_phi_nablax2_t[i_time_idx]  = np.sum(dEZ_reynolds_phi_nablax2_x )*dx
            dEZ_reynolds_Pprp_nablax2_t[i_time_idx] = np.sum(dEZ_reynolds_Pprp_nablax2_x)*dx
            dEZ_reynolds_phi_nablaxy_t[i_time_idx]  = np.sum(dEZ_reynolds_phi_nablaxy_x )*dx
            dEZ_reynolds_Pprp_nablaxy_t[i_time_idx] = np.sum(dEZ_reynolds_Pprp_nablaxy_x)*dx
            dEZ_vDx_t[   i_time_idx]                = np.sum(dEZ_vDx_x          )*dx
            dEZ_upar_t[    i_time_idx]              = np.sum(dEZ_upar_x         )*dx

        return time[time_idx_eval], EZ_t, dEZ_reynolds_phi_nablax2_t, dEZ_reynolds_Pprp_nablax2_t, dEZ_reynolds_phi_nablaxy_t, dEZ_reynolds_Pprp_nablaxy_t, dEZ_vDx_t, dEZ_upar_t, EZ_t_deltaphi2

    #######  Get Reynolds stress spectrum in kx', ky' of the nonzonal phi and Pperp
    def get_Reynolds_NZ_spectrum(self, time_min=0, time_max=99999, time_idx_skip=1):

        time = self.get_time_array()
        time_max = min(time[-1], time_max)
        if time_min < 0:
            time_min = time_max - np.abs(time_min)
        time_idx_min = np.argmin(np.abs(time-time_min))
        time_idx_max = np.argmin(np.abs(time-time_max))
        time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)
        dt = np.gradient(time)

        kx = self.ncdata['kx'][:]
        ky = self.ncdata['ky'][:]

        dEZ_dt_reynolds_phi_kx_ky  = np.zeros((len(kx),len(ky)), dtype='complex')
        dEZ_dt_reynolds_Pprp_kx_ky = np.zeros((len(kx),len(ky)), dtype='complex')

        # Geometric coefficients
        dl_over_B_avg = self.dl_over_B_avg()
        _, _, _, gds21, gds22, bmag = self.get_FLR()

        # Time-average
        for i_time_idx, time_idx in enumerate(time_idx_eval):
            print("Evaluating reynolds contributions: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")
        
            # Load phi and Pprp
            phi_zed_kx_ky,  zed, kx, ky, _  = self.get_quantity_zed_kx_ky("phi",           time_idx=time_idx)
            Pprp_zed_kx_ky,   _,  _,  _, _  = self.get_quantity_zed_kx_ky("pressure_perp", time_idx=time_idx)
            phiZ_zed_kx = phi_zed_kx_ky[:,:,0]
    
            nablaxnablaphi_zed_kx_ky = 1j* (ky[None,None,:]*phi_zed_kx_ky*(gds21/bmag**2)[:,None,None] + kx[None,:,None]*phi_zed_kx_ky*(gds22/bmag**2)[:,None,None])
    
            # Inner loop over kx of phi_Z that contribute to the energy exchange
            for i_kx in range(len(kx)):
                mult_fac_zed_kx_ky = 0.5*dl_over_B_avg[:,None,None] * ((1j*kx[i_kx])**2 * np.conj(phiZ_zed_kx[:,i_kx]))[:,None,None] * nablaxnablaphi_zed_kx_ky
    
                delta_kx_vals = (np.arange(len(kx)) - i_kx)%(len(kx))
                dEZ_dt_reynolds_phi_kx_ky  += np.sum(mult_fac_zed_kx_ky * np.conj( 1j*ky[None,None,:] *phi_zed_kx_ky[:,delta_kx_vals]) , axis=0) * dt[time_idx]
                dEZ_dt_reynolds_Pprp_kx_ky += np.sum(mult_fac_zed_kx_ky * np.conj( 1j*ky[None,None,:]*Pprp_zed_kx_ky[:,delta_kx_vals]) , axis=0) * dt[time_idx]

        # Correct time-normalisation to get average
        dEZ_dt_reynolds_phi_kx_ky  = dEZ_dt_reynolds_phi_kx_ky  / np.sum(dt)
        dEZ_dt_reynolds_Pprp_kx_ky = dEZ_dt_reynolds_Pprp_kx_ky / np.sum(dt)
        
        return kx, ky, dEZ_dt_reynolds_phi_kx_ky, dEZ_dt_reynolds_Pprp_kx_ky


    #######  Get Reynolds stress spectrum in kz, kx of the nonzonal phi and Pperp
    def get_Reynolds_kz_kxNZ_spectrum(self, time_min=0, time_max=99999, time_idx_skip=1):

        time = self.get_time_array()
        time_max = min(time[-1], time_max)
        if time_min < 0:
            time_min = time_max - np.abs(time_min)
        time_idx_min = np.argmin(np.abs(time-time_min))
        time_idx_max = np.argmin(np.abs(time-time_max))
        time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)
        dt = np.gradient(time)

        kx = self.ncdata['kx'][:]
        ky = self.ncdata['ky'][:]

        dEZ_dt_reynolds_phi_kz_kx  = np.zeros((len(kx),len(kx)), dtype='complex')
        dEZ_dt_reynolds_Pprp_kz_kx = np.zeros((len(kx),len(kx)), dtype='complex')

        # Geometric coefficients
        dl_over_B_avg = self.dl_over_B_avg()
        _, _, _, gds21, gds22, bmag = self.get_FLR()

        # Time-average
        for i_time_idx, time_idx in enumerate(time_idx_eval):
            print("Evaluating reynolds contributions: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")
        
            # Load phi and Pprp
            phi_zed_kx_ky,  zed, kx, ky, _  = self.get_quantity_zed_kx_ky("phi",           time_idx=time_idx)
            Pprp_zed_kx_ky,   _,  _,  _, _  = self.get_quantity_zed_kx_ky("pressure_perp", time_idx=time_idx)
            phiZ_zed_kz = phi_zed_kx_ky[:,:,0]
    
            nablaxnablaphi_zed_kx_ky = 1j* (ky[None,None,:]*phi_zed_kx_ky*(gds21/bmag**2)[:,None,None] + kx[None,:,None]*phi_zed_kx_ky*(gds22/bmag**2)[:,None,None])
    
            # Sum over zed and ky of nonzonal terms that contribute to the energy exchange
            for i_kz in range(len(kx)):
                for i_kx in range(len(kx)):
                    delta_i_kx = (i_kz - i_kx)%(len(kx))
                    dEZ_dt_reynolds_phi_kz_kx[ i_kz,i_kx] = np.sum(0.5*dl_over_B_avg[:,None]*(1j*kx[i_kz])**2*phiZ_zed_kz[:,i_kz,None]* nablaxnablaphi_zed_kx_ky[:,i_kx,:]*np.conj(1j*ky[None,None,:]* phi_zed_kx_ky[:,delta_i_kx,:])) * dt[time_idx]
                    dEZ_dt_reynolds_Pprp_kz_kx[i_kz,i_kx] = np.sum(0.5*dl_over_B_avg[:,None]*(1j*kx[i_kz])**2*phiZ_zed_kz[:,i_kz,None]* nablaxnablaphi_zed_kx_ky[:,i_kx,:]*np.conj(1j*ky[None,None,:]*Pprp_zed_kx_ky[:,delta_i_kx,:])) * dt[time_idx]

        # Correct time-normalisation to get average
        dEZ_dt_reynolds_phi_kz_kx  = dEZ_dt_reynolds_phi_kz_kx  / np.sum(dt)
        dEZ_dt_reynolds_Pprp_kz_kx = dEZ_dt_reynolds_Pprp_kz_kx / np.sum(dt)
        
        return kx, dEZ_dt_reynolds_phi_kz_kx, dEZ_dt_reynolds_Pprp_kz_kx


    #######  Get constributions to zonal flow energy evolution equation
    def get_time_avg_zonal_energy_contributions_kx(self, time_min=0, time_max=1e10, time_idx_skip=1, alt_slow_eval=False, omega_min=None, omega_max=None):

        time = self.get_time_array()
        time_max = min(time[-1], time_max)
        if time_min < 0:
            time_min = time_max - np.abs(time_min)
        time_idx_min = np.argmin(np.abs(time-time_min))
        time_idx_max = np.argmin(np.abs(time-time_max))
        time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)
        time_eval = time[time_idx_eval]

        # Geometric quantities we will need
        dl_over_B_avg = self.dl_over_B_avg()
        shat     = self.ncdata.variables['shat'].getValue()
        vdriftx = self.ncdata.variables['gbdrift0'][:,0]/(2*shat)
        _, _, _, _, gds22, bmag = self.get_FLR()
        nablax2 = gds22/bmag**2
        _, _, zed = self.get_kx_ky_zed()
        dl_costheta = self.get_zed_weight("cos", zed)

        for i_time_idx, time_idx in enumerate(time_idx_eval):
            print("Evaluating zonal energy contributions: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")

            # Get energy contributions
            if not alt_slow_eval:
                #if omega_min is not None or omega_max is not None:
                #    print("No implementation yet of omega filter for alt_slow_eval=False. Returning.")
                #    return

                # Evaluate everything directly in k-space
                phi_zed_kx_ky, zed, kx, ky, _ = self.get_quantity_zed_kx_ky(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx)
                dxphi_zed_kx_ky, zed, kx, ky, _ = self.get_quantity_zed_kx_ky(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1)
                reynolds_phi_nablax2_zed_kx_ky, _, _, _, _  = self.get_quantity_zed_kx_ky("Reynolds_phi_nablax2",  time_idx=time_idx, kx_order=2)
                reynolds_Pprp_nablax2_zed_kx_ky, _, _, _, _ = self.get_quantity_zed_kx_ky("Reynolds_Pprp_nablax2", time_idx=time_idx, kx_order=2)
                reynolds_phi_nablaxy_zed_kx_ky, _, _, _, _  = self.get_quantity_zed_kx_ky("Reynolds_phi_nablaxy",  time_idx=time_idx, kx_order=2)
                reynolds_Pprp_nablaxy_zed_kx_ky, _, _, _, _ = self.get_quantity_zed_kx_ky("Reynolds_Pprp_nablaxy", time_idx=time_idx, kx_order=2)
                upar_zed_kx_ky, _, _, _, _ = self.get_quantity_zed_kx_ky(quantity="upar", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=0)
                dxP_zed_kx_ky, _, _, _, _ = self.get_quantity_zed_kx_ky(quantity="pressure", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1)
                pressure_tr_zed_kx_ky, _, _, _, _ = self.get_quantity_zed_kx_ky(quantity="pressure_transport", time_idx=time_idx, kx_order=1)
                par_mom_tr_zed_kx_ky, _, _, _, _ = self.get_quantity_zed_kx_ky(quantity="par_mom_transport", time_idx=time_idx, kx_order=1)
                dxupar_zed_kx_ky, _, _, _, _ = self.get_quantity_zed_kx_ky(quantity="upar", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1)

                reynolds_phi_nablax2_zed_kx  = reynolds_phi_nablax2_zed_kx_ky[:,:,0]
                reynolds_Pprp_nablax2_zed_kx = reynolds_Pprp_nablax2_zed_kx_ky[:,:,0]
                reynolds_phi_nablaxy_zed_kx  = reynolds_phi_nablaxy_zed_kx_ky[:,:,0]
                reynolds_Pprp_nablaxy_zed_kx = reynolds_Pprp_nablaxy_zed_kx_ky[:,:,0]
                phi_zed_kx                   = phi_zed_kx_ky[ :,kx>=0,0]
                dxphi_zed_kx                 = dxphi_zed_kx_ky[ :,kx>=0,0]
                upar_zed_kx                  = upar_zed_kx_ky[:,kx>=0,0]
                dxP_zed_kx                   = dxP_zed_kx_ky[ :,kx>=0,0]
                pressure_tr_zed_kx           = pressure_tr_zed_kx_ky[ :,:,0]
                par_mom_tr_zed_kx            = par_mom_tr_zed_kx_ky[ :,:,0]
                dxupar_zed_kx                = dxupar_zed_kx_ky[:,kx>=0,0]
                kx = kx[kx>=0]

                # Obtain parallel derivative of upar term
                dupar_zed_kx = np.zeros_like(upar_zed_kx)
                uparB_zed_kx = upar_zed_kx / bmag[:,None]
                gradpar  = self.ncdata.variables['gradpar'][:]
                dzed = zed[1]-zed[0]
                for i_zed in range(len(zed)-1):
                    if i_zed == 0:
                        dupar_zed_kx[0] = (uparB_zed_kx[1]-uparB_zed_kx[-1]) / dzed
                    else:
                        dupar_zed_kx[i_zed] = 0.5*(uparB_zed_kx[i_zed+1]-uparB_zed_kx[i_zed-1]) / dzed
                dupar_zed_kx[-1] = (uparB_zed_kx[0]-uparB_zed_kx[-2]) / dzed
                dupar_zed_kx = dupar_zed_kx * (gradpar*bmag)[:,None]

            else:
                # Evaluate everything in real space and then transform back to k-space
                reynolds_phi_nablax2_zed_x_y,  zed, x, y, _ = self.get_quantity_zed_x_y("Reynolds_phi_nablax2",  time_idx=time_idx, kx_order=2)
                reynolds_Pprp_nablax2_zed_x_y, zed, x, y, _ = self.get_quantity_zed_x_y("Reynolds_Pprp_nablax2", time_idx=time_idx, kx_order=2)
                reynolds_phi_nablaxy_zed_x_y,  zed, x, y, _ = self.get_quantity_zed_x_y("Reynolds_phi_nablaxy",  time_idx=time_idx, kx_order=2)
                reynolds_Pprp_nablaxy_zed_x_y, zed, x, y, _ = self.get_quantity_zed_x_y("Reynolds_Pprp_nablaxy", time_idx=time_idx, kx_order=2)
                dxP_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="pressure", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1)
                phi_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx)
                dxphi_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1)
                upar_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="upar", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=0) 
                dxupar_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="upar", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1) 
                pressure_tr_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="pressure_transport", time_idx=time_idx)
                par_mom_tr_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="par_mom_transport", time_idx=time_idx)
        
                # Obtain parallel derivative of upar term
                dupar_zed_x_y = np.zeros_like(upar_zed_x_y)
                uparB_zed_x_y = upar_zed_x_y / bmag[:,None,None]
                gradpar  = self.ncdata.variables['gradpar'][:]
                dzed = zed[1]-zed[0]
                for i_zed in range(len(zed)-1):
                    if i_zed == 0:
                        dupar_zed_x_y[0] = (uparB_zed_x_y[1]-uparB_zed_x_y[-1]) / dzed
                    else:
                        dupar_zed_x_y[i_zed] = 0.5*(uparB_zed_x_y[i_zed+1]-uparB_zed_x_y[i_zed-1]) / dzed
                dupar_zed_x_y[-1] = (uparB_zed_x_y[0]-uparB_zed_x_y[-2]) / dzed
                dupar_zed_x_y = dupar_zed_x_y * (gradpar*bmag)[:,None,None]
        
                # Take y-averages
                dy = y[1]-y[0]
                reynolds_phi_nablax2_zed_x  = np.sum(reynolds_phi_nablax2_zed_x_y,  axis=2)*dy
                reynolds_Pprp_nablax2_zed_x = np.sum(reynolds_Pprp_nablax2_zed_x_y, axis=2)*dy
                reynolds_phi_nablaxy_zed_x  = np.sum(reynolds_phi_nablaxy_zed_x_y,  axis=2)*dy
                reynolds_Pprp_nablaxy_zed_x = np.sum(reynolds_Pprp_nablaxy_zed_x_y, axis=2)*dy
                dxP_zed_x                   = np.sum(dxP_zed_x_y,                   axis=2)*dy
                phi_zed_x                   = np.sum(phi_zed_x_y,                   axis=2)*dy
                dxphi_zed_x                 = np.sum(dxphi_zed_x_y,                 axis=2)*dy
                dupar_zed_x                 = np.sum(dupar_zed_x_y,                 axis=2)*dy
                dxupar_zed_x                = np.sum(dxupar_zed_x_y,                axis=2)*dy
                pressure_tr_zed_x           = np.sum(pressure_tr_zed_x_y,           axis=2)*dy
                par_mom_tr_zed_x            = np.sum(par_mom_tr_zed_x_y,            axis=2)*dy

                # FFT for each zed value
                for i_zed in range(len(zed)):
                    reynolds_phi_nablax2_kx, kx  = get_fft_k(reynolds_phi_nablax2_zed_x[i_zed],    x)
                    reynolds_Pprp_nablax2_kx, kx = get_fft_k(reynolds_Pprp_nablax2_zed_x[i_zed],   x)
                    reynolds_phi_nablaxy_kx, kx  = get_fft_k(reynolds_phi_nablaxy_zed_x[i_zed],    x)
                    reynolds_Pprp_nablaxy_kx, kx = get_fft_k(reynolds_Pprp_nablaxy_zed_x[i_zed],   x)
                    dxP_kx,             _        = get_fft_k(dxP_zed_x[i_zed]            , x)
                    phi_kx,             _        = get_fft_k(phi_zed_x[i_zed]            , x)
                    dxphi_kx,           _        = get_fft_k(dxphi_zed_x[i_zed]          , x)
                    dupar_kx,           _        = get_fft_k(dupar_zed_x[i_zed]          , x)
                    dxupar_kx,          _        = get_fft_k(dxupar_zed_x[i_zed]         , x)
                    pressure_tr_kx,     _        = get_fft_k(pressure_tr_zed_x[i_zed]    , x)
                    par_mom_tr_kx,      _        = get_fft_k(par_mom_tr_zed_x[i_zed]     , x)

                    if i_zed == 0:
                        reynolds_phi_nablax2_zed_kx  = np.zeros((len(zed),len(kx)), dtype='complex')
                        reynolds_Pprp_nablax2_zed_kx = np.zeros((len(zed),len(kx)), dtype='complex')
                        reynolds_phi_nablaxy_zed_kx  = np.zeros((len(zed),len(kx)), dtype='complex')
                        reynolds_Pprp_nablaxy_zed_kx = np.zeros((len(zed),len(kx)), dtype='complex')
                        dxP_zed_kx                   = np.zeros((len(zed),len(kx)), dtype='complex')
                        phi_zed_kx                   = np.zeros((len(zed),len(kx)), dtype='complex')
                        dxphi_zed_kx                 = np.zeros((len(zed),len(kx)), dtype='complex')
                        dupar_zed_kx                 = np.zeros((len(zed),len(kx)), dtype='complex')
                        dxupar_zed_kx                = np.zeros((len(zed),len(kx)), dtype='complex')
                        pressure_tr_zed_kx           = np.zeros((len(zed),len(kx)), dtype='complex')
                        par_mom_tr_zed_kx            = np.zeros((len(zed),len(kx)), dtype='complex')
                    
                    reynolds_phi_nablax2_zed_kx[i_zed]  = reynolds_phi_nablax2_kx
                    reynolds_Pprp_nablax2_zed_kx[i_zed] = reynolds_Pprp_nablax2_kx
                    reynolds_phi_nablaxy_zed_kx[i_zed]  = reynolds_phi_nablaxy_kx
                    reynolds_Pprp_nablaxy_zed_kx[i_zed] = reynolds_Pprp_nablaxy_kx
                    dxP_zed_kx[i_zed]                   = dxP_kx
                    phi_zed_kx[i_zed]                   = phi_kx
                    dxphi_zed_kx[i_zed]                 = dxphi_kx
                    dupar_zed_kx[i_zed]                 = dupar_kx
                    dxupar_zed_kx[i_zed]                = dxupar_kx
                    pressure_tr_zed_kx[i_zed]           = pressure_tr_kx
                    par_mom_tr_zed_kx[i_zed]            = par_mom_tr_kx

            if i_time_idx == 0:
                EZ_t_kx                        = np.zeros((len(time_idx_eval),len(kx)))
                dEZ_reynolds_phi_nablax2_t_kx  = np.zeros((len(time_idx_eval),len(kx)))
                dEZ_reynolds_Pprp_nablax2_t_kx = np.zeros((len(time_idx_eval),len(kx)))
                dEZ_reynolds_phi_nablaxy_t_kx  = np.zeros((len(time_idx_eval),len(kx)))
                dEZ_reynolds_Pprp_nablaxy_t_kx = np.zeros((len(time_idx_eval),len(kx)))
                dEZ_vDx_P_t_kx                 = np.zeros((len(time_idx_eval),len(kx)))
                dEZ_upar_t_kx                  = np.zeros((len(time_idx_eval),len(kx)))
                dE_mean_pressure_tr_t_kx       = np.zeros((len(time_idx_eval),len(kx)))
                dE_delt_pressure_tr_t_kx       = np.zeros((len(time_idx_eval),len(kx)))
                dE_par_mom_tr_t_kx             = np.zeros((len(time_idx_eval),len(kx)))
                du_par_mom_tr_t_kx             = np.zeros((len(time_idx_eval),len(kx)))
                du_cos_par_mom_tr_t_kx         = np.zeros((len(time_idx_eval),len(kx)))

            for i_kx in range(len(kx)):

                mean_dxP_zed = np.sum(dl_over_B_avg*dxP_zed_kx[:,i_kx])
    
                EZ_t_kx[i_time_idx, i_kx]                        =  np.sum(dl_over_B_avg * 2*np.abs(dxphi_zed_kx[:,i_kx])**2 * nablax2/2 )
                dEZ_reynolds_phi_nablax2_t_kx[i_time_idx, i_kx]  =  np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(reynolds_phi_nablax2_zed_kx[:,i_kx])) )
                dEZ_reynolds_Pprp_nablax2_t_kx[i_time_idx, i_kx] =  np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(reynolds_Pprp_nablax2_zed_kx[:,i_kx])) )
                dEZ_reynolds_phi_nablaxy_t_kx[i_time_idx, i_kx]  =  np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(reynolds_phi_nablaxy_zed_kx[:,i_kx])) )
                dEZ_reynolds_Pprp_nablaxy_t_kx[i_time_idx, i_kx] =  np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(reynolds_Pprp_nablaxy_zed_kx[:,i_kx])) )
                dEZ_vDx_P_t_kx[i_time_idx, i_kx]                 = -np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(dxP_zed_kx[:,i_kx])) * vdriftx) *2
                dEZ_upar_t_kx[i_time_idx, i_kx]                  = -np.sum(dl_over_B_avg * 2*np.real(phi_zed_kx[:,i_kx]        *np.conj(dupar_zed_kx[:,i_kx]))) *2
                dE_mean_pressure_tr_t_kx[i_time_idx, i_kx]       =  np.sum(dl_over_B_avg * 2*np.real(pressure_tr_zed_kx[:,i_kx]*np.conj(mean_dxP_zed))) *2 * 4/7 # ~ (dP/dt)_{NL}
                dE_delt_pressure_tr_t_kx[i_time_idx, i_kx]       =  np.sum(dl_over_B_avg * 2*np.real(pressure_tr_zed_kx[:,i_kx]*np.conj((dxP_zed_kx[:,i_kx]-mean_dxP_zed)))) *2 * 4/7 # ~ (dP/dt)_{NL}
                dE_par_mom_tr_t_kx[i_time_idx, i_kx]             =  np.sum(dl_over_B_avg * 2*np.real(dxupar_zed_kx[:,i_kx]     *np.conj(par_mom_tr_zed_kx[:,i_kx]))) *2 # = (dU/dt)_{NL}
                du_par_mom_tr_t_kx[i_time_idx, i_kx]             =  np.sum(dl_over_B_avg * 2*np.real(dxphi_zed_kx[:,i_kx]     *np.conj(par_mom_tr_zed_kx[:,i_kx]))) *2 # = vE*(dU/dt)_{NL}
                du_cos_par_mom_tr_t_kx[i_time_idx, i_kx]         =  np.sum(dl_costheta   * 2*np.real(dxphi_zed_kx[:,i_kx]     *np.conj(par_mom_tr_zed_kx[:,i_kx]))) *2 # = vE*(dU/dt)_{NL}*cos(theta)

        # Time-average (note dt may vary over time)
        dt = np.gradient(time_eval)
        EZ_kx            = np.sum(EZ_t_kx*dt[:,None],           axis=0)/(time_eval[-1]-time_eval[0])
        dEZ_reynolds_phi_nablax2_kx  = np.sum(dEZ_reynolds_phi_nablax2_t_kx*dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
        dEZ_reynolds_Pprp_nablax2_kx = np.sum(dEZ_reynolds_Pprp_nablax2_t_kx*dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
        dEZ_reynolds_phi_nablaxy_kx  = np.sum(dEZ_reynolds_phi_nablaxy_t_kx*dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
        dEZ_reynolds_Pprp_nablaxy_kx = np.sum(dEZ_reynolds_Pprp_nablaxy_t_kx*dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
        dEZ_vDx_P_kx     = np.sum(dEZ_vDx_P_t_kx   *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
        dEZ_upar_kx      = np.sum(dEZ_upar_t_kx    *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
        dE_mean_pressure_tr_kx = np.sum(dE_mean_pressure_tr_t_kx  *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
        dE_delt_pressure_tr_kx = np.sum(dE_delt_pressure_tr_t_kx  *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
        dE_par_mom_tr_kx = np.sum(dE_par_mom_tr_t_kx  *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
        du_par_mom_tr_kx = np.sum(du_par_mom_tr_t_kx  *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])
        du_cos_par_mom_tr_kx = np.sum(du_cos_par_mom_tr_t_kx  *dt[:,None], axis=0)/(time_eval[-1]-time_eval[0])

        return kx, EZ_kx, dEZ_reynolds_phi_nablax2_kx, dEZ_reynolds_Pprp_nablax2_kx, dEZ_reynolds_phi_nablaxy_kx, dEZ_reynolds_Pprp_nablaxy_kx, dEZ_vDx_P_kx, dEZ_upar_kx, dE_mean_pressure_tr_kx, dE_delt_pressure_tr_kx, dE_par_mom_tr_kx, du_par_mom_tr_kx, du_cos_par_mom_tr_kx



    #######  Get constributions to zonal flow energy evolution equation as a function of x
    def get_dt_zonal_energy_contributions_x(self, time_idx=-1, nx=None, ny=None, kxmin_filter=np.infty, kymin_filter=np.infty, kxmax_filter=-1, kymax_filter=-1):

        reynolds_phi_nablax2_zed_x_y, zed, x, y, _ = self.get_quantity_zed_x_y("Reynolds_phi_nablax2", time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0) # without x-derivative in front, so need to multiply by dx2phiZ to get energy derivative
        reynolds_Pprp_nablax2_zed_x_y, zed, x, y, _ = self.get_quantity_zed_x_y("Reynolds_Pprp_nablax2", time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0) # without x-derivative in front, so need to multiply by dx2phiZ to get energy derivative
        reynolds_phi_nablaxy_zed_x_y, zed, x, y, _ = self.get_quantity_zed_x_y("Reynolds_phi_nablaxy", time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0) # without x-derivative in front, so need to multiply by dx2phiZ to get energy derivative
        reynolds_Pprp_nablaxy_zed_x_y, zed, x, y, _ = self.get_quantity_zed_x_y("Reynolds_Pprp_nablaxy", time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0) # without x-derivative in front, so need to multiply by dx2phiZ to get energy derivative
        dxP_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="pressure", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
        phi_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
        dxphi_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
        dx2phi_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=2, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kymax_filter=kymax_filter)
        upar_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y(quantity="upar", only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=0, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kymax_filter=kymax_filter)

        # Obtain deltaphi
        dl_over_B_avg = self.dl_over_B_avg()
        mean_phiZ_x_y = np.sum(phi_zed_x_y*dl_over_B_avg[:,None,None], axis=0)
        delta_phi_zed_x_y = np.zeros_like(phi_zed_x_y)
        for i_zed in range(len(zed)):
            delta_phi_zed_x_y[i_zed] = phi_zed_x_y[i_zed] - mean_phiZ_x_y
        deltaphiZ_2_zed_x_y = delta_phi_zed_x_y**2

        # Obtain parallel derivative of upar term
        _, _, _, _, gds22, bmag = self.get_FLR()
        #dupar_dzed_x_y = np.gradient(upar_zed_x_y/bmag[:,None,None], zed, axis=0) * bmag[:,None,None]
         #   f_zed_x_y = np.gradient(f_zed_x_y*gradpar[:,None,None], zed, axis=0)
        # Use periodicity
        dupar_zed_x_y = np.zeros_like(upar_zed_x_y)
        uparB_zed_x_y = upar_zed_x_y / bmag[:,None,None]
        gradpar  = self.ncdata.variables['gradpar'][:]
        dzed = zed[1]-zed[0]
        for i_zed in range(len(zed)-1):
            if i_zed == 0:
                dupar_zed_x_y[0] = (uparB_zed_x_y[1]-uparB_zed_x_y[-1]) / dzed
            else:
                dupar_zed_x_y[i_zed] = 0.5*(uparB_zed_x_y[i_zed+1]-uparB_zed_x_y[i_zed-1]) / dzed
        dupar_zed_x_y[-1] = (uparB_zed_x_y[0]-uparB_zed_x_y[-2]) / dzed

        dupar_zed_x_y = dupar_zed_x_y * (gradpar*bmag)[:,None,None]
#        print(dupar_zed_x_y[:,0,0])
        #print(np.shape(gradpar))
        #print(np.shape(bmag))
        #print(np.sum(dupar_zed_x_y[:,0,0]*dl_over_B_avg))
        #assert(np.abs(np.sum(dupar_zed_x_y[:,0,0]*dl_over_B_avg)) < 1e-14)

        # Get energies
        dy = y[1]-y[0]
        shat     = self.ncdata.variables['shat'].getValue()
        vdriftx = self.ncdata.variables['gbdrift0'][:,0]/(2*shat)
        nablax2 = gds22/bmag**2

        EZ_x           = np.sum(dl_over_B_avg[:,None,None]*0.5*dxphi_zed_x_y**2 *nablax2[:,None,None], axis=(0,2))*dy
        tau = 1
        EZ_deltaphi2_x =-np.sum(dl_over_B_avg[:,None,None]*tau*deltaphiZ_2_zed_x_y                             , axis=(0,2))*dy
        dEZ_reynolds_phi_nablax2_x  = np.sum(dl_over_B_avg[:,None,None]*dx2phi_zed_x_y*reynolds_phi_nablax2_zed_x_y              , axis=(0,2))*dy
        dEZ_reynolds_Pprp_nablax2_x = np.sum(dl_over_B_avg[:,None,None]*dx2phi_zed_x_y*reynolds_Pprp_nablax2_zed_x_y              , axis=(0,2))*dy
        dEZ_reynolds_phi_nablaxy_x  = np.sum(dl_over_B_avg[:,None,None]*dx2phi_zed_x_y*reynolds_phi_nablaxy_zed_x_y              , axis=(0,2))*dy
        dEZ_reynolds_Pprp_nablaxy_x = np.sum(dl_over_B_avg[:,None,None]*dx2phi_zed_x_y*reynolds_Pprp_nablaxy_zed_x_y              , axis=(0,2))*dy
        dEZ_vDx_x    =-np.sum(dl_over_B_avg[:,None,None]*2* phi_zed_x_y*dxP_zed_x_y  *vdriftx[:,None,None]     , axis=(0,2))*dy
        dEZ_upar_x   =-np.sum(dl_over_B_avg[:,None,None]*2*       phi_zed_x_y*dupar_zed_x_y                    , axis=(0,2))*dy

        return x, EZ_x, EZ_deltaphi2_x, dEZ_reynolds_phi_nablax2_x, dEZ_reynolds_Pprp_nablax2_x, dEZ_reynolds_phi_nablaxy_x, dEZ_reynolds_Pprp_nablaxy_x, dEZ_vDx_x, dEZ_upar_x


    #######  Get zonal flow energy contributions as a function of omega and x
    def get_EZ_omega_x(self, quantity, time_min=0, time_max=99999, time_idx_skip=1, nx=None):

        time = self.get_time_array()
        time_max = min(time[-1], time_max)
        if time_min < 0:
            time_min = time_max - np.abs(time_min)
        time_idx_min = np.argmin(np.abs(time-time_min))
        time_idx_max = np.argmin(np.abs(time-time_max))
        time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)
        time_eval = time[time_idx_eval]

        for i_time_idx, time_idx in enumerate(time_idx_eval):
            print("Evaluating zonal energy: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")
            kx_order = 0
            if quantity == "phi":
                kx_order = 1
            ### NOTE QUANTITY MUST NOT BE EQUAL PHI, BUT ASSUMED IN VARIABLE NAMING
            dxphi_zed_x_y, zed, x, y, _ = self.get_quantity_zed_x_y(quantity=quantity, only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=kx_order, nx=nx)

            if i_time_idx == 0:
                dxphi_t_zed_x = np.zeros((len(time_eval),len(zed),len(x)))

            dxphi_t_zed_x[i_time_idx] = np.sum(dxphi_zed_x_y, axis=2)*(y[1]-y[0])

        # Resample to equal time-intervals
        dt = (np.gradient(time_eval)).max()
        time_interp = np.arange(time_eval[0], time_eval[-1], dt)
        dxphi_t_zed_x_interp_func = interp(time_eval, dxphi_t_zed_x, assume_sorted=True, axis=0)
        dxphi_t_zed_x_interp = dxphi_t_zed_x_interp_func(time_interp)

        # Fourier transform to omega
        dxphi_omega_zed_x = np.fft.fft(dxphi_t_zed_x_interp, axis=0)/len(time_interp)
        omega = np.fft.fftfreq(len(time_interp), d=dt)*(2*np.pi)

        idx_sort = np.argsort(omega)
        dxphi_omega_zed_x = dxphi_omega_zed_x[idx_sort]
        omega = omega[idx_sort]

        # Geometric quantities
        dl_over_B_avg = self.dl_over_B_avg()
        _, _, _, _, gds22, bmag = self.get_FLR()
        nablax2 = gds22/bmag**2

        # Sum over zed
        EZ_omega_x     = np.sum(dl_over_B_avg[None,:,None]*0.5*np.abs(dxphi_omega_zed_x)**2 *nablax2[None,:,None], axis=1)

        return omega, x, EZ_omega_x


    #######  Get zonal flow energy contributions as a function of omega and x
    def get_EZ_omega_kx(self, quantity, time_min=0, time_max=99999, time_idx_skip=1, nx=None):

        # if time min is negative, count as tmax - |tmin|
        time = self.get_time_array()
        time_max = min(time[-1], time_max)
        if time_min < 0:
            time_min = time_max - np.abs(time_min)
        time_idx_min = np.argmin(np.abs(time-time_min))
        time_idx_max = np.argmin(np.abs(time-time_max))
        time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)
        time_eval = time[time_idx_eval]

        for i_time_idx, time_idx in enumerate(time_idx_eval):
            print("Evaluating zonal energy: time idx %5i/%5i" % (i_time_idx+1, len(time_idx_eval)), end="\r")
            ### NOTE QUANTITY MUST NOT BE EQUAL PHI, BUT ASSUMED IN VARIABLE NAMING
            kx_order = 0
            if quantity == "phi":
                kx_order = 1
            dxphi_zed_kx_ky, zed, kx, ky, _ = self.get_quantity_zed_kx_ky(quantity=quantity, only_zonal=True, remove_zonal=False, time_idx=time_idx, kx_order=kx_order)

            if i_time_idx == 0:
                dxphi_t_zed_kx = np.zeros((len(time_eval),len(zed),len(kx)), dtype='complex')

            #Ly = 2*np.pi/(ky[1]-ky[0])
            dxphi_t_zed_kx[i_time_idx] = dxphi_zed_kx_ky[:,:,0]# * Ly/2

        # Resample to equal time-intervals
        dt = (np.gradient(time_eval)).max()
        time_interp = np.arange(time_eval[0], time_eval[-1], dt)
        dxphi_t_zed_kx_interp_func = interp(time_eval, dxphi_t_zed_kx, assume_sorted=True, axis=0)
        dxphi_t_zed_kx_interp = dxphi_t_zed_kx_interp_func(time_interp)

        # Fourier transform to omega
        dxphi_omega_zed_kx = np.fft.fft(dxphi_t_zed_kx_interp, axis=0)/len(time_interp)
        omega = np.fft.fftfreq(len(time_interp), d=dt)*(2*np.pi)

        # Geometric quantities
        dl_over_B_avg = self.dl_over_B_avg()
        _, _, _, _, gds22, bmag = self.get_FLR()
        nablax2 = gds22/bmag**2

        # Sum over zed
        EZ_omega_kx    = np.sum(dl_over_B_avg[None,:,None]*0.5*np.abs(dxphi_omega_zed_kx)**2 *nablax2[None,:,None], axis=1)

        return omega, kx, EZ_omega_kx

    #######  Get zonal flow energy contributions as a function of omega
    def get_EZ_omega(self, quantity="phi", time_min=0, time_max=99999, time_idx_skip=1, nx=None):
        omega, x, EZ_omega_x   = self.get_EZ_omega_x(quantity=quantity, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, nx=nx)
        # Sum over x
        EZ_omega     = np.sum(EZ_omega_x, axis=1)*(x[1]-x[0])
        return omega, EZ_omega
        

    #######  Get energy balance over time
    # Units:
    def get_energies_over_time(self, species_idx=0):

        if self.code == "stella":
            print("To be implemented.")
     

        elif self.code == "GS2":
            time = self.get_time_array()

            delfs2 = self.ncdata['heating_energy_delfs2'][:,species_idx]
            hs2    = self.ncdata['heating_energy_hs2'][:,species_idx]
            phis2  = self.ncdata['heating_energy_phis2'][:,species_idx]

        return delfs2, hs2, phis2, time

    #######  Get lowest order moments squared, over time
    # Units:
    def get_moments2_over_time(self, species_idx=0, remove_zonal=True):

        if self.code == "stella":
            print("To be implemented.")

        elif self.code == "GS2":
            time = self.get_time_array()

            # Use moments
            if remove_zonal:
                phi2_by_ky    = self.ncdata['phi2_by_ky'][:,:]
                dens2_by_ky   = self.ncdata['density2_by_ky'][:,species_idx,:]
                upar2_by_ky   = self.ncdata['upar2_by_ky'][:   ,species_idx,:]
                tpar2_by_ky   = self.ncdata['tpar2_by_ky'][:   ,species_idx,:]
                tperp2_by_ky  = self.ncdata['tperp2_by_ky'][:  ,species_idx,:]

                phi2_by_ky[:,0]   = 0
                dens2_by_ky[:,0]  = 0
                upar2_by_ky[:,0]  = 0
                tpar2_by_ky[:,0]  = 0
                tperp2_by_ky[:,0] = 0

                phi2   = np.sum(phi2_by_ky,   axis=1)
                dens2  = np.sum(dens2_by_ky,  axis=1)
                upar2  = np.sum(upar2_by_ky,  axis=1)
                tpar2  = np.sum(tpar2_by_ky,  axis=1)
                tperp2 = np.sum(tperp2_by_ky, axis=1)
            else:
                phi2  = self.ncdata['phi2'][:]
                dens2  = self.ncdata['ntot2'][:,species_idx]
                upar2  = self.ncdata['upar2'][:,species_idx]
                tpar2  = self.ncdata['tpar2'][:,species_idx]
                tperp2 = self.ncdata['tperp2'][:,species_idx]

        return phi2, dens2, upar2, tpar2, tperp2, time

    #######  Plot total fluxes over time
    def plot_flux_over_time(self, axs=None, label=None, species_idx=0, ls='-', color=None, marker=None, timeavg=None, timemax=np.infty, log=False):
        if axs is None:
            fig, axs = plt.subplots(3,1,figsize=(12,9))
            #plt.subplots_adjust(hspace=0)
            plt.subplots_adjust(hspace=0,left=0.08,right=0.95,top=0.9,bottom=0.1,wspace=0.45)

        # Plot fluxes
        pflx, vflx, qflx, time = self.get_fluxes_over_time(species_idx=species_idx)

        if log:
            pflx = np.abs(pflx)
            vflx = np.abs(vflx)
            qflx = np.abs(qflx)

        vflx = np.nan_to_num(vflx, 1)

        if timeavg is not None:
            timemax  = min(timemax, time[-1])
            pflx_avg = np.average(pflx[(time > timemax-timeavg) & (time <= timemax)])
            vflx_avg = np.average(vflx[(time > timemax-timeavg) & (time <= timemax)])
            qflx_avg = np.average(qflx[(time > timemax-timeavg) & (time <= timemax)])

            print(self.filename_base + ": qflx_avg = %e" % (qflx_avg))

            xmin_plot = max(timemax-timeavg, 0)
            xmax_plot = timemax
            axs[0].plot([xmin_plot, xmax_plot], [pflx_avg, pflx_avg], ls=ls, marker=marker, c='0.5', lw=2)
            axs[1].plot([xmin_plot, xmax_plot], [vflx_avg, vflx_avg], ls=ls, marker=marker, c='0.5', lw=2)
            axs[2].plot([xmin_plot, xmax_plot], [qflx_avg, qflx_avg], ls=ls, marker=marker, c='0.5', lw=2)

        #axs[0].plot(time, pflx, label=label, marker=marker)
        #axs[1].plot(time, vflx, label=label, marker=marker)
        #axs[2].plot(time, qflx, label=label, marker=marker)
        axs[0].plot(time, pflx, label=label, ls=ls, marker=marker, c=color)
        axs[1].plot(time, vflx, label=label, ls=ls, marker=marker, c=color)
        axs[2].plot(time, qflx, label=label, ls=ls, marker=marker, c=color)

        if log:
            for ax in axs:
                ax.set_yscale('log')

        axs[0].set_xticklabels([])
        axs[1].set_xticklabels([])

        axs[0].set_ylabel(r"$\Gamma$")
        #axs[1].set_ylabel(r"$Q$")
        axs[1].set_ylabel(r"$\Pi$")
        axs[2].set_ylabel(r"$Q$")
        axs[2].set_xlabel(r"$t$")

        axs[0].legend()

        axs[0].set_xlim(xmin=0)
        axs[1].set_xlim(xmin=0)
        axs[2].set_xlim(xmin=0)

        axs[0].grid()
        axs[1].grid()
        axs[2].grid()
            

        if timeavg is not None:
            return axs, pflx_avg, vflx_avg, qflx_avg
        else:
            return axs

    #######  Plot fluxes as a function of (ky, zed) for given slices of kx and t
    def plot_flux_spectra(self, fig=None, ax=None, species_idx=0, tube=0, time_idx=-1, kx_idx=0):

        qflx_t_zed_kx_ky, time, zed, kx, ky = self.read_flux_spectra(species_idx, tube)

        if fig is None and ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        print("Note len(ky) = %i, len(zed) = %i, len(kx) = %i, len(t) = %i." % (len(ky), len(zed), len(kx), len(time)))

        Y, X = np.meshgrid( ky, zed)
        Z = np.abs(qflx_t_zed_kx_ky)[time_idx, :, kx_idx, :]

        eps_rel = 1e-4
        im = ax.pcolormesh(X, Y, Z, 
        #im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=max(Z.min(), eps_rel*Z.max()), vmax=Z.max()),
                   shading='auto', cmap='inferno')

        ax.set_xlabel(r"$\zeta$ (scaled)")
        ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax.set_ylabel(r"$k_y \rho_i$")

        return fig, ax, im, time[time_idx], kx[kx_idx]

    #######  Plot tube-averaged fluxes as a function of (kx, ky) for given t
    def plot_flux_spectra_kx_ky(self, fig=None, ax=None, species_idx=0, tube=0, time_idx=-1, normalise_ky=False):

        qflx_t_zed_kx_ky, time, zed, kx, ky = self.read_flux_spectra(species_idx, tube)

        qflx_zed_kx_ky = qflx_t_zed_kx_ky[time_idx]

        # zeta-summed
        dl_over_B_avg = self.dl_over_B_avg()
        qflx_kx_ky = np.sum(qflx_zed_kx_ky*dl_over_B_avg[:,None,None], axis=0)

        if fig is None and ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        Y, X = np.meshgrid( ky, kx)

        Z = np.abs(qflx_kx_ky)

        if normalise_ky:
            for i_ky in range(len(ky)):
                Z[:,i_ky] = Z[:,i_ky]/ky[i_ky]

        eps_rel = 1e-4
        im = ax.pcolormesh(X, Y, Z, 
        #im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=max(Z.min(), eps_rel*Z.max()), vmax=Z.max()),
                   shading='auto', cmap='inferno')

        ax.set_xlabel(r"$k_x \rho_i$")
        ax.set_ylabel(r"$k_y \rho_i$")
        ax.set_title(r"$Q_{k_x, k_y} (t=%.2f)$" % (time[time_idx]))

        return fig, ax, im

    def plot_quantities_over_zed(self, fig=None, ax=None, mult_zed=1, zed_times_nfield_periods=False, time_idx=-1, ls=None, color=None, norm_all=False, **kwargs):


        plot_phi       = kwargs.get('plot_phi'       , False)
        norm_phi       = kwargs.get('norm_phi'       , True)
        log_phi        = kwargs.get('log_phi'        , False)
        time_avg       = kwargs.get('time_avg'       , None)
        kx_idx_phi     = kwargs.get('kx_idx_phi'     , 0)
        ky_idx_phi     = kwargs.get('ky_idx_phi'     , 0)
        eval_reim_phi  = kwargs.get('eval_reim_phi'  , True)
        squared_phi    = kwargs.get('squared_phi'    , False)
        remove_zonal_phi = kwargs.get('remove_zonal_phi', False)
        label_phi      = kwargs.get('label_phi'      , "")
        return_phi     = kwargs.get('return_phi'     , False)

        plot_nablax2   = kwargs.get('plot_nablax2'   , False)
        plot_nablaxy   = kwargs.get('plot_nablaxy'   , False)
        plot_nablay2   = kwargs.get('plot_nablay2'   , False)
        plot_B         = kwargs.get('plot_B'         , False)
        norm_B         = kwargs.get('norm_B'         , False)
        plot_Gamma0    = kwargs.get('plot_Gamma0'    , False)
        plot_omega_s_k = kwargs.get('plot_omega_s_k' , False)
        norm_omega_s_k = kwargs.get('norm_omega_s_k' , True)
        plot_gi        = kwargs.get('plot_gi'        , False)
        plot_ge        = kwargs.get('plot_ge'        , False)
        norm_factor_omega_s_k = kwargs.get('norm_factor_omega_s_k' , None)
        plot_qflx      = kwargs.get('plot_qflx',       False)

        _, _, gds2, gds21, gds22, bmag = self.get_FLR(ky_idx=0, kx_idx=0)

        if fig is None and ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        zed = self.ncdata.variables['zed'][:] * mult_zed
        time_val = self.ncdata.variables['t'][time_idx]
        set_xlim = True
        if zed_times_nfield_periods:
            geom_quantities = np.loadtxt(self.geo_file, skiprows=2).T
            zed = geom_quantities[1] * mult_zed
            set_xlim = False

        if plot_Gamma0:
            Gamma0   = self.get_Gamma0()
            plot_y_over_zed(ax, zed, Gamma0, label=r"$\Gamma_0(b)$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)

        if plot_omega_s_k:
            omega_s_k, _, _ = self.get_omega_s_k()
            if norm_omega_s_k:
                if norm_factor_omega_s_k is None:
                    norm_factor_omega_s_k = np.abs(omega_s_k).max()
                print("omega star over omega curvature normalised by %e" % (norm_factor_omega_s_k))
                omega_s_k = omega_s_k / norm_factor_omega_s_k

            plot_y_over_zed(ax, zed, omega_s_k, label=r"$\omega_\star^T / \omega_\kappa$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)


        if plot_B:
            if norm_B:
                bmag = bmag/bmag.max()
            plot_y_over_zed(ax, zed, bmag, label=r"$B$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)

        if plot_nablax2:
            plot_y_over_zed(ax, zed, gds22, label=r"$|\nabla x|^2$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)
        if plot_nablaxy:
            plot_y_over_zed(ax, zed, gds21, label=r"$\nabla x\cdot\nabla y$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)
        if plot_nablay2:
            plot_y_over_zed(ax, zed, gds2,  label=r"$|\nabla y|^2$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all)

        if plot_phi:
            phi_t, _ = self.read_phi_vs_zed(normalise_phi=norm_phi, time_avg=time_avg, time_idx=time_idx, kx_idx=kx_idx_phi, ky_idx=ky_idx_phi, eval_real=False, squared=squared_phi, remove_zonal=remove_zonal_phi)

            if log_phi:
                phi_plot = np.log(np.abs(phi_t))
                eval_reim_phi = False
            else:
                phi_plot = phi_t

            if eval_reim_phi:
                if ky_idx_phi != 0:
                    label    = r"$\varphi_r$"
                    plot_y_over_zed(ax, zed, np.real(phi_plot), label=label+label_phi, set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, lw=2)
                    label    = r"$\varphi_i$"
                    plot_y_over_zed(ax, zed, np.imag(phi_plot), label=label+label_phi, set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, lw=1)
                else:
                    plot_y_over_zed(ax, zed, np.real(phi_plot), label=label_phi, set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, lw=2)

            else:
                if log_phi:
                    label    = r"$\mathrm{log}|\varphi|$"
                else:
                    label    = r"$|\varphi|$"
                if squared_phi:
                    label = label + r"$^2$"
                plot_y_over_zed(ax, zed, phi_plot, label=label+label_phi, set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, lw=1)


        if plot_qflx:
            qflx_t_zed_kx_ky, _, _, _, _ = self.read_flux_spectra(species_idx=0, tube=0)
            qflx = np.sum( qflx_t_zed_kx_ky[time_idx], axis=(1,2))
            if norm_all:
                qflx = qflx/qflx.max()
            plot_y_over_zed(ax, zed, qflx, label=r"$Q$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, lw=2)


        #vpa_index = 0
        vpa_index = None
        if plot_gi:
            gz_i, _ = self.read_g_vs_zed(species_idx=0, vpa_index=vpa_index, time_idx=time_idx, normalise=False)
            norm_g = np.abs(gz_i).max()
            plot_y_over_zed(ax, zed, gz_i/norm_g, label=r"$g_i$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, alpha=0.5)

        if plot_ge:
            gz_e, _ = self.read_g_vs_zed(species_idx=1, vpa_index=vpa_index, time_idx=time_idx, normalise=False)
            #if not plot_gi:
            norm_g = np.abs(gz_e).max()
            plot_y_over_zed(ax, zed, gz_e/norm_g, label=r"$g_e$", set_xlim=set_xlim, ls=ls, color=color, norm=norm_all, alpha=0.5)

        ax.set_title(r"$t=%.2f$" % (time_val))

        if return_phi:
            return fig, ax, phi_plot, zed
        else:
            return fig, ax
            
    def plot_quantity_zed_t(self, quantity, fig=None, ax=None, vmin=None, vmax=None, species_idx=0, logarithmic=False, remove_zonal=False, only_zonal=False, sideband=False, time_idx_skip=1, normalise_each_t=False, cmap='inferno', kx_order=0, ky_order=0, nx=None, ny=None, avg_norm=None, time_min=0, time_max=99999, mult_zed=None, kxmin_filter=np.infty, plot_zed_avg=True):


        zed    = self.ncdata.variables['zed'][:]
#        kx     = self.ncdata.variables['kx'][:]
#        ky     = self.ncdata.variables['ky'][:]

        time_all   = self.ncdata.variables['t'][:]
        time_idx_min = np.argmin(np.abs(time_all-time_min))
        time_idx_max = np.argmin(np.abs(time_all-time_max))
        time_plot    = time_all[time_idx_min:time_idx_max:time_idx_skip]
        time_idxs = range(time_idx_min, time_idx_max, time_idx_skip)
        assert(len(time_plot)==len(time_idxs))
        assert(len(time_plot)>0)

        zed_weight = self.get_zed_weight(mult_zed=mult_zed, zed=zed)

        f_t_zed = np.zeros((len(time_plot), len(zed) ))
        for i_idx, time_idx in enumerate(time_idxs):
            print("Evaluating time_idx %.6i/%i..." % (i_idx+1, len(time_idxs)), end="\r")

            x_der_taken = False
            y_der_taken = False
            if quantity == "phi-phi":
                phi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, remove_zonal=True, only_zonal=False, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                f_zed_x_y = phi_zed_x_y**2

            elif quantity == "phi-pressure_perp":
                phi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                Pprp_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                f_zed_x_y = phi_zed_x_y * Pprp_zed_x_y

            elif quantity == "dyphi-T":
                dyphi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                T_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("temperature", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                f_zed_x_y = dyphi_zed_x_y * T_zed_x_y

            elif quantity == "dyphi-upar":
                dyphi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                upar_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("upar", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                f_zed_x_y = dyphi_zed_x_y * upar_zed_x_y

            elif quantity == "dyphi-P":
                dyphi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                P_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("pressure", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                f_zed_x_y = dyphi_zed_x_y * P_zed_x_y

            elif quantity == "dyphi-chi":
                dyphi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                chi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("chi", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                f_zed_x_y = dyphi_zed_x_y * chi_zed_x_y

            elif quantity == "dyphi-dyPprp":
                dyphi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                dyPprp_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                f_zed_x_y = dyphi_zed_x_y * dyPprp_zed_x_y

            elif quantity == "dxphi-dyPprp":
                dxphi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, kx_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                dyPprp_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                f_zed_x_y = dxphi_zed_x_y * dyPprp_zed_x_y

            elif quantity == "dyphi-dyphi":
                dyphi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                f_zed_x_y = dyphi_zed_x_y**2

            elif quantity == "kx-avg":
                phi_zed_kx_ky, zed, kx, ky, time_eval = self.get_quantity_zed_kx_ky("phi", time_idx=time_idx, species_idx=species_idx, kxmin_filter=kxmin_filter)
                kx_avg = np.sum( kx[None,:,None] * np.abs(phi_zed_kx_ky[:,:,1:])**2, axis=(1,2)) / np.sum( np.abs(phi_zed_kx_ky[:,:,1:])**2, axis=(1,2))

                f_zed_x_y = kx_avg[:,None,None]

            elif quantity == "dxphi-dyphi":
                dxphi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, kx_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                dyphi_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                f_zed_x_y = dxphi_zed_x_y * dyphi_zed_x_y

            else:
                f_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity=quantity, time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, nx=nx, ny=ny, kxmin_filter=kxmin_filter)
                x_der_taken = True
                y_der_taken = True

            # Take derivatives by finite differences if needed
            if not x_der_taken:
                for i in range(kx_order):
                    f_zed_x_y = np.gradient(f_zed_x_y, axis=1)/(x[1]-x[0])
            if not y_der_taken:
                for i in range(ky_order):
                    f_zed_x_y = np.gradient(f_zed_x_y, axis=2)/(y[1]-y[0])

            # Average over x-y
            if avg_norm == "abs":
                f_zed = np.sum( np.abs(f_zed_x_y), axis=(1,2) )
            elif avg_norm == 2:
                f_zed = np.sqrt( np.sum( f_zed_x_y**2, axis=(1,2) ) )
            elif avg_norm == "center":
                f_zed = f_zed_x_y[:,0,0]
            elif avg_norm == "zonal_center":
                f_zed = np.sum(f_zed_x_y[:,0], axis=1)
            else:
                f_zed = np.sum( f_zed_x_y, axis=(1,2) )

            # Save to array
            f_t_zed[i_idx] = f_zed

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,10))
            plt.subplots_adjust(left=0.15,right=0.95)

        f_t_zed = f_t_zed*zed_weight[None,:]

        X, Y = np.meshgrid(time_plot, zed)
        Z    = f_t_zed.T

        if normalise_each_t:
            for time_idx in range(len(time_plot)):
                Z[:,time_idx] = Z[:,time_idx]/max(np.abs(Z[:,time_idx]))

        if fig is None and ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        if logarithmic:
            Z = np.abs(Z)

        if vmax is None:
            vmax = Z.max()
        if vmax == "last":
            vmax = np.abs(Z[:,-1]).max()
        if vmin == "symm":
            vmin = -vmax
        elif vmin is None:
            if logarithmic:
                vmin = 1e-2*vmax
            else:
                vmin = Z.min()

        if logarithmic:
            im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
        else:
            im = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)

        if plot_zed_avg:
            zed_avg_t = np.sum(np.abs(f_t_zed)*zed[None,:], axis=1)/np.sum(np.abs(f_t_zed), axis=1)*10
            ax.plot(time_plot, zed_avg_t, ls='--', lw=2, c='k')

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\zeta$")

        return fig, ax, im

    # Parallel correlation function, (14) in Barnes et al. 2011
    def plot_parallel_correlation_function(self, quantity="phi", time_idx=-1, time_avg=0, fig=None, ax=None, zeta_max=False, k_min=None, k_max=None, no_plot=False, kx_instead_of_ky=False, keep_only_zonal=False, vmin=None, vmax=None):

        time = self.get_time_array(GX_big=True)
        time_max = time[time_idx]
        time_idx_min = np.argmin(np.abs(time-(time_max-time_avg)))-3
        #time_idx     = np.argmin(np.abs(time-(time_max)))
        #dt = np.gradient(time[idx_min:time_idx])
        #quantity = np.sum(quantity_vs_t[idx_min:time_idx]*dt, axis=0)/np.sum(dt)

        if quantity == "phi":
            if self.code == "stella":
                # phi_vs_t(t, tube, zed, theta0, ky, ri)
                quantity = np.mean(self.ncdata.variables['phi_vs_t'][time_idx_min:time_idx,0,:,:,:,:],axis=0) # zed-kx-ky-ri
            elif self.code == "GX":
                if self.GX_old_version:
                    #quantity = np.transpose( np.mean(self.ncdata['Special']['Phi_z'][time_idx_min:time_idx], axis=0) , axes=[2,1,0,3])
                    print("WARNING! You are loading Phi_z in GX, which had issues in early code versions.")
                    quantity = np.transpose( self.ncdata['Special']['Phi_z'], axes=[2,1,0,3] )
                else:
                    quantity = np.transpose( np.mean(self.ncdata_big['Diagnostics']['Phi'][time_idx_min:time_idx], axis=0) , axes=[2,1,0,3])
            elif self.code == "GS2":
                quantity = np.transpose( self.ncdata.variables['phi'] , axes=[2,1,0,3] )

        elif quantity == "temperature":
            # temperature(t, species, tube, zed, kx, ky, ri)
            quantity = np.mean(self.ncdata.variables['temperature'][time_idx_min:time_idx,0,0,:,:,:,:],axis=0) # zed-kx-ky-ri

        elif quantity == "upar":
            quantity = np.mean(self.ncdata.variables['upar'][time_idx_min:time_idx,0,0,:,:,:,:],axis=0) # zed-kx-ky-ri
        else:
            print("ENTER VALID QUANTITY!")
            return 

        if keep_only_zonal:
            quantity[:,:,1:,:] = 0
            quantity[:,0,0,:] = 0
        else:
            quantity[:,:,0,:] = 0
    
        kx, ky, zed = self.get_kx_ky_zed()

        if self.code == "GX" and self.GX_old_version:
            #print("-------------------------")
            #print(quantity[zed > 0, 3, 3, 3, 0])
            #print(quantity[zed < 0, 3, 3, 3, 0])
            #print("-------------------------")
            quantity[:,kx>0,:,:] = 0

        assert(np.shape(quantity)[0] == len(zed))
        assert(np.shape(quantity)[1] == len(kx))
        assert(np.shape(quantity)[2] == len(ky))
        assert(np.shape(quantity)[3] == 2)

        if zeta_max:
            # Find zed where phi peaks
            quantity_sum = np.sum(np.abs(quantity[:,:,:,0]+1j*quantity[:,:,:,1])**2, axis=(1,2))
            arg_zed_ctr = np.argmax(phi_sum)
            print("zeta(phi=phi_max)/zeta_max = %e" % (zed[arg_zed_ctr]/np.max(zed)))
        else:
            # Find zed=0
            arg_zed_ctr = np.argmin(np.abs(zed))

        if kx_instead_of_ky:
            idx_kx_sort = np.argsort(kx)
            k = kx[idx_kx_sort]
            k_other = ky
            quantity_tmp = quantity[:,idx_kx_sort]
            quantity     = np.transpose(quantity_tmp, (0,2,1,3))
        else:
            k = ky
            k_other = kx
        correlation_func = np.zeros(shape=(len(zed), len(k)))

        for i_k in range(len(k)):
            f_C_k_ctr = quantity[arg_zed_ctr,:, i_k,0] + 1j*quantity[arg_zed_ctr,:,i_k,1]
            for i_zed in range(len(zed)):

                f_C_k = quantity[i_zed,:,i_k,0] + 1j*quantity[i_zed,:,i_k,1]

                if k_min is None:
                    k_min = 0
                if k_max is None:
                    k_max = np.infty

                idx_k = np.where( (np.abs(k_other)>k_min) & (np.abs(k_other)<k_max))
                #idx_k = np.abs(k_other) > k_min
                correlation_func[i_zed, i_k] = np.sum( np.real( f_C_k[idx_k] * np.conj(f_C_k_ctr[idx_k]))) / np.sum( np.abs(f_C_k_ctr[idx_k])**2 )
                    #correlation_func[i_zed, i_k] = np.sum( np.real( f_C_k * np.conj(f_C_k_ctr))) / np.sum( np.abs(f_C_k_ctr)**2 )

        if not no_plot:
            if ax is None:
                fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))
    
            X, Y = np.meshgrid(zed, k)
            Z = correlation_func.T
    
            im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)
    
            ax.set_xlabel(r"$\Delta \zeta$")
            if kx_instead_of_ky:
                ax.set_ylabel(r"$k_x \rho_i$")
            else:
                ax.set_ylabel(r"$k_y \rho_i$")
            ax.set_title(r"$\mathcal{C}$")
        else:
            fig = None
            ax = None
            im = None

        # Evaluate average delta-chi
        avg_delta_chi = np.zeros_like(k)
        for i_k in range(len(k)):
            avg_delta_chi[i_k] = np.mean(correlation_func[:, i_k])

        return fig, ax, im, avg_delta_chi, k

    # Parallel correlation function as a function of kx and ky
    def get_parallel_correlation_function_kx_ky(self, quantity="phi", time_idx=-1, zeta_max=False, k_min=None):

        if quantity == "phi":
            # phi_vs_t(t, tube, zed, theta0, ky, ri)
            phi_vs_t = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:] # zed-kx-ky-ri
        elif quantity == "temperature":
            # temperature(t, species, tube, zed, kx, ky, ri)
            phi_vs_t = self.ncdata.variables['temperature'][time_idx,0,0,:,:,:,:]

        zed      = self.ncdata.variables['zed'][:]
        ky       = self.ncdata.variables['ky'][:]
        kx       = self.ncdata.variables['kx'][:] 

        if zeta_max:
            # Find zed where phi peaks
            phi_sum = np.sum(np.abs(phi_vs_t[:,:,:,0]+1j*phi_vs_t[:,:,:,1])**2, axis=(1,2))
            arg_zed_ctr = np.argmax(phi_sum)
            print("zeta(phi=phi_max)/zeta_max = %e" % (zed[arg_zed_ctr]/np.max(zed)))
        else:
            # Find zed=0
            arg_zed_ctr = np.argmin(np.abs(zed))

        idx_kx_sort = np.argsort(kx)
        kx = kx[idx_kx_sort]
        phi_vs_t = phi_vs_t[:,idx_kx_sort]

        correlation_func_zed_kx_ky = np.zeros(shape=(len(zed), len(kx), len(ky)))

        phi_C_k_ctr  = phi_vs_t[arg_zed_ctr,:,:,0] + 1j*phi_vs_t[arg_zed_ctr,:,:,1]

        for i_zed in range(len(zed)):
            phi_C_k_delt = phi_vs_t[i_zed,:,:,0] + 1j*phi_vs_t[i_zed,:,:,1]
            correlation_func_zed_kx_ky[i_zed] = np.abs(phi_C_k_delt)/np.abs(phi_C_k_ctr)
            #correlation_func_zed_kx_ky[i_zed] = np.real(phi_C_k_ctr*np.conj(phi_C_k_delt))#/np.abs(phi_C_k_ctr)**2

        # Evaluate average delta-chi
        avg_delta_chi = np.mean(correlation_func_zed_kx_ky, axis=0)

        time  = self.ncdata.variables['t'][time_idx]
        return correlation_func_zed_kx_ky, avg_delta_chi, kx, ky, time

# Get quantity (zed, kx, ky)
    def get_quantity_zed_kx_ky(self, quantity, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, alt_slow_eval=False):

        if remove_zonal and only_zonal:
            print("WARNING! Both only_zonal and remove_zonal were set to True, will thus return f_x_y = 0.")

        kx, ky, zed = self.get_kx_ky_zed()
        time_all    = self.get_time_array(GX_big=True)

        if time_avg is not None:
            time_eval = time_all[time_idx]
            time_min  = max(0,            time_eval-time_avg/2)
            time_max  = min(time_all[-1], time_eval+time_avg/2)
            time_idx_min = np.argmin( np.abs(time_all-time_min) )
            time_idx_max = np.argmin( np.abs(time_all-time_max) )
            time_idx = np.arange(time_idx_min,time_idx_max+1)
 
        if time_val is not None:
            time_idx = np.argmin( np.abs(time_all-time_val) )

        if not alt_slow_eval:
            if quantity=="phi":
                if self.code == "stella":
                    # phi_vs_t(t, tube, zed, theta0, ky, ri)
                    f_zed_kx_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
                elif self.code == "GS2":
                    f_kx_ky_ri = np.transpose( self.ncdata.variables['phi_igomega_by_mode'][time_idx] , axes=[1,0,2] )
                elif self.code == "GX":
                    if self.GX_old_version:
                        print("WARNING! You are loading Phi_z in GX, which had issues in early code versions.")
                        f_zed_kx_ky_ri = np.transpose( self.ncdata['Special']['Phi_z'], axes=[2,1,0,3] )
                    else:
                        f_zed_kx_ky_ri = np.transpose( self.ncdata_big['Diagnostics']['Phi'][time_idx] , axes=[2,1,0,3] )

            elif quantity=="deltaphi":
                f_zed_kx_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
                zed_weight = self.dl_over_B_avg()
                f_zed_kx_ky_ri = f_zed_kx_ky_ri - np.sum(f_zed_kx_ky_ri*zed_weight[:,None,None,None], axis=0)

            elif quantity=="(1-Gamma0)phi":
                kperp2_zed_kx_ky = self.ncdata.variables['kperp2'][:][:,species_idx,:,:] 
                Gamma0 = specialfunc.iv(0, kperp2/2) * np.exp(-kperp2/2)
                f_zed_kx_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]*(1-Gamma0)[:,:,:,None]
 
            elif quantity=="E_Z":
                if self.code == "stella":
                    # phi_vs_t(t, tube, zed, theta0, ky, ri)
                    f_zed_kx_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
                    f_zed_kx_ky_ri[:,:,1:,:] = 0
                    f_zed_kx_ky = (f_zed_kx_ky_ri[:,:,:,0] + 1j*f_zed_kx_ky_ri[:,:,:,1])*1j*kx[None,:,None]
                    f_zed_kx_ky_ri[:,:,:,0] = np.abs(f_zed_kx_ky)**2 /2
                    f_zed_kx_ky_ri[:,:,:,1] = 0

            elif quantity=="density":
                # density(t, species, tube, zed, kx, ky, ri)
                f_zed_kx_ky_ri = self.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
            elif quantity=="qpar":
                f_zed_kx_ky_ri = self.ncdata.variables['qpar'][time_idx,species_idx,0,:,:,:,:]
            elif quantity=="upar":
                # upar(t, species, tube, zed, kx, ky, ri)
                f_zed_kx_ky_ri = self.ncdata.variables['upar'][time_idx,species_idx,0,:,:,:,:]
            #elif quantity=="upar_over_phi":
            #    # upar(t, species, tube, zed, kx, ky, ri)
            #    f_zed_kx_ky_ri = self.ncdata.variables['upar'][time_idx,species_idx,0,:,:,:,:]/self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
            elif quantity=="temperature":
                if self.code == "stella":
                    # temperature(t, species, tube, zed, kx, ky, ri)
                    f_zed_kx_ky_ri = self.ncdata.variables['temperature'][time_idx,species_idx,0,:,:,:,:]
                elif self.code == "GX":
                    Tpar_zed_kx_ky_ri = np.transpose( self.ncdata_big['Diagnostics']['Tpar'][time_idx, species_idx] , axes=[2,1,0,3] )
                    Tprp_zed_kx_ky_ri = np.transpose( self.ncdata_big['Diagnostics']['Tperp'][time_idx, species_idx] , axes=[2,1,0,3] )
                    f_zed_kx_ky_ri = Tpar_zed_kx_ky_ri + Tprp_zed_kx_ky_ri
            elif quantity=="temperature_par": #(xpa^2-1/2)
                if self.code == "stella":
                    # temperature(t, species, tube, zed, kx, ky, ri)
                    P_zed_kx_ky_ri    = self.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
                    try:
                        Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
                    except:
                        Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
                    n_zed_kx_ky_ri    = self.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
                    f_zed_kx_ky_ri = P_zed_kx_ky_ri - 0.5*Pprp_zed_kx_ky_ri - 0.5*n_zed_kx_ky_ri
                elif self.code == "GX":
                    f_zed_kx_ky_ri = np.transpose( self.ncdata_big['Diagnostics']['Tpar'][time_idx, species_idx] , axes=[2,1,0,3] )
            elif quantity=="temperature_perp": #(xprp^2-1)
                if self.code == "stella":
                    # temperature(t, species, tube, zed, kx, ky, ri)
                    try:
                        Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
                    except:
                        Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
                    n_zed_kx_ky_ri    = self.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
                    f_zed_kx_ky_ri = Pprp_zed_kx_ky_ri - n_zed_kx_ky_ri
                elif self.code == "GX":
                    f_zed_kx_ky_ri = np.transpose( self.ncdata_big['Diagnostics']['Tperp'][time_idx, species_idx] , axes=[2,1,0,3] )
            elif quantity=="pressure":
                f_zed_kx_ky_ri = self.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
            elif quantity=="chi":
                try:
                    f_zed_kx_ky_ri = self.ncdata.variables['chi'][time_idx,species_idx,0,:,:,:,:]
                except:
                    print("chi not found in NETCDF! Using pressure instead.")
                    f_zed_kx_ky_ri = self.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]

            #elif quantity=="temp_over_phi":
            #    # upar(t, species, tube, zed, kx, ky, ri)
            #    f_zed_kx_ky_ri = self.ncdata.variables['temperature'][time_idx,species_idx,0,:,:,:,:]/self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
            elif quantity=="pressure_par":
                P_zed_kx_ky_ri = self.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
                Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
                f_zed_kx_ky_ri = P_zed_kx_ky_ri-0.5*Pprp_zed_kx_ky_ri
            elif quantity=="pressure_perp":
                # pressure_perp(t, species, tube, zed, kx, ky, ri)
                try:
                    f_zed_kx_ky_ri = self.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
                except:
                    f_zed_kx_ky_ri = self.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
            elif quantity=="qflx":
               # qflx_kxky(t, species, tube, zed, kx, ky)
               f_zed_kx_ky = self.ncdata.variables['qflx_kxky'][time_idx,species_idx,0,:,:,:]
               f_zed_kx_ky_ri = np.zeros( (len(zed), len(kx), len(ky), 2))
               f_zed_kx_ky_ri[:,:,:,0] = f_zed_kx_ky

            else:
                alt_slow_eval = True

        if alt_slow_eval:
           # Evaluate in real space first and then Fourier transform
           f_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity=quantity, time_idx=time_idx, time_val=time_val, time_avg=time_avg, remove_zonal=remove_zonal, only_zonal=only_zonal)
           f_zed_x = np.sum(f_zed_x_y, axis=2)*2/len(y)
           for i_zed in range(len(zed)):
               f_kx, kx = get_fft_k(f_zed_x[i_zed], x)
               if i_zed == 0:
                   f_zed_kx = np.zeros((len(zed), len(kx)), dtype='complex')

               f_zed_kx[i_zed] = f_kx

           ## CHECK NORMALISATION
           #dx = x[1]-x[0]
           #integral_x = np.sum(reynolds_stress_zed_x[0])*dx
           #print("\nIntegral_x: %e, kx=0: %e" % (integral_x, reynolds_stress_zed_kx[0,0]*(x[-1]-x[0])))

           return f_zed_kx[:,:,None]*(1j*kx[None,:,None])**kx_order, zed, kx, ky, time_eval
    
        time_eval = time_all[time_idx]
         
        if time_avg is not None:
            f_zed_kx_ky_ri = np.mean(f_zed_kx_ky_ri, axis=0)

        f_zed_kx_ky = f_zed_kx_ky_ri[:,:,:,0] + 1j*f_zed_kx_ky_ri[:,:,:,1]

        # x-derivatives
        f_zed_kx_ky = f_zed_kx_ky * (1j*kx[None,:,None])**kx_order
        #f_zed_kx_ky = f_zed_kx_ky * (1j*kx[None,:,None]/(kx[1]-kx[0]))**kx_order

        # y-derivatives
        f_zed_kx_ky = f_zed_kx_ky * (1j*ky[None,None,:])**ky_order
        #f_zed_kx_ky = f_zed_kx_ky * (1j*ky[None,None,:]/(ky[1]-ky[0]))**ky_order

        # Filter zonal if requested
        if remove_zonal:
            f_zed_kx_ky[:,:,0]= 0
        if only_zonal:
            f_zed_kx_ky[:,:,1:]= 0

        return f_zed_kx_ky, zed, kx, ky, time_eval


    def get_quantity_kx_ky(self, quantity, zed_val = None, zed_idx=None, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, mult_zed=None, par_der_order=0, mean_delt_zed=None, alt_slow_eval=False, sort_kx=False):

        if remove_zonal and only_zonal:
            print("WARNING! Both only_zonal and remove_zonal were set to True, will thus return f_x_y = 0.")

        if quantity=="phi" and self.code == "GS2":
            f_kx_ky_ri = np.transpose( self.ncdata.variables['phi_igomega_by_mode'][time_idx] , axes=[1,0,2] )
            zed       = self.ncdata.variables['theta'][:]
            f_zed_kx_ky_ri = np.tile(f_kx_ky_ri[None,:,:,:].T, len(zed)).T
            f_zed_kx_ky = f_zed_kx_ky_ri[:,:,:,0] + 1j*f_zed_kx_ky_ri[:,:,:,1]
            ky        = self.ncdata.variables['ky'][:]
            kx        = self.ncdata.variables['kx'][:]
            time_eval = self.ncdata.variables['t'][time_idx]
        else:
            f_zed_kx_ky, zed, kx, ky, time_eval = self.get_quantity_zed_kx_ky(quantity, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, alt_slow_eval=alt_slow_eval)

            #elif self.code == "GX":
                #phi2_vs_kxky = np.transpose(self.ncdata['Spectra']['Pkxkyst'][:,0,:,:], axes=(0,2,1))
                #print("Warning! Not really plotting phi spectrum due to GX diagnostics.")
                #kperp = np.sqrt(kx[None,:,None]**2 + ky[None,None,:]**2)
                #Gamma0fac_vs_kxky = specialfunc.iv(0, kperp/2) * np.exp(-kperp/2)
                #phi2_vs_kxky = phi2_vs_kxky / (1-Gamma0fac_vs_kxky)
 
                #f_kx_ky_ri = np.sqrt(phi2_vs_kxky)

        # if zed_val is not None, find zed_idx matching zed_val most closely
        if zed_val is not None:
            zed_idx = np.argmin( np.abs(zed - zed_val) )

        zed_weight = self.get_zed_weight(mult_zed=mult_zed, zed=zed)

        # Take zed derivatives if needed
        for i in range(par_der_order):
            gradpar  = self.ncdata.variables['gradpar'][:]
            # Use periodicity
            _, _, _, _, gds22, bmag = self.get_FLR()
            f_zed_kx_ky = np.gradient(f_zed_kx_ky*gradpar[:,None,None], zed, axis=0)
 
        try:
            tmp = f_kx_ky[0,0]
        except:
            f_zed_kx_ky = f_zed_kx_ky * zed_weight[:,None,None]

            if mean_delt_zed is not None:
                dl_over_B_avg = self.dl_over_B_avg()
                mean_f_kx_ky = np.sum(dl_over_B_avg[:,None,None]*f_zed_kx_ky, axis=0)
                mean_f_zed_kx_ky = np.zeros_like(f_zed_kx_ky)
                for i_zed in range(len(zed)):
                    mean_f_zed_kx_ky[i_zed] = mean_f_kx_ky

                if mean_delt_zed == "mean":
                    f_zed_kx_ky = mean_f_zed_kx_ky
                elif mean_delt_zed == "delt":
                    f_zed_kx_ky = f_zed_kx_ky - mean_f_zed_kx_ky

            if zed_idx is None:
                f_kx_ky = np.sum(f_zed_kx_ky, axis=0)
            else:
                f_kx_ky = f_zed_kx_ky[zed_idx]

        # Sort kx if required
        if sort_kx:
            idx_sort = np.argsort(kx)
            kx = kx[idx_sort]
            f_kx_ky = f_kx_ky[idx_sort]

        return f_kx_ky, kx, ky, time_eval


    def get_quantity_zed_x_y(self, quantity, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, kxmin_filter=np.infty, kymin_filter=np.infty, kxmax_filter=-1, kymax_filter=-1, abs_squared=False, quantity_mult=None):

        if remove_zonal and only_zonal:
            print("WARNING! Both only_zonal and remove_zonal were set to True, will thus return f_x_y = 0.")

        kx, ky, zed = self.get_kx_ky_zed()
        time_all    = self.get_time_array(GX_big=True)
        dl_over_B_avg = self.dl_over_B_avg()

        if time_val is not None:
            time_idx = np.argmin( np.abs(time_all-time_val) )

        time_eval = time_all[time_idx]

        if time_avg is not None:
            time_min  = max(0,            time_eval-time_avg/2)
            time_max  = min(time_all[-1], time_eval+time_avg/2)
            time_idx_min = np.argmin( np.abs(time_all-time_min) )
            time_idx_max = np.argmin( np.abs(time_all-time_max) )
            time_idx = np.arange(time_idx_min,time_idx_max+1)
            time_avg_vals = time_all[time_idx]
            dt_avg_vals   = np.gradient(time_avg_vals)
            
        if quantity=="phi":
            # phi_vs_t(t, tube, zed, theta0, ky, ri)
            if self.code == "stella":
                f_zed_kx_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
            elif self.code == "GS2":
                f_kx_ky_ri = np.transpose( self.ncdata.variables['phi_igomega_by_mode'][time_idx] , axes=[1,0,2] )
                f_zed_kx_ky_ri = np.tile(f_kx_ky_ri[None,:,:,:].T, len(zed)).T
            elif self.code == "GX":
                if self.GX_old_version:
                    print("WARNING! You are loading Phi_z in GX, which had issues in early code versions.")
                    f_zed_kx_ky_ri = np.transpose( self.ncdata['Special']['Phi_z'], axes=[2,1,0,3] )
                else:
                    f_zed_kx_ky_ri = np.transpose( self.ncdata_big['Diagnostics']['Phi'][time_idx] , axes=[2,1,0,3] )

        elif quantity=="(1-Gamma0)phi":
            kperp2_zed_kx_ky = self.ncdata.variables['kperp2'][:][:,species_idx,:,:] 
            Gamma0 = specialfunc.iv(0, kperp2_zed_kx_ky/2) * np.exp(-kperp2_zed_kx_ky/2)
            f_zed_kx_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]*(1-Gamma0)[:,:,:,None]
 
        elif quantity[:3] == "RH_":
            if quantity=="RH_phi_I" or quantity=="RH_phi" or quantity == "RHnon_phi":
                f_zed_kx_ri = self.ncdata.variables['RH_phi_I'][time_idx,species_idx,0,:,:,:]
            elif quantity == "RH_fluxes_collisional":
                f_zed_kx_ri = self.ncdata.variables[quantity][time_idx,species_idx,0,:,:,:]
#            elif quantity=="RH_phi":
#                RH_phi_I_zed_kx_ri = self.ncdata.variables["RH_phi_I"][time_idx,species_idx,0,:,:,:]
#                RH_inertia_zed_kx = self.ncdata.variables['RH_inertia'][species_idx,0,:,:,0]
#                f_zed_kx_ri = RH_phi_I_zed_kx_ri / RH_inertia_zed_kx[:,:,None]
            else:
                # Quantity must be one of the nonlinear RH fluxes
                if species_idx == "sum":
                    try:
                        nspecies = len(self.ncdata.dimensions['species'])
                    except:
                        nspecies = 1
                    RH_flux_zed_kx_ky_ri = self.ncdata.variables[quantity][time_idx,0,0,:,:,:,:]
                    for i_spec in np.arange(nspecies-1):
                        RH_flux_zed_kx_ky_ri += self.ncdata.variables[quantity][time_idx,i_spec+1,0,:,:,:,:]

                else:
                    RH_flux_zed_kx_ky_ri = self.ncdata.variables[quantity][time_idx,species_idx,0,:,:,:,:]
                # Sum over ky
                if time_avg is None:
                    axis_ky = 2
                else:
                    axis_ky = 3
                f_zed_kx_ri = np.sum(RH_flux_zed_kx_ky_ri, axis=axis_ky)


            ky = self.ncdata['ky'][:]
        
            if time_avg is None:
                f_zed_kx_ky_ri = np.zeros( (np.shape(f_zed_kx_ri)[0], np.shape(f_zed_kx_ri)[1], len(ky), 2) )
                f_zed_kx_ky_ri[:,:,0,:] = f_zed_kx_ri

                if quantity == "RH_phi" or quantity == "RHnon_phi":
                    # Divide by RH inertia
                    RH_inertia_zed_kx = self.ncdata.variables['RH_inertia'][species_idx,0,:,:,0]
                    dl_over_B_avg = self.dl_over_B_avg()
                    RH_inertia_kx = np.sum(dl_over_B_avg[:,None]*RH_inertia_zed_kx, axis=0)
                    f_zed_kx_ky_ri = f_zed_kx_ky_ri / RH_inertia_kx[None,:,None,None]

                    if quantity == "RHnon_phi":
                        phi_zed_kx_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
                        phi_zed_kx_ky_ri[:,:,1:,:] = 0
                        f_zed_kx_ky_ri = phi_zed_kx_ky_ri - f_zed_kx_ky_ri

            else:
                f_zed_kx_ky_ri = np.zeros( (np.shape(f_zed_kx_ri)[0], np.shape(f_zed_kx_ri)[1], np.shape(f_zed_kx_ri)[2], len(ky), 2) )
                f_zed_kx_ky_ri[:,:,:,0,:] = f_zed_kx_ri

                if quantity == "RH_phi":
                    # Divide by RH inertia
                    RH_inertia_zed_kx = self.ncdata.variables['RH_inertia'][species_idx,0,:,:,0]
                    dl_over_B_avg = self.dl_over_B_avg()
                    RH_inertia_kx = np.sum(dl_over_B_avg[:,None]*RH_inertia_zed_kx, axis=0)
                    f_zed_kx_ky_ri = f_zed_kx_ky_ri / RH_inertia_kx[None,None,:,None,None]


        elif quantity=="density":
            # density(t, species, tube, zed, kx, ky, ri)
            f_zed_kx_ky_ri = self.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
        elif quantity=="upar":
            # upar(t, species, tube, zed, kx, ky, ri)
            f_zed_kx_ky_ri = self.ncdata.variables['upar'][time_idx,species_idx,0,:,:,:,:]

        elif quantity=="unonPS":
            costheta = self.get_zed_weight("cos", zed) / self.get_zed_weight(None, zed)

            # upar(t, species, tube, zed, kx, ky, ri)
            upar_zed_kx_ky_ri = self.ncdata.variables['upar'][time_idx,species_idx,0,:,:,:,:]

            # phi_vs_t(t, tube, zed, theta0, ky, ri)
            phi_zed_kx_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
            dxphi_zed_kx_ky_ri = np.zeros_like(phi_zed_kx_ky_ri)
            dxphi_zed_kx_ky_ri[:,:,:,0] = -kx[None,:,None]  * phi_zed_kx_ky_ri[:,:,:,1]
            dxphi_zed_kx_ky_ri[:,:,:,1] =  kx[None,:,None]  * phi_zed_kx_ky_ri[:,:,:,0]

            qinp   = self.safety_factor
            eps    = 0.5/2.778

            #FACTOR OF TWO
            f_zed_kx_ky_ri =  (upar_zed_kx_ky_ri + qinp*dxphi_zed_kx_ky_ri*costheta[:,None,None,None] + 0.8*qinp*np.sqrt(eps)*dxphi_zed_kx_ky_ri)*costheta[:,None,None,None]
            #f_zed_kx_ky_ri =  (upar_zed_kx_ky_ri + 2*qinp*dxphi_zed_kx_ky_ri*costheta[:,None,None,None] + 1.6*qinp*np.sqrt(eps)*dxphi_zed_kx_ky_ri)*costheta[:,None,None,None]


        elif quantity=="unonRH":
            costheta = self.get_zed_weight("cos", zed) / self.get_zed_weight(None, zed)

            # upar(t, species, tube, zed, kx, ky, ri)
            upar_zed_kx_ky_ri = self.ncdata.variables['upar'][time_idx,species_idx,0,:,:,:,:]

            # phi_vs_t(t, tube, zed, theta0, ky, ri)
            phi_zed_kx_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
            dxphi_zed_kx_ky_ri = np.zeros_like(phi_zed_kx_ky_ri)
            dxphi_zed_kx_ky_ri[:,:,:,0] = -kx[None,:,None]  * phi_zed_kx_ky_ri[:,:,:,1]
            dxphi_zed_kx_ky_ri[:,:,:,1] =  kx[None,:,None]  * phi_zed_kx_ky_ri[:,:,:,0]

            qinp   = self.safety_factor
            eps    = 0.5/2.778

            #FACTOR OF TWO
            f_zed_kx_ky_ri = upar_zed_kx_ky_ri + qinp*dxphi_zed_kx_ky_ri*costheta[:,None,None,None] + 0.8*qinp*np.sqrt(eps)*dxphi_zed_kx_ky_ri
            #f_zed_kx_ky_ri = upar_zed_kx_ky_ri + 2*qinp*dxphi_zed_kx_ky_ri*costheta[:,None,None,None] + 1.6*qinp*np.sqrt(eps)*dxphi_zed_kx_ky_ri

        elif quantity=="upar-over-B":
            _, _, _, _, _, bmag = self.get_FLR()
            f_zed_kx_ky_ri = self.ncdata.variables['upar'][time_idx,species_idx,0,:,:,:,:]/bmag[:,None,None,None]
        elif quantity=="pressure_par": #(xpa^2)
            P_zed_kx_ky_ri = self.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
            try:
                Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
            except:
                Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
            f_zed_kx_ky_ri = P_zed_kx_ky_ri-0.5*Pprp_zed_kx_ky_ri
        elif quantity=="pressure_perp": #(xperp^2)
            # pressure_perp(t, species, tube, zed, kx, ky, ri)
            try:
                f_zed_kx_ky_ri = self.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
            except:
                f_zed_kx_ky_ri = self.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
        elif quantity=="pressure": #(xpa^2+xprp^2/2)
            try:
                f_zed_kx_ky_ri = self.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
            except:
                n_zed_kx_ky_ri = self.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:] #1
                T_zed_kx_ky_ri = self.ncdata.variables['temperature'][time_idx,species_idx,0,:,:,:,:] #(xpa^2+xprp^2-3/2)/(3/2)
                Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:] # xprp^2
                f_zed_kx_ky_ri = 1.5*(T_zed_kx_ky_ri+n_zed_kx_ky_ri) - 0.5*Pprp_zed_kx_ky_ri

        elif quantity=="pressure-phi":
            f_zed_kx_ky_ri_1 = self.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
            f_zed_kx_ky_ri_2 = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
            f_zed_kx_ky_ri = f_zed_kx_ky_ri_1 + f_zed_kx_ky_ri_2
        elif quantity=="pressure_perp-phi":
            try:
                f_zed_kx_ky_ri_1 = self.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
            except:
                f_zed_kx_ky_ri_1 = self.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
            f_zed_kx_ky_ri_2 = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
            f_zed_kx_ky_ri = f_zed_kx_ky_ri_1 + f_zed_kx_ky_ri_2
        elif quantity=="dtP_GAM":
            fP_zed_kx_ky_ri = self.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
            try:
                fchi_zed_kx_ky_ri = self.ncdata.variables['chi'][time_idx,species_idx,0,:,:,:,:]
            except:
                print("chi not found in NETCDF! Using pressure instead.")
                fchi_zed_kx_ky_ri = fP_zed_kx_ky_ri
            fphi_zed_kx_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx,0,:,:,:,:]
            tau = 1
            f_zed_kx_ky_ri = fchi_zed_kx_ky_ri+fP_zed_kx_ky_ri/tau + fphi_zed_kx_ky_ri*(7/4+1/tau)
        elif quantity=="chi":
            try:
                f_zed_kx_ky_ri = self.ncdata.variables['chi'][time_idx,species_idx,0,:,:,:,:]
            except:
                print("chi not found in NETCDF! Using pressure instead.")
                f_zed_kx_ky_ri = self.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
        elif quantity=="temperature": #(xpa^2+xprp^2-1.5)/1.5
            if self.code == "stella":
                # temperature(t, species, tube, zed, kx, ky, ri)
                f_zed_kx_ky_ri = self.ncdata.variables['temperature'][time_idx,species_idx,0,:,:,:,:]
            elif self.code == "GX":
                Tpar_zed_kx_ky_ri = np.transpose( self.ncdata_big['Diagnostics']['Tpar'][time_idx, species_idx] , axes=[2,1,0,3] )
                Tprp_zed_kx_ky_ri = np.transpose( self.ncdata_big['Diagnostics']['Tperp'][time_idx, species_idx] , axes=[2,1,0,3] )
                f_zed_kx_ky_ri = Tpar_zed_kx_ky_ri + Tprp_zed_kx_ky_ri
        elif quantity=="temperature_par": #(xpa^2-1/2)
            # temperature(t, species, tube, zed, kx, ky, ri)
            P_zed_kx_ky_ri    = self.ncdata.variables['pressure'][time_idx,species_idx,0,:,:,:,:]
            try:
                Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
            except:
                Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
            n_zed_kx_ky_ri    = self.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
            f_zed_kx_ky_ri = P_zed_kx_ky_ri - 0.5*Pprp_zed_kx_ky_ri - 0.5*n_zed_kx_ky_ri
        elif quantity=="temperature_perp": #(xprp^2-1)
            # temperature(t, species, tube, zed, kx, ky, ri)
            try:
                Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_perp'][time_idx,species_idx,0,:,:,:,:]
            except:
                Pprp_zed_kx_ky_ri = self.ncdata.variables['pressure_prp'][time_idx,species_idx,0,:,:,:,:]
            n_zed_kx_ky_ri    = self.ncdata.variables['density'][time_idx,species_idx,0,:,:,:,:]
            f_zed_kx_ky_ri = Pprp_zed_kx_ky_ri - n_zed_kx_ky_ri
        elif quantity=="qpar":
            # qpar(t, species, tube, zed, kx, ky, ri)
            f_zed_kx_ky_ri = self.ncdata.variables['qpar'][time_idx,species_idx,0,:,:,:,:]
        elif quantity=="qperp":
            # qperp(t, species, tube, zed, kx, ky, ri)
            f_zed_kx_ky_ri = self.ncdata.variables['qperp'][time_idx,species_idx,0,:,:,:,:]
        elif quantity=="qpar-over-B":
            _, _, _, _, _, bmag = self.get_FLR()
            # qpar(t, species, tube, zed, kx, ky, ri)
            f_zed_kx_ky_ri = self.ncdata.variables['qpar'][time_idx,species_idx,0,:,:,:,:]/bmag[:,None,None,None]
        elif quantity=="vflx_pol_phi_slab_kxz":
            # vflx_pol_phi_slab_kxz(t, species, tube, zed, kx, ri)
            f_zed_kx_ky_ri = np.zeros(shape=(len(zed), len(kx), len(ky), 2))
            f_zed_kx_ky_ri[:,:,0,:] = self.ncdata.variables['vflx_pol_phi_slab_kxz'][time_idx,species_idx,0]
        elif quantity=="vflx_pol_phi_shear_kxz":
            f_zed_kx_ky_ri = np.zeros(shape=(len(zed), len(kx), len(ky), 2))
            f_zed_kx_ky_ri[:,:,0,:] = self.ncdata.variables['vflx_pol_phi_shear_kxz'][time_idx,species_idx,0]
        elif quantity=="vflx_pol_Tperp_slab_kxz":
            f_zed_kx_ky_ri = np.zeros(shape=(len(zed), len(kx), len(ky), 2))
            f_zed_kx_ky_ri[:,:,0,:] = self.ncdata.variables['vflx_pol_Tperp_slab_kxz'][time_idx,species_idx,0]
        elif quantity=="vflx_pol_Tperp_shear_kxz":
            f_zed_kx_ky_ri = np.zeros(shape=(len(zed), len(kx), len(ky), 2))
            f_zed_kx_ky_ri[:,:,0,:] = self.ncdata.variables['vflx_pol_Tperp_shear_kxz'][time_idx,species_idx,0]

        else:
            ### Composite quantities that can be evaluated directly in real space
            if quantity=="deltaphi_2":
                phi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=remove_zonal, only_zonal=only_zonal, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                phi_mean = np.mean(phi, axis=0)
                for i_zed in range(len(zed)):
                    delta_phi = phi[i_zed] - phi_mean
                f_zed_x_y = delta_phi**2

            elif quantity=="deltaphi":
                phi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=remove_zonal, only_zonal=only_zonal, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                #phi[:] = np.mean(phi, axis=0)
                #delta_phi = phi
                zed_weight = self.dl_over_B_avg()
                delta_phi = phi - np.sum(phi*zed_weight[:,None,None], axis=0)
                f_zed_x_y = delta_phi

            elif quantity=="dyphi-qpar-over-B":
                _, _, _, _, _, bmag = self.get_FLR()
                # qpar(t, species, tube, zed, kx, ky, ri)
                dyphi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                qpar_over_B , _, _, _, _    = self.get_quantity_zed_x_y(quantity="qpar-over-B", time_idx=time_idx, species_idx=species_idx, time_val=time_val, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dyphi*qpar_over_B

            elif quantity=="dyphi-upar":
                dyphi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                upar, _, _, _, _ = self.get_quantity_zed_x_y(quantity="upar", time_idx=time_idx, species_idx=species_idx, time_val=time_val, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dyphi*upar


            elif quantity=="dyphi-dxphi":
                dyphi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dxphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dyphi*dxphi

            elif quantity=="dyphi-dyphi":
                dyphi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dyphi**2

            elif quantity=="dyPprp-dxphi":
                dyPprp, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="pressure_perp", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dxphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dyPprp*dxphi

            elif quantity=="dyphiPprp-dxphi":
                dyphi, zed, x, y, time_eval  = self.get_quantity_zed_x_y(quantity="phi",           time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dyPprp, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="pressure_perp", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dxphi, _, _, _, _            = self.get_quantity_zed_x_y(quantity="phi",           time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = (dyphi+dyPprp)*dxphi

            elif quantity=="dyphiPprp-dyphi":
                dyphi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dyPprp, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="pressure_perp", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = (dyphi+dyPprp)*dyphi

            elif quantity=="dyPprp-dyphi":
                dyPprp, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="pressure_perp", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dyphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dyPprp*dyphi

            elif quantity=="dyphi-P":
                pressure, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=0, time_avg=time_avg, nx=nx, ny=ny)
                dyphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = pressure*dyphi

            elif quantity=="dyphi-chi":
                chi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="chi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=0, time_avg=time_avg, nx=nx, ny=ny)
                dyphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = chi*dyphi

            elif quantity=="dxTZ_dyphi2":
                dxTZ, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="temperature", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=False, only_zonal=True, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dyphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dxTZ*dyphi**2

            elif quantity=="dxTZtot_dyphi2":
                tprim = self.ncdata.variables['tprim'][0]
                dxTZ, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="temperature", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=False, only_zonal=True, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dyphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = (dxTZ-tprim)*dyphi**2

            elif quantity=="dxPZ_dyphi2":
                dxPZ, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=False, only_zonal=True, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dyphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dxPZ*dyphi**2

            elif quantity=="dxdeltaphiZ_dyphi_dyP":
                dxdeltaphiZ, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="deltaphi", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=False, only_zonal=True, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dyphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi",      time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dyP, _, _, _, _   = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dxdeltaphiZ*dyphi*dyP

            elif quantity=="NL_heat_flux_transp":
                dyphi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dxphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
                dyP, _, _, _, _ = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dxP, _, _, _, _ = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
                f_zed_x_y = -dyphi*(dxphi*dyP - dyphi*dxP)

            elif quantity=="vMy_heat_flux_transp":
                dyphi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dy2phi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=2, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
                P, _, _, _, _ = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
                dy2P, _, _, _, _ = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=2, time_avg=time_avg, nx=nx, ny=ny)
                dychi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="chi",      time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
                tau = 1
                f_zed_x_y = -(dy2phi+dy2P)*P/tau - dyphi*(dychi + 7/4*dyphi)

            elif quantity=="vMx_heat_flux_transp":
                dyphi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dxphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
                dydxphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
                P, _, _, _, _ = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
                dydxP, _, _, _, _ = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dxchi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="chi",      time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
                tau = 1
                f_zed_x_y = -(dydxphi+dydxP)*P/tau - dyphi*(dxchi + 7/4*dxphi)

            elif quantity=="kappa_transp":
                tprim = self.ncdata.variables['tprim'][0]
                #fprim = self.ncdata.variables['fprim'][0]
                dyphi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                # Recall dxT0 < 0 as temperature is higher in core
                f_zed_x_y = dyphi**2 * (-tprim)

            elif quantity=="dyphi-dyupar-over-B":
                dyphi,  zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dyupar, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="upar-over-B", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dyphi*dyupar

            elif quantity=="d2yphi-dypressure":
                d2yphi,  zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=2, time_avg=time_avg, nx=nx, ny=ny)
                dypres, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = d2yphi*dypres

            elif quantity=="d2yphi-dxpressure":
                d2yphi,  zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=2, time_avg=time_avg, nx=nx, ny=ny)
                dxpres, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, remove_zonal=True)
                f_zed_x_y = d2yphi*dxpres

            elif quantity=="dyphi2":
                dyphi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dyphi**2

            elif quantity=="dxphiZ_dyphi_dyP":
                dxphiZ, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=False, only_zonal=True, kx_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dyphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                dyP,   _, _, _, _ = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny)
                f_zed_x_y = dxphiZ*dyphi*dyP

            elif quantity=="dyphi-T":
                temp, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="temperature", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=0, nx=nx, ny=ny)
                dyphi, _, _, _, _ = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, nx=nx, ny=ny)
                f_zed_x_y = dyphi*temp

            elif quantity=="dyphi-dyP":
                dyphi, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="phi", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                dyP  , zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity="pressure", time_idx=time_idx, species_idx=species_idx, time_val=time_val, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                f_zed_x_y = dyphi*dyP

            elif quantity[:8] == "Reynolds":
                _, _, _, gds21, gds22, bmag = self.get_FLR()

                dxphi_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("phi",           time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)
                dyphi_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("phi",           time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)
                RS_factor_zed_x_y_nablax2 = dxphi_zed_x_y*(gds22/bmag**2)[:,None,None]
                RS_factor_zed_x_y_nablaxy = dyphi_zed_x_y*(gds21/bmag**2)[:,None,None]
                f_zed_x_y = np.zeros_like(RS_factor_zed_x_y_nablax2)

                if quantity in ["Reynolds", "Reynolds_nablax2", "Reynolds_Pprp_nablax2"]:
                    dyPprp_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)
                    f_zed_x_y += RS_factor_zed_x_y_nablax2 * dyPprp_zed_x_y
                if quantity in ["Reynolds", "Reynolds_nablaxy", "Reynolds_Pprp_nablaxy"]:
                    dyPprp_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)
                    f_zed_x_y += RS_factor_zed_x_y_nablaxy * dyPprp_zed_x_y
                if quantity in ["Reynolds", "Reynolds_nablax2", "Reynolds_phi_nablax2"]:
                    f_zed_x_y += RS_factor_zed_x_y_nablax2 * dyphi_zed_x_y
                if quantity in ["Reynolds", "Reynolds_nablaxy", "Reynolds_phi_nablaxy"]:
                    f_zed_x_y += RS_factor_zed_x_y_nablaxy * dyphi_zed_x_y

            elif quantity == "dEZ_Reynolds":
                #reynolds_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("Reynolds",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                #dx2phi_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("phi", kx_order=2, time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)
                #f_zed_x_y = dx2phi_zed_x_y*reynolds_zed_x_y
                dx_reynolds_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("Reynolds", kx_order=1, time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                dxphi_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("phi", kx_order=1, time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)
                f_zed_x_y = dxphi_zed_x_y*dx_reynolds_zed_x_y

#                print(np.mean(reynolds_zed_x_y, axis=(0,2)))
#                print(np.mean(dx2phi_zed_x_y, axis=(0,2)))
#                print(np.mean(dx2phi_zed_x_y, axis=(0,2))*np.mean(reynolds_zed_x_y, axis=(0,2)))
#                print(np.mean(np.mean(dx2phi_zed_x_y,axis=0)*np.mean(reynolds_zed_x_y,axis=0), axis=1))
#                dx2phi_x_y = np.mean(dx2phi_zed_x_y, axis=0)
#                reynolds_x_y = np.mean(reynolds_zed_x_y, axis=0)
#                print(np.mean(dx2phi_x_y*reynolds_x_y, axis=1))
#                print("###############################")


            elif quantity == "dEZ_vdriftx":
                phiZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("phi", kx_order=0, time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)
                dxPZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("pressure", kx_order=1, time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)
                vdriftx = self.get_zed_weight("vdriftx")

                f_zed_x_y = phiZ_zed_x_y*dxPZ_zed_x_y*vdriftx[:,None,None]

            elif quantity == "qpar_mom_transport":

                dyphi_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("phi",           time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)
                qparNZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("qpar",           time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)

                f_zed_x_y = qparNZ_zed_x_y * dyphi_zed_x_y # To be multiplied by dx(qparZ) to get energy time derivative contribution

            elif quantity == "dEZ_qpar_mom_transport":
                qpar_mom_transport_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("qpar_mom_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                dxqparZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("qpar",           time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)
                f_zed_x_y = -2*dxqparZ_zed_x_y * qpar_mom_transport_zed_x_y

            elif quantity == "par_mom_transport":

                dyphi_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("phi",           time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)
                uparNZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("upar",           time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)

                f_zed_x_y = uparNZ_zed_x_y * dyphi_zed_x_y # To be multiplied by dx(uparZ) to get energy time derivative contribution

            elif quantity == "dEZ_par_mom_transport":
                #par_mom_transport_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("par_mom_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                #dxuparZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("upar",           time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)
                #f_zed_x_y = 2*dxuparZ_zed_x_y * par_mom_transport_zed_x_y

                dx_par_mom_transport_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("par_mom_transport",time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                uparZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("upar",           time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)
                f_zed_x_y = -2*uparZ_zed_x_y * dx_par_mom_transport_zed_x_y

            elif quantity == "duZ_par_mom_transport":
                par_mom_transport_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("par_mom_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                dxvEZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("phi",           time_idx=time_idx, species_idx=species_idx, kx_order=2, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)
                f_zed_x_y = -2*dxvEZ_zed_x_y * par_mom_transport_zed_x_y

            elif quantity == "temperature_transport":

                dyphi_zed_x_y, zed, x, y, time_eval      = self.get_quantity_zed_x_y("phi",      time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)
                tempNZ_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("temperature", time_idx=time_idx, species_idx=species_idx, ky_order=0, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)

                f_zed_x_y = tempNZ_zed_x_y * dyphi_zed_x_y # To be multiplied by dx(tempZ) to get energy time derivative contribution

            elif quantity == "dEZ_mean_temperature_transport":
                temperature_transp_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("temperature_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                dxtempZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("temperature",       time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)

                dl_over_B_avg = self.dl_over_B_avg()
                mean_dxtempZ_x_y = np.sum(dl_over_B_avg[:,None,None]*dxtempZ_zed_x_y, axis=0)
                mean_dxtempZ_zed_x_y = np.zeros_like(dxtempZ_zed_x_y)
                for i_zed in range(len(zed)):
                    mean_dxtempZ_zed_x_y[i_zed] = mean_dxtempZ_x_y
                f_zed_x_y = -4/3*mean_dxtempZ_zed_x_y*temperature_transp_zed_x_y

            elif quantity == "dEZ_delt_temperature_transport":
                temperature_transp_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("temperature_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                dxtempZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("temperature",       time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)

                dl_over_B_avg = self.dl_over_B_avg()
                mean_dxtempZ_x_y = np.sum(dl_over_B_avg[:,None,None]*dxtempZ_zed_x_y, axis=0)
                mean_dxtempZ_zed_x_y = np.zeros_like(dxtempZ_zed_x_y)
                for i_zed in range(len(zed)):
                    mean_dxtempZ_zed_x_y[i_zed] = mean_dxtempZ_x_y
                delt_dxtempZ_zed_x_y = dxtempZ_zed_x_y - mean_dxtempZ_zed_x_y
                f_zed_x_y = -4/3*delt_dxtempZ_zed_x_y*temperature_transp_zed_x_y

            elif quantity == "pressure_transport":

                dyphi_zed_x_y, zed, x, y, time_eval      = self.get_quantity_zed_x_y("phi",      time_idx=time_idx, species_idx=species_idx, ky_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)
                presNZ_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("pressure", time_idx=time_idx, species_idx=species_idx, ky_order=0, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, remove_zonal=True)

                f_zed_x_y = presNZ_zed_x_y * dyphi_zed_x_y # To be multiplied by dx(presZ) to get energy time derivative contribution

            elif quantity == "dEZ_mean_pressure_transport":
                pressure_transp_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("pressure_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                dxpresZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("pressure",       time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)

                dl_over_B_avg = self.dl_over_B_avg()
                mean_dxpresZ_x_y = np.sum(dl_over_B_avg[:,None,None]*dxpresZ_zed_x_y, axis=0)
                mean_dxpresZ_zed_x_y = np.zeros_like(dxpresZ_zed_x_y)
                for i_zed in range(len(zed)):
                    mean_dxpresZ_zed_x_y[i_zed] = mean_dxpresZ_x_y
                f_zed_x_y = -4/3*mean_dxpresZ_zed_x_y*pressure_transp_zed_x_y

            elif quantity == "dEZ_delt_pressure_transport":
                pressure_transp_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("pressure_transport",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                dxpresZ_zed_x_y, zed, x, y, time_eval  = self.get_quantity_zed_x_y("pressure",       time_idx=time_idx, species_idx=species_idx, kx_order=1, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)

                dl_over_B_avg = self.dl_over_B_avg()
                mean_dxpresZ_x_y = np.sum(dl_over_B_avg[:,None,None]*dxpresZ_zed_x_y, axis=0)
                mean_dxpresZ_zed_x_y = np.zeros_like(dxpresZ_zed_x_y)
                for i_zed in range(len(zed)):
                    mean_dxpresZ_zed_x_y[i_zed] = mean_dxpresZ_x_y
                delt_dxpresZ_zed_x_y = dxpresZ_zed_x_y - mean_dxpresZ_zed_x_y
                f_zed_x_y = -4/3*delt_dxpresZ_zed_x_y*pressure_transp_zed_x_y

            elif quantity == "P_RH_coll":
                #RH_flux_coll_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_collisional",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0)
                #dx_phi_RH_zed_x_y, zed, x, y, time_eval    = self.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True, kx_order=1)
                #f_zed_x_y = RH_flux_coll_zed_x_y*dx_phi_RH_zed_x_y

                dx_phi_RH_zed_x_y, zed, x, y, time_eval    = self.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True, kx_order=1)
                try:
                    RH_flux_coll_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_collisional",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0)
                    f_zed_x_y = RH_flux_coll_zed_x_y*dx_phi_RH_zed_x_y
                except:
                    f_zed_x_y = np.zeros_like(dx_phi_RH_zed_x_y)

            elif quantity == "Pi_RH_even":
                try:
                    Pi_RH_phi_even_passing_zed_x_y, zed, x, y, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_even_passing",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=-1)
                    Pi_RH_phi_even_trapped_zed_x_y, _, _, _, _   = self.get_quantity_zed_x_y("RH_fluxes_phi_even_trapped",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=-1)
                    f_zed_x_y = Pi_RH_phi_even_passing_zed_x_y + Pi_RH_phi_even_trapped_zed_x_y
                except:
                    f_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_even",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=-1)


            elif quantity == "Pi_RH_odd":
                try:
                    Pi_RH_phi_odd_passing_zed_x_y, zed, x, y, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_odd_passing",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=-1)
                    Pi_RH_phi_odd_trapped_zed_x_y, _, _, _, _   = self.get_quantity_zed_x_y("RH_fluxes_phi_odd_trapped",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=-1)
                    f_zed_x_y = Pi_RH_phi_odd_passing_zed_x_y + Pi_RH_phi_odd_trapped_zed_x_y
                except:
                    f_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_odd",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=-1)

            elif quantity == "Pi_RH_NL":
                Pi_RH_even_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("Pi_RH_even", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter) 
                Pi_RH_odd_zed_x_y,    _, _, _,        _  = self.get_quantity_zed_x_y("Pi_RH_odd",  time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter) 
                f_zed_x_y = Pi_RH_even_zed_x_y + Pi_RH_odd_zed_x_y

            elif quantity == "P_RH_even":
                try:
                    #dx_RH_flux_phi_even_passing_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_even_passing",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=1)
                    #dx_RH_flux_phi_even_trapped_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_even_trapped",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=1)
                    #dx_RH_flux_phi_even_zed_x_y = dx_RH_flux_phi_even_passing_zed_x_y + dx_RH_flux_phi_even_trapped_zed_x_y
                    RH_flux_phi_even_passing_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_even_passing",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0)
                    RH_flux_phi_even_trapped_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_even_trapped",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0)
                    RH_flux_phi_even_zed_x_y = RH_flux_phi_even_passing_zed_x_y + RH_flux_phi_even_trapped_zed_x_y
                except:
                    #dx_RH_flux_phi_even_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_even",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=1)
                    RH_flux_phi_even_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_even",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0)


                #phi_RH_zed_x_y, zed, x, y, time_eval    = self.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)
                dx_phi_RH_zed_x_y, zed, x, y, time_eval    = self.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True, kx_order=1)

                #f_zed_x_y = dx_RH_flux_phi_even_zed_x_y*phi_RH_zed_x_y
                f_zed_x_y = RH_flux_phi_even_zed_x_y*dx_phi_RH_zed_x_y

            elif quantity == "P_RH_odd":

                try:
                    #dx_RH_flux_phi_odd_passing_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_odd_passing",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=1)
                    #dx_RH_flux_phi_odd_trapped_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_odd_trapped",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=1)
                    #dx_RH_flux_phi_odd_zed_x_y = dx_RH_flux_phi_odd_passing_zed_x_y + dx_RH_flux_phi_odd_trapped_zed_x_y
                    RH_flux_phi_odd_passing_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_odd_passing",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0)
                    RH_flux_phi_odd_trapped_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_odd_trapped",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0)
                    RH_flux_phi_odd_zed_x_y = RH_flux_phi_odd_passing_zed_x_y + RH_flux_phi_odd_trapped_zed_x_y
                except:
                    #dx_RH_flux_phi_odd_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_odd",time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=1)
                    RH_flux_phi_odd_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_odd",time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, kx_order=0)

                #phi_RH_zed_x_y, zed, x, y, time_eval    = self.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True)
                dx_phi_RH_zed_x_y, zed, x, y, time_eval    = self.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True, kx_order=1)
                #f_zed_x_y = -dx_RH_flux_phi_odd_zed_x_y*phi_RH_zed_x_y
                f_zed_x_y = RH_flux_phi_odd_zed_x_y*dx_phi_RH_zed_x_y

            elif quantity == "P_RH_tot":
                P_RH_NL_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("P_RH_NL", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter) 
                P_RH_coll_zed_x_y,    _, _, _,        _  = self.get_quantity_zed_x_y("P_RH_coll",  time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter) 
                f_zed_x_y = P_RH_NL_zed_x_y + P_RH_coll_zed_x_y

            elif quantity == "P_RH_NL":
                P_RH_even_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y("P_RH_even", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter) 
                P_RH_odd_zed_x_y,    _, _, _,        _  = self.get_quantity_zed_x_y("P_RH_odd",  time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter) 
                f_zed_x_y = P_RH_even_zed_x_y + P_RH_odd_zed_x_y
                #RH_flux_phi_odd_zed_x_y, _, _, _, _ = self.get_quantity_zed_x_y("RH_fluxes_phi_odd", time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)
                #dx_phi_RH_zed_x_y, zed, x, y, time_eval    = self.get_quantity_zed_x_y("RH_phi",            time_idx=time_idx, species_idx=species_idx, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, only_zonal=True, kx_order=1)
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
            if time_avg is None:
                f_t_zed_kx_ky_ri = f_zed_kx_ky_ri[None,:,:,:,:]
            else:
                f_t_zed_kx_ky_ri = f_zed_kx_ky_ri

            # Complex variable
            f_t_zed_kx_ky = f_t_zed_kx_ky_ri[:,:,:,:,0] + 1j*f_t_zed_kx_ky_ri[:,:,:,:,1]

            # Filter out kx's
            f_t_zed_kx_ky[:,:,np.abs(kx)>kxmin_filter,:] = 0
            f_t_zed_kx_ky[:,:,np.abs(kx)<kxmax_filter,:] = 0

            # Filter out ky's
            f_t_zed_kx_ky[:,:,:,ky>kymin_filter] = 0
            f_t_zed_kx_ky[:,:,:,ky<kymax_filter] = 0

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

            # Time average
            if time_avg is not None:
                f_zed_x_y = np.sum(f_t_zed_x_y*dt_avg_vals[:,None,None,None], axis=0)/np.sum(dt_avg_vals)
            else:
                f_zed_x_y = f_t_zed_x_y[0,:,:,:]

        else:


            # Time average
            if time_avg is not None:
                f_zed_kx_ky_ri = np.sum(f_zed_kx_ky_ri*dt_avg_vals[:,None,None,None,None], axis=0)/np.sum(dt_avg_vals)

            # Filter out kx's
            f_zed_kx_ky_ri[:,np.abs(kx)>kxmin_filter,:,:] = 0
            f_zed_kx_ky_ri[:,np.abs(kx)<kxmax_filter,:,:] = 0

            # Filter out ky's
            f_zed_kx_ky_ri[:,:,ky>kymin_filter,:] = 0
            f_zed_kx_ky_ri[:,:,ky<kymax_filter,:] = 0

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

            if self.code == "GX":
                # Ensure kx is in FFT form
                idx_kx_0 = np.argmin(np.abs(kx))
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


    def get_quantity_x_y(self, quantity, zed_val = None, zed_idx=None, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, mult_zed=None, kxmin_filter=1e10, kymin_filter=1e10, kxmax_filter=-1, kymax_filter=-1, par_der_order=0, abs_squared=False):

        x_der_taken = False
        f_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity=quantity, time_idx=time_idx, species_idx=species_idx, time_val=time_val ,remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, nx=nx, ny=ny, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter, abs_squared=abs_squared)
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
            gradpar  = self.ncdata.variables['gradpar'][:]
            # Use periodicity
            _, _, _, _, gds22, bmag = self.get_FLR()
            f_zed_x_y = np.gradient(f_zed_x_y*gradpar[:,None,None], zed, axis=0)

        # if zed_val is not None, find zed_idx matching zed_val most closely
        if zed_val is not None:
            zed_idx = np.argmin( np.abs(zed - zed_val) )
    
        if zed_idx is None:
            zed_weight = self.get_zed_weight(mult_zed=mult_zed, zed=zed)
            f_x_y = np.sum(f_zed_x_y*zed_weight[:,None,None], axis=0)
 
        else:
            f_x_y = f_zed_x_y[zed_idx]

        return f_x_y, x, y, time_eval

    def plot_quantity_3d_torus(self, quantity="phi", fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, vmin=None, vmax=None, cmap=None,torus_rmax=0.6, torus_rmin=0.25, Delta_zeta=np.pi/3, nzeta=50, xlim=np.infty, lighting=True, ikymin=0, ikymax=None):


        f_theta_kx_ky, theta, kx, ky, time_eval = self.get_quantity_zed_kx_ky(quantity, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)
        qinp   = self.safety_factor
        shat   = self.ncdata.variables['shat'].getValue()
        jtwist = self.ncdata.variables['jtwist'].getValue()
        drhodpsi   = self.ncdata.variables['drhodpsi'].getValue()
        dqinp_dx = 1/(jtwist)#/jtwist #shat*qinp*drhodpsi/(2*np.pi)
        #print(dqinp_dx)
        theta0 = self.ncdata.variables['theta0'][:]

        #idx_sort_theta = np.argsort(theta)
        #theta = theta[idx_sort_theta]
        #f_theta_kx_ky = f_theta_kx_ky[idx_sort_theta]

        x = np.linspace(-1, 1, len(kx), endpoint=False)*np.pi/(kx[1]-kx[0])
        if xlim is not None:
            x = x[np.abs(x)<xlim]
        nx = len(x)
        zeta = np.linspace(-np.pi+Delta_zeta, np.pi, nzeta)

        if ikymax is None:
            ky = ky[ikymin:]
            f_theta_kx_ky = f_theta_kx_ky[:,:,ikymin:]
        else:
            ky = ky[ikymin:ikymax]
            f_theta_kx_ky = f_theta_kx_ky[:,:,ikymin:ikymax]


        #print(f_theta_kx_ky[0, :10,3])
        #print(f_theta_kx_ky[-1,:10,3])

        # Evaluate quantity at x extrema
        theta_2D_lastx, zeta_2D_lastx = np.meshgrid(theta, zeta)
        f_theta_zeta_firstx = np.zeros((len(theta), len(zeta)))
        f_theta_zeta_lastx  = np.zeros((len(theta), len(zeta)))
        for i_theta in range(len(theta)):
            for i_zeta in range(len(zeta)):
                #eikonalplus = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[0] + ky[None,:]/ky[1]*(zeta[i_zeta]-qinp*theta[i_theta])
                ##eikonalmin  = -eikonalplus
                #eikonalmin  = (kx[:,None] + shat*ky[None,:]*theta[i_theta])*x[0] - ky[None,:]/ky[1]*(zeta[i_zeta]-qinp*theta[i_theta])
                #f_theta_zeta_firstx[i_theta, i_zeta] = np.sum( np.real( f_theta_kx_ky[i_theta]*np.exp(1j*eikonalplus) + np.conj(f_theta_kx_ky[i_theta])*np.exp(1j*eikonalmin) ))

                qinp_x = qinp + dqinp_dx*x[-1] / x[-1]
                eikonal = ky[None,:]/(ky[1]-ky[0])*( zeta[i_zeta]-qinp_x*(theta[i_theta]-theta0) )
                eikonal[:,0] = kx*x[-1]

                #eikonal = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[-1] + ky[None,:]/(ky[1]-ky[0])*(zeta[i_zeta]-qinp*theta[i_theta])/3
                #eikonal = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[-1] + ky[None,:]/(ky[1]-ky[0])*(zeta[i_zeta]-qinp*theta[i_theta])
                f_theta_zeta_lastx[ i_theta, i_zeta] = np.sum( np.real(f_theta_kx_ky[i_theta])*np.cos(eikonal) - np.imag(f_theta_kx_ky[i_theta])*np.sin(eikonal) )

                qinp_x = qinp + dqinp_dx*x[0] / x[-1]
                eikonal = ky[None,:]/(ky[1]-ky[0])*( zeta[i_zeta]-qinp_x*(theta[i_theta]-theta0) )
                eikonal[:,0] = kx*x[0]
                #eikonal = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[0]  + ky[None,:]/(ky[1]-ky[0])*(zeta[i_zeta]-qinp*theta[i_theta])
                f_theta_zeta_firstx[i_theta, i_zeta] = np.sum( np.real(f_theta_kx_ky[i_theta])*np.cos(eikonal) - np.imag(f_theta_kx_ky[i_theta])*np.sin(eikonal) )

        # Evaluate quantity at zeta cuts
        theta_2D_zeta, x_2D_zeta = np.meshgrid(theta, x)
        f_theta_x_b = np.zeros((len(theta), nx))
        f_theta_x_e = np.zeros((len(theta), nx))
        for i_theta in range(len(theta)):
            for i_x in range(nx):
                #eikonal_bplus = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[i_x] + ky[None,:]/ky[1]*(zeta[0]- qinp*theta[i_theta])
                ##eikonal_bmin  = -eikonal_bplus
                #eikonal_bmin  = (kx[:,None] + shat*ky[None,:]*theta[i_theta])*x[i_x] - ky[None,:]/ky[1]*(zeta[0]- qinp*theta[i_theta])
                #eikonal_eplus = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[i_x] + ky[None,:]/ky[1]*(zeta[-1]-qinp*theta[i_theta])
                ##eikonal_emin  = -eikonal_eplus
                #eikonal_emin  = (kx[:,None] + shat*ky[None,:]*theta[i_theta])*x[i_x] - ky[None,:]/ky[1]*(zeta[-1]-qinp*theta[i_theta])
                #f_theta_x_b[i_theta, i_x] = np.sum( np.real(f_theta_kx_ky[i_theta]*np.exp(1j*eikonal_bplus) + np.conj(f_theta_kx_ky[i_theta])*np.exp(1j*eikonal_bmin) ) )
                #f_theta_x_e[i_theta, i_x] = np.sum( np.real(f_theta_kx_ky[i_theta]*np.exp(1j*eikonal_eplus) + np.conj(f_theta_kx_ky[i_theta])*np.exp(1j*eikonal_emin) ) )
                qinp_x = qinp + dqinp_dx*x[i_x] / x[-1]
                eikonal_b = ky[None,:]/(ky[1]-ky[0])*( zeta[ 0]-qinp_x*(theta[i_theta]-theta0) )
                eikonal_b[:,0] = kx*x[i_x]
                eikonal_e = ky[None,:]/(ky[1]-ky[0])*( zeta[-1]-qinp_x*(theta[i_theta]-theta0) )
                eikonal_e[:,0] = kx*x[i_x]

                #eikonal_b = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[i_x] + ky[None,:]/(ky[1]-ky[0])*(zeta[0]- qinp*theta[i_theta])
                #eikonal_e = (kx[:,None] - shat*ky[None,:]*theta[i_theta])*x[i_x] + ky[None,:]/(ky[1]-ky[0])*(zeta[-1]-qinp*theta[i_theta])
                f_theta_x_b[i_theta, i_x] = np.sum( np.real(f_theta_kx_ky[i_theta])*np.cos(eikonal_b) - np.imag(f_theta_kx_ky[i_theta])*np.sin(eikonal_b) )
                f_theta_x_e[i_theta, i_x] = np.sum( np.real(f_theta_kx_ky[i_theta])*np.cos(eikonal_e) - np.imag(f_theta_kx_ky[i_theta])*np.sin(eikonal_e) )

        if vmin is None or vmax is None:
            vmax =  max( np.abs(f_theta_x_b).max(), np.abs(f_theta_x_e).max(),np.abs(f_theta_zeta_firstx).max(), np.abs(f_theta_zeta_lastx).max())
            vmin = -vmax

        #### PLOTS
        from mayavi import mlab
        resolution = 4
        mlab.options.offscreen = True
        fig_mlab = mlab.figure(size=(1024, 1024))

        # Plot at first & last x
        X_rmax = np.cos(zeta_2D_lastx)*(1 + torus_rmax*np.cos(theta_2D_lastx))
        Y_rmax = np.sin(zeta_2D_lastx)*(1 + torus_rmax*np.cos(theta_2D_lastx))
        Z_rmax =                            torus_rmax*np.sin(theta_2D_lastx)
        X_rmin = np.cos(zeta_2D_lastx)*(1 + torus_rmin*np.cos(theta_2D_lastx))
        Y_rmin = np.sin(zeta_2D_lastx)*(1 + torus_rmin*np.cos(theta_2D_lastx))
        Z_rmin =                            torus_rmin*np.sin(theta_2D_lastx)

        Out = mlab.mesh(X_rmin,Y_rmin,Z_rmin,colormap='coolwarm',scalars=f_theta_zeta_firstx.T,figure=fig_mlab, vmin=vmin, vmax=vmax, resolution=resolution)
        Out.actor.property.lighting = lighting
        Out = mlab.mesh(X_rmax,Y_rmax,Z_rmax,colormap='coolwarm',scalars=f_theta_zeta_lastx.T ,figure=fig_mlab, vmin=vmin, vmax=vmax, resolution=resolution)
        Out.actor.property.lighting = lighting

        # Plot at zeta cuts
        X_b = np.cos(zeta[0]) * (1 + np.cos(theta_2D_zeta) * (torus_rmin + (torus_rmax-torus_rmin)*(x_2D_zeta-x[0])/(x[-1]-x[0]) ))
        Y_b = np.sin(zeta[0]) * (1 + np.cos(theta_2D_zeta) * (torus_rmin + (torus_rmax-torus_rmin)*(x_2D_zeta-x[0])/(x[-1]-x[0]) ))
        X_e = np.cos(zeta[-1])* (1 + np.cos(theta_2D_zeta) * (torus_rmin + (torus_rmax-torus_rmin)*(x_2D_zeta-x[0])/(x[-1]-x[0]) ))
        Y_e = np.sin(zeta[-1])* (1 + np.cos(theta_2D_zeta) * (torus_rmin + (torus_rmax-torus_rmin)*(x_2D_zeta-x[0])/(x[-1]-x[0]) ))
        Z =                          np.sin(theta_2D_zeta) * (torus_rmin + (torus_rmax-torus_rmin)*(x_2D_zeta-x[0])/(x[-1]-x[0]) )

        Out = mlab.mesh(X_b,Y_b,Z,colormap='coolwarm',scalars=f_theta_x_b.T,figure=fig_mlab, vmin=vmin, vmax=vmax, resolution=resolution)
        Out.actor.property.lighting = lighting

        Out = mlab.mesh(X_e,Y_e,Z,colormap='coolwarm',scalars=f_theta_x_e.T,figure=fig_mlab, vmin=vmin, vmax=vmax, resolution=resolution)
        Out.actor.property.lighting = lighting

        #mlab.view(azimuth=0, elevation=45, figure=fig_mlab)
        #mlab.view(azimuth=Delta_zeta*180/np.pi*0.2, elevation=0, figure=fig_mlab)
        mlab.view(azimuth=Delta_zeta*180/np.pi*0.25 +180, elevation=70, figure=fig_mlab)
        #mlab.view(azimuth=Delta_zeta, elevation=70, figure=fig_mlab)
        imgmap = mlab.screenshot(figure=fig_mlab, mode='rgba')

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,8))
        ax.imshow(imgmap)
        mlab.close()

        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])

        if quantity == "phi":
            title = r"$\varphi$"
        elif quantity == "density":
            title = r"$n$"
        elif quantity == "upar":
            title = r"$u_\parallel$"
        elif quantity == "temperature":
            title = r"$T$"
        elif quantity == "pressure_perp":
            title = r"$P_\perp$"
        elif quantity == "qpar":
            title = r"$q_\parallel$"
        elif quantity == "qperp":
            title = r"$q_\perp$"
        elif quantity == "dyphi-dxphi":
            title = r"$\partial_y \varphi \partial_x \varphi$"
        elif quantity == "dyphi-dyphi":
            title = r"$(\partial_y \varphi)^2"
        elif quantity == "dyPrp-dxphi":
            title = r"$\partial_y P_\perp \partial_x \varphi$"
        elif quantity == "dyPrp-dyphi":
            title = r"$\partial_y P_\perp \partial_y \varphi$"
        elif quantity == "dyT-dxphi":
            title = r"$\partial_y T \partial_x \varphi$"
        else:
            title = ""

        if remove_zonal:
            title = title+r"$_\mathrm{NZ}$"
        if only_zonal:
            title = title+r"$_\mathrm{Z}$"
        title = title+r"$(t v_{Ti}/a=%.2f)$" % (time_eval)

        if time_avg is not None:
            title = title + r"$_{\Delta t = %.1f}$" % (time_avg)

        fig.suptitle(title)

        return fig, ax, vmin, vmax


    def plot_quantity_poloidal_ring(self, quantity="phi", fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, vmin=None, vmax=None, cmap=None, xmin=None, xmax=None, ymin=None, ymax=None, rorigin_fac=2, zed_idx_skip=1, kyfilter_fac=None, kymin_filter=np.infty):

        if kyfilter_fac is not None:
            ky = self.ncdata['ky'][:]
            kymin_filter = ky[1]*kyfilter_fac*0.9999
            print(kymin_filter)

        quantity_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity=quantity, species_idx=species_idx, time_val=time_val, time_idx=time_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, ny=ny, nx=nx, kymin_filter=kymin_filter)

        if ax is None:
            fig, ax = plt.subplots(figsize=(12,10), subplot_kw=dict(projection='polar'))

        if kyfilter_fac is not None:
            ymin = y[0] /(kyfilter_fac-1)
            ymax = y[-1]/(kyfilter_fac-1)
            print(ymin)
            print(ymax)

        if xmin is not None:
            quantity_zed_x_y = quantity_zed_x_y[:,x>xmin]
            x = x[x>xmin]
        if xmax is not None:
            quantity_zed_x_y = quantity_zed_x_y[:,x<xmax]
            x = x[x<xmax]
        if ymin is not None:
            quantity_zed_x_y = quantity_zed_x_y[:,:,y>ymin]
            y = y[y>ymin]
        if ymax is not None:
            quantity_zed_x_y = quantity_zed_x_y[:,:,y<ymax]
            y = y[y<ymax]

        ax.set_rorigin(-rorigin_fac*x[-1])

        #zed = zed[::zed_idx_skip]
        #quantity_zed_x_y = quantity_zed_x_y[::zed_idx_skip,:,:]

        zed = zed[::zed_idx_skip]
        quantity_zed_x_y = quantity_zed_x_y[::zed_idx_skip,:,:]

        nzed = len(zed)
        dzed = zed[1]-zed[0]
        ny   = len(y)
        dy   = y[1]-y[0]
        Ly   = y[-1]-y[0] + dy
        angle     = np.zeros(int(nzed*ny))
        f_x_angle = np.zeros(shape=(len(x),int(nzed*ny)))
        for i_zed in range(nzed):
            angle[int(i_zed*ny):int((i_zed+1)*ny)] = zed[i_zed] + dzed*y/Ly
            #f_x_angle[:,int(i_zed*ny):int((i_zed+1)*ny)] = quantity_zed_x_y[0,:,:]
            f_x_angle[:,int(i_zed*ny):int((i_zed+1)*ny)] = quantity_zed_x_y[i_zed,:,:]

        assert(np.all(np.diff(angle)>=0))

        # Bring back between -pi and pi
        angle = (angle-angle[0])/(angle[-1]-angle[0])*2*np.pi - np.pi

        X, Y = np.meshgrid(x, angle)
        Z = f_x_angle.T

        Zabsmax = np.abs(Z).max()
        if vmin is None:
            vmin = -Zabsmax
        if vmax is None:
            vmax = Zabsmax

        im = ax.pcolormesh(Y, X, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)

        if quantity == "phi":
            title = r"$\varphi$"
        elif quantity == "density":
            title = r"$n$"
        elif quantity == "upar":
            title = r"$u_\parallel$"
        elif quantity == "temperature":
            title = r"$T$"
        elif quantity == "pressure_perp":
            title = r"$P_\perp$"
        elif quantity == "qpar":
            title = r"$q_\parallel$"
        elif quantity == "qperp":
            title = r"$q_\perp$"
        elif quantity == "dyphi-dxphi":
            title = r"$\partial_y \varphi \partial_x \varphi$"
        elif quantity == "dyphi-dyphi":
            title = r"$(\partial_y \varphi)^2"
        elif quantity == "dyPrp-dxphi":
            title = r"$\partial_y P_\perp \partial_x \varphi$"
        elif quantity == "dyPrp-dyphi":
            title = r"$\partial_y P_\perp \partial_y \varphi$"
        elif quantity == "dyT-dxphi":
            title = r"$\partial_y T \partial_x \varphi$"
        else:
            title = ""

        if remove_zonal:
            title = title+r"$_\mathrm{NZ}$"
        if only_zonal:
            title = title+r"$_\mathrm{Z}$"
        title = title+r"$(t=%.2f)$" % (time_eval)

        if time_avg is not None:
            title = title + r"$_{\Delta t = %.1f}$" % (time_avg)

        fig.suptitle(title)

        return fig, ax, im, vmin, vmax


    def plot_quantity_box_zed_x_y(self, quantity="phi", fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, symm=False, vmin=None, vmax=None, kxmin_filter=np.infty, kymin_filter=np.infty, kxmax_filter=-1, kymax_filter=-1, cmap=None, xmin=None, xmax=None, ymin=None, ymax=None, zed_neg=True):

        quantity_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity=quantity, species_idx=species_idx, time_val=time_val, time_idx=time_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, ny=ny, nx=nx, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)

        if ax is None:
            fig = plt.figure(figsize=(12,9))
            ax = fig.add_subplot(projection='3d')

        if xmin is not None:
            quantity_zed_x_y = quantity_zed_x_y[:,x>xmin]
            x = x[x>xmin]
        if xmax is not None:
            quantity_zed_x_y = quantity_zed_x_y[:,x<xmax]
            x = x[x<xmax]
        if ymin is not None:
            quantity_zed_x_y = quantity_zed_x_y[:,:,y>ymin]
            y = y[y>ymin]
        if ymax is not None:
            quantity_zed_x_y = quantity_zed_x_y[:,:,y<ymax]
            y = y[y<ymax]

        if zed_neg:
            quantity_zed_x_y = quantity_zed_x_y[zed<0]
            zed = zed[zed<0]


        #xmax = 20
        #x = x[np.abs(x)<xmax]
        #quantity_zed_x_y = quantity_zed_x_y 

        X1, X2, X3 = np.meshgrid(zed, x, y, indexing='ij')
        data    = quantity_zed_x_y
        #data    = np.transpose(quantity_zed_x_y, axes=(1,0,2))

        if symm:
            vmax = (np.abs(data)).max()
            vmin = -vmax

        kw = {
            'vmin': vmin,
            'vmax': vmax,
            'cmap': cmap,
            'levels': 50
        }

        # Plot contour surfaces

        # zed-x (top)
        _ = ax.contourf(
            X1[:, :, -1], X2[:, :, -1], data[:, :, -1],
            zdir='z', offset=y.max(), **kw
        )

        # zed-y (side)
        _ = ax.contourf(
            X1[:, 0, :], data[:, 0, :], X3[:, 0, :],
            zdir='y', offset=x.min(), **kw
        )

        # x-y (front)
        im = ax.contourf(
            data[-1, :, :], X2[-1, :, :], X3[-1, :, :],
            zdir='x', offset=zed.max(), **kw
        )
        ax.set(xlim=[X1.min(), X1.max()], ylim=[X2.min(), X2.max()], zlim=[X3.min(), X3.max()])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_box_aspect((2,1,1))
        ax.view_init(azim=-30, elev=15)
        # --
    
        if quantity == "phi":
            title = r"$\varphi$"
        elif quantity == "density":
            title = r"$n$"
        elif quantity == "upar":
            title = r"$u_\parallel$"
        elif quantity == "temperature":
            title = r"$T$"
        elif quantity == "pressure_perp":
            title = r"$P_\perp$"
        elif quantity == "qpar":
            title = r"$q_\parallel$"
        elif quantity == "qperp":
            title = r"$q_\perp$"
        elif quantity == "dyphi-dxphi":
            title = r"$\partial_y \varphi \partial_x \varphi$"
        elif quantity == "dyphi-dyphi":
            title = r"$(\partial_y \varphi)^2"
        elif quantity == "dyPrp-dxphi":
            title = r"$\partial_y P_\perp \partial_x \varphi$"
        elif quantity == "dyPrp-dyphi":
            title = r"$\partial_y P_\perp \partial_y \varphi$"
        elif quantity == "dyT-dxphi":
            title = r"$\partial_y T \partial_x \varphi$"
        else:
            title = ""

        if remove_zonal:
            title = title+r"$_\mathrm{NZ}$"
        if only_zonal:
            title = title+r"$_\mathrm{Z}$"
        title = title+r"$(t=%.2f)$" % (time_eval)

        if time_avg is not None:
            title = title + r"$_{\Delta t = %.1f}$" % (time_avg)

        #ax.set_title(title)
        ax.set_xlabel("\n\n"+r"$\zeta$")
        ax.set_ylabel("\n"+r"$x/\rho_i$")
        ax.set_zlabel(r"$y/\rho_i$")
        fig.suptitle(title)

        return fig, ax, im, vmin, vmax


    def plot_quantity_x_y(self, quantity="phi", fig=None, ax=None, zed_val=None, zed_idx=None, mult_zed=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, show_iota_x=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, symm=False, vmin=None, vmax=None, kxmin_filter=np.infty, kymin_filter=np.infty, kxmax_filter=-1, kymax_filter=-1, cmap=None, xmin=None, xmax=None, ymin=None, ymax=None, interpolation=False, projection_3d=False, plot_contours=False, suptitle=True, xy_layout=True):

        quantity_x_y, x, y, time_eval = self.get_quantity_x_y(quantity=quantity, zed_val=zed_val, zed_idx=zed_idx, mult_zed=mult_zed, species_idx=species_idx, time_val=time_val, time_idx=time_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, ny=ny, nx=nx, kxmin_filter=kxmin_filter, kymin_filter=kymin_filter, kxmax_filter=kxmax_filter, kymax_filter=kymax_filter)

        if ax is None:
            if projection_3d:
                fig, ax = plt.figure(figsize=(12,9)).add_subplot(projection='3d')
            else:
                fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        if xmin is not None:
            quantity_x_y = quantity_x_y[x>xmin]
            x = x[x>xmin]
        if xmax is not None:
            quantity_x_y = quantity_x_y[x<xmax]
            x = x[x<xmax]
        if ymin is not None:
            quantity_x_y = quantity_x_y[:,y>ymin]
            y = y[y>ymin]
        if ymax is not None:
            quantity_x_y = quantity_x_y[:,y<ymax]
            y = y[y<ymax]

        X, Y = np.meshgrid(x, y)
        Z    = quantity_x_y.T

        if symm or vmin == "symm":
            vmax = (np.abs(Z)).max()
            vmin = -vmax

        if projection_3d:
            # From https://matplotlib.org/stable/gallery/mplot3d/contour3d_3.html#sphx-glr-gallery-mplot3d-contour3d-3-py
            # Plot projections of the contours for each dimension.  By choosing offsets
            # that match the appropriate axes limits, the projected contours will sit on
            # the 'walls' of the graph.
            ax.contour(X, Y, Z, zdir='z', offset=vmin,  cmap=cmap)
            ax.contour(X, Y, Z, zdir='x', offset=x[0],  cmap=cmap)
            ax.contour(X, Y, Z, zdir='y', offset=y[-1], cmap=cmap)
            # Plot the 3D surface
            im = ax.plot_surface(X, Y, Z, color="None", edgecolor='k', lw=0.5, rstride=8, cstride=8, alpha=0.1)
            ax.set_xlim([x[0],x[-1]])
            ax.set_ylim([y[0],y[-1]])
            ax.set_zlim([vmin, vmax])

        else:
            if interpolation:
                shading='gouraud'
            else:
                shading='auto'
    
            if vmin is not None and vmax is not None:
                if cmap is None:
                    cmap = 'coolwarm'
                if plot_contours:
                    #dx, dy = np.gradient(Z)
                    #im = ax.quiver(X, Y, -dx, -dy, scale=50)
                    Z = Z/np.abs(Z).max()
                    levels = [-1, -2/3, -1/3, 0, 1/3, 2/3, 1]
                    im = ax.contour(X, Y, Z, levels=levels, colors=cmap, vmin=vmin, vmax=vmax, linewidths=2)
                else:
                    #im = ax.pcolormesh(X, Y, Z, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)

                    dx = x[1]-x[0]
                    dy = y[1]-y[0]

                    if xy_layout:
                        im = ax.imshow(Z, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap, extent=[x.min()-dx/2, x.max()-dx/2, y.min()-dy/2, y.max()-dy/2], aspect='auto', origin='lower')
                    else:
                        im = ax.imshow(Z.T, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap, extent=[y.min()-dy/2, y.max()-dy/2, x.min()-dx/2, x.max()-dx/2], aspect='auto', origin='lower')
                    #im = ax.pcolormesh(X, Y, Z.T, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)

            else:
                if cmap is None:
                    cmap = 'inferno'
                if plot_contours:
                    #dZdx, dZdy = np.gradient(Z, x[1]-x[0], y[1]-y[0])
                    #magnitude = np.sqrt(dZdx**2 + dZdy**2)
                    #dZdx /= magnitude
                    #dZdy /= magnitude
                    #im = ax.quiver(X, Y, -dZdy, dZdx, scale=30, pivot='middle')#, scale=1e-10*Z.max()/(x[1]-x[0]))
                    
                    Z = Z/np.abs(Z).max()
                    levels = [-1, -2/3, -1/3, 0, 1/3, 2/3, 1]
                    im = ax.contour(X, Y, Z, levels=levels, colors=cmap, linewidths=2)
                    #im = ax.streamplot(X[0,:], Y[:,0], dx, dy)#, scale=1e-10*Z.max()/(x[1]-x[0]))
                else:

                    im = ax.pcolormesh(X, Y, Z, shading=shading, cmap=cmap, rasterized=True)

                ax.set_aspect('equal')
    

        if quantity == "phi":
            title = r"$\varphi$"
        elif quantity == "density":
            title = r"$n$"
        elif quantity == "upar":
            title = r"$u_\parallel$"
        elif quantity == "temperature":
            title = r"$T$"
        elif quantity == "pressure_perp":
            title = r"$P_\perp$"
        elif quantity == "qpar":
            title = r"$q_\parallel$"
        elif quantity == "qperp":
            title = r"$q_\perp$"
        elif quantity == "dyphi-dxphi":
            title = r"$\partial_y \varphi \partial_x \varphi$"
        elif quantity == "dyphi-dyphi":
            title = r"$(\partial_y \varphi)^2"
        elif quantity == "dyPrp-dxphi":
            title = r"$\partial_y P_\perp \partial_x \varphi$"
        elif quantity == "dyPrp-dyphi":
            title = r"$\partial_y P_\perp \partial_y \varphi$"
        elif quantity == "dyT-dxphi":
            title = r"$\partial_y T \partial_x \varphi$"
        else:
            title = ""

        if remove_zonal:
            title = title+r"$_\mathrm{NZ}$"
        if only_zonal:
            title = title+r"$_\mathrm{Z}$"
        title = title+r"$(t=%.2f$ $a/v_T)$" % (time_eval)

        if time_avg is not None:
            title = title + r"$_{\Delta t = %.1f}$" % (time_avg)

        #ax.set_title(title)
        if projection_3d:
            ax.set_xlabel("\n"+r"$x/\rho_i$")
            ax.set_ylabel("\n"+r"$y/\rho_i$")
            ax.set_zlabel("\n"+title)
        else:
            if xy_layout:
                ax.set_xlabel(r"$x/\rho_i$")
                ax.set_ylabel(r"$y/\rho_i$")
            else:
                ax.set_xlabel(r"$y/\rho_i$")
                ax.set_ylabel(r"$x/\rho_i$")
            if suptitle:
                fig.suptitle(title)

        return fig, ax, im, vmin, vmax


    def plot_spectrum2(self, quantity, kx_or_ky, fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, c=None, lw=None, label=None, marker='.', scale_kmin=True, scale_CB=False, zed_val=None, zed_idx=None, ls='-', mult_zed=None):

        if quantity == "upar_over_phi":
            upar_kx_ky, kx, ky, time_eval = self.get_quantity_kx_ky(quantity="upar", zed_val=zed_val, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)
            phi_kx_ky, kx, ky, time_eval = self.get_quantity_kx_ky(quantity="phi", zed_val=zed_val, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)
            quantity_kx_ky = np.abs(upar_kx_ky)/np.abs(phi_kx_ky)
            quantity_zed_kx_ky = quantity_kx_ky[None,:,:]

        elif quantity == "temp_over_phi":
            temp_kx_ky, kx, ky, time_eval = self.get_quantity_kx_ky(quantity="temperature", zed_val=zed_val, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)
            phi_kx_ky, kx, ky, time_eval = self.get_quantity_kx_ky(quantity="phi", zed_val=zed_val, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)
            quantity_kx_ky = np.abs(temp_kx_ky)/np.abs(phi_kx_ky)

            quantity_zed_kx_ky = quantity_kx_ky[None,:,:]

        else:
            quantity_zed_kx_ky, zed, kx, ky, time_eval = self.get_quantity_zed_kx_ky(quantity=quantity, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)
            #quantity_kx_ky, kx, ky, time_eval = self.get_quantity_kx_ky(quantity=quantity, zed_val=zed_val, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg)


        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        if kx_or_ky == "kx":
            kx = np.abs(kx)
            k = kx[1:]
            dk = ky[1]-ky[0]
            quantity_zed_k = np.mean(np.real(quantity_zed_kx_ky[:,1:]*np.conj(quantity_zed_kx_ky[:,1:])), axis=2)/dk

            #quantity_k = np.sum(np.abs(quantity_kx_ky[1:])**2, axis=1)/dk

        if kx_or_ky == "ky":
            k  = ky[1:]
            dk = kx[1]-kx[0]
            quantity_zed_k = np.mean(np.real(quantity_zed_kx_ky[:,:,1:]*np.conj(quantity_zed_kx_ky[:,:,1:])), axis=1)/dk
            #quantity_k = np.sum( np.abs(quantity_kx_ky[:,1:])**2, axis=0)*dk

        # Average over zed
        zed_weight = self.get_zed_weight(mult_zed=mult_zed, zed=zed)
        quantity_k = np.sum(quantity_zed_k*zed_weight[:,None], axis=0)

#        # Rescale quantities according to Critical Balance
#        if scale_CB:
#            print("Rescaling according to critical balance")
#            tprim = self.ncdata.variables['tprim'][0]
#            kappa = tprim
#            if kx_or_ky == "kx":
##                quantity_k = quantity_k / kappa**(7/2)
#            if kx_or_ky == "ky":
#                k = k * kappa
##                quantity_k = quantity_k / kappa**(4)
#            quantity_k = quantity_k / kappa**(3)
                

        if scale_kmin:
#            print("Rescaling quantity with kmin (to be able to compare sims with different x0,y0)")
            quantity_k = quantity_k / np.abs(k[1]-k[0])**2

        ax.loglog(k, quantity_k, label=label, lw=lw, c=c, marker=marker, ls=ls)
#        ax.set_xlim(xmin=0.75*k.min())
        
        return fig, ax, time_eval

    def plot_Q_x_y(self, fig=None, ax=None, zed_idx=None, time_idx=-1, species_idx=0, time_val=None):

        Q_x_y, _, _, x, y, time_eval = self.get_Q_x_y(zed_idx=zed_idx, time_val=time_val, time_idx=time_idx, species_idx=species_idx)

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        X, Y = np.meshgrid(x, y)
        Z = Q_x_y.T

        im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='inferno')

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(r"$Q(t=%.2f)$" % (time_eval))

        ax.set_aspect('equal')

        return fig, ax, im

    def plot_quantity_x(self, quantity="phi", species_idx=0, fig=None, ax=None, zed_idx=None, time_idx=-1, label=None, ls=None, color=None, marker=None, normalise=False, time_avg=None, nx=None, mult_zed=None, kx_order=0, kxmin_filter=1e5, mult=1, plot_factor=1):

        f_Z,       x, _, time_eval = self.get_quantity_x_y(quantity=quantity, species_idx=species_idx, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=False, only_zonal=True, kx_order=kx_order, time_avg=time_avg, nx=nx, mult_zed=mult_zed, kxmin_filter=kxmin_filter)

        # Make 1D array
        f_Z       = mult*f_Z[:,0]

        if normalise:
            norm_val = 1/np.abs(f_Z).max()
            f_Z_plot = f_Z*norm_val*plot_factor
        else:
            norm_val = 1
            f_Z_plot = f_Z*plot_factor

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(8,5))

        title = r"$t v_T/a = %.2f$" % (time_eval)
        if time_avg is not None:
            title = title + r"$_{\Delta t = %.1f}$" % (time_avg)
        fig.suptitle(title)

        ax.plot(x, f_Z_plot,  ls=ls, c=color, marker=marker, label=label)
        ax.grid(True, alpha=0.5)
        ax.set_xlim(xmin=x[0],xmax=x[-1])

        if label is not None:
            ax.legend()

        ax.set_xlabel(r"$x/\rho_i$")
        return fig, ax, norm_val, x, f_Z

    def plot_quantity_zonal(self, quantity="phi", species_idx=0, fig=None, axs=None, zed_idx=None, time_idx=-1, label=None, ls=None, color=None, marker=None, substract_background_temp=False, normalise=False, time_avg=None, nx=None, sum_nonzonal=False, mult_zed=None, kx_order_min=0, kxmin_filter=1e5, mult=1):

        if not sum_nonzonal:
            f_Z,       x, _, time_eval = self.get_quantity_x_y(quantity=quantity, species_idx=species_idx, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=False, only_zonal=True, kx_order=kx_order_min+0, time_avg=time_avg, nx=nx, mult_zed=mult_zed, kxmin_filter=kxmin_filter)
            fprime_Z,  x, _, time_eval = self.get_quantity_x_y(quantity=quantity, species_idx=species_idx, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=False, only_zonal=True, kx_order=kx_order_min+1, time_avg=time_avg, nx=nx, mult_zed=mult_zed, kxmin_filter=kxmin_filter)
            fdprime_Z, x, _, time_eval = self.get_quantity_x_y(quantity=quantity, species_idx=species_idx, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=False, only_zonal=True, kx_order=kx_order_min+2, time_avg=time_avg, nx=nx, mult_zed=mult_zed, kxmin_filter=kxmin_filter)

            # Make 1D array
            f_Z       = mult*f_Z[:,0]
            fprime_Z  = mult*fprime_Z[:,0]
            fdprime_Z = mult*fdprime_Z[:,0]

        else:
            f_Z,       x, _, time_eval = self.get_quantity_x_y(quantity=quantity, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=True, only_zonal=False, kx_order=0, time_avg=time_avg, nx=nx)
            fprime_Z,  x, _, time_eval = self.get_quantity_x_y(quantity=quantity, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=True, only_zonal=False, kx_order=1, time_avg=time_avg, nx=nx)
            fdprime_Z, x, _, time_eval = self.get_quantity_x_y(quantity=quantity, zed_idx=zed_idx, time_idx=time_idx, remove_zonal=True, only_zonal=False, kx_order=2, time_avg=time_avg, nx=nx)

            # Make 1D array
            f_Z       = mult*np.sqrt(np.mean(np.abs(f_Z)**2       , axis=1))
            fprime_Z  = mult*np.sqrt(np.mean(np.abs(fprime_Z)**2  , axis=1))
            fdprime_Z = mult*np.sqrt(np.mean(np.abs(fdprime_Z)**2 , axis=1))


#        if quantity=="temperature" and substract_background:
#            self.ncdata = nc4.Dataset(self.netcdf_file,'r')
#            tprim  = self.ncdata.variables['tprim'][:]
#            f_Z    = f_Z - tprim*x
#            fdprime_Z    = fdprime_Z - tprim

        if normalise:
            f_Z       = f_Z       / np.abs(f_Z      ).max()
            fprime_Z  = fprime_Z  / np.abs(fprime_Z ).max()
            fdprime_Z = fdprime_Z / np.abs(fdprime_Z).max()

        if axs is None:
            fig, axs = plt.subplots(nrows=3,ncols=1, figsize=(8,14), sharex=True)
            plt.subplots_adjust(left=0.15,right=0.95, hspace=0.05)

        title = r"$t= %.2f$" % (time_eval)
        if time_avg is not None:
            title = title + r"$_{\Delta t = %.1f}$" % (time_avg)
        fig.suptitle(title)

        axs[0].plot(x, f_Z,       ls=ls, c=color, marker=marker, label=label)
        axs[1].plot(x, fprime_Z,  ls=ls, c=color, marker=marker, label=r"$\partial_x $" + label)
        axs[2].plot(x, fdprime_Z, ls=ls, c=color, marker=marker, label=r"$\partial^2_x $" + label)

        for ax in axs:
            ax.grid(True)
            ax.set_xlim(xmin=x[0],xmax=x[-1])

        if label is not None:
            axs[0].legend()
            axs[1].legend()
            axs[2].legend()

        axs[2].set_xlabel(r"$x/\rho$")
        #axs[0].set_ylabel(r"$f_Z$")
        #axs[1].set_ylabel(r"$f'_Z$")
        #axs[2].set_ylabel(r"$f''_Z$")

        return fig, axs

    def get_quantity_omega_zed_kx(self, quantity, time_min, time_max, time_idx_skip=1, species_idx=0, remove_zonal=False, only_zonal=False, kx_order=0, omega_min=-np.infty, omega_max=np.infty, alt_slow_eval=True): 

        time_all  =  self.ncdata.variables['t'][:]
        time_idx_min = np.argmin(np.abs(time_all - time_min))
        time_idx_max = np.argmin(np.abs(time_all - time_max))
        time_idxs = range(time_idx_min, time_idx_max, time_idx_skip)
        time = time_all[time_idxs]
        assert(len(time) > 0)
        assert(len(time_idxs) == len(time))

        # Obtain quantity as a function of time
        for i_idx, time_idx in enumerate(time_idxs):
            print("Quantity = " + quantity + ": evaluating time_idx %.6i/%i..." % (i_idx+1, len(time_idxs)), end="\r")
            f_zed_kx_ky, zed, kx, ky, time_eval = self.get_quantity_zed_kx_ky(quantity=quantity, time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, alt_slow_eval=alt_slow_eval)
            if i_idx == 0:
                f_t_zed_kx = np.zeros((len(time),len(zed),len(kx)), dtype='complex')
            f_t_zed_kx[i_idx] = f_zed_kx_ky[:,:,0]

        # Resample to equal time-intervals
        dt = (np.gradient(time)).max()
        time_interp = np.arange(time[0], time[-1], dt)
        f_interp = interp(time, f_t_zed_kx, assume_sorted=True, axis=0)
        f_t_zed_kx_interp = f_interp(time_interp)

        # Fourier transform time to omega
        f_omega_zed_kx = np.fft.fft(f_t_zed_kx_interp, axis=0)
        omega = np.fft.fftfreq(len(time_interp), d=dt)*(2*np.pi)

        # Filter out omega outside range
        f_omega_zed_kx = f_omega_zed_kx[(omega<omega_max) & (omega>omega_min)]
        omega  = omega[(omega<omega_max) & (omega>omega_min)]

#        idx_sort = np.argsort(omega)
#        f_omega_zed_kx = f_omega_zed_kx[idx_sort]
#        omega = omega[idx_sort]

        return f_omega_zed_kx, omega, zed, kx

    def get_quantity_filtered_in_omega(self, f_t, time, omega_min=-np.infty, omega_max=np.infty):

        # Resample to equal time-intervals
        dt = np.max(np.gradient(time))
        time_interp = np.arange(time[0], time[-1], dt)
        f_interp = interp(time, f_t, assume_sorted=True, axis=0)
        f_t_interp = f_interp(time_interp)

        # Fourier transform time to omega
        f_omega = np.fft.fft(f_t_interp, axis=0)
        omega = np.fft.fftfreq(len(time_interp), d=dt)*(2*np.pi)

        # Filter out omega outside range
        f_omega = f_omega[(omega<=omega_max) & (omega>=omega_min)]
        omega   =   omega[(omega<=omega_max) & (omega>=omega_min)]
        #return f_omega, omega

        # Transform back to real space
        f_t_filtered = np.fft.ifft(f_omega, axis=0)
        time_new = np.linspace(time[0], time[0]+2*np.pi/omega[1], len(omega))

        return f_t_filtered, time_new


    def plot_quantity_kx_omega(self, quantity, time_min, time_max, time_idx_skip=1, fig=None, ax=None, vmin=None, vmax=None, species_idx=0, logarithmic=False, remove_zonal=False, only_zonal=False, cmap='inferno', kx_order=0, par_der_order=0, mult_zed=None, zed_val=None, no_plot=False, omega_min=-np.infty, omega_max=np.infty, time_der=False, plot_omega2_kx2=False, mean_delt_zed=None, alt_slow_eval=False, append_mirror=False, normalise_each_kx=False, omega_norm=1, scale_eps=1):

        kx, ky, zed = self.get_kx_ky_zed()
        time_all    = self.get_time_array(GX_big=True)
        dl_over_B_avg = self.dl_over_B_avg()
        time_idx_min = np.argmin(np.abs(time_all - time_min))
        time_idx_max = np.argmin(np.abs(time_all - time_max))
        time_idxs = range(time_idx_min, time_idx_max, time_idx_skip)
        time = time_all[time_idxs]

        assert(len(time) > 0)
        assert(len(time_idxs) == len(time))

        for i_idx, time_idx in enumerate(time_idxs):
            print("Evaluating time_idx %.6i/%i..." % (i_idx+1, len(time_idxs)), end="\r")
            f_kx_ky, kx, ky, _ = self.get_quantity_kx_ky(quantity=quantity, remove_zonal=remove_zonal, only_zonal=only_zonal, time_idx=time_idx, kx_order=kx_order, mult_zed=mult_zed, par_der_order=par_der_order, mean_delt_zed=mean_delt_zed, alt_slow_eval=alt_slow_eval, zed_val=zed_val)
            if i_idx == 0:
                f_kx_ky_t = np.zeros((len(kx), len(ky), len(time)), dtype=np.complex_)
            f_kx_ky_t[:,:,i_idx] = f_kx_ky

        # Add mirror of sample if required
        if append_mirror:
            time = np.concatenate((time, time+time[-1]+time[1]-time[0]))
            f_kx_ky_t = np.concatenate((f_kx_ky_t, f_kx_ky_t[:,:,::-1]), axis=2)

        # Resample to equal time-intervals
        dt = (np.gradient(time)).max()
        time_interp = np.arange(time[0], time[-1], dt)
        f_interp = interp(time, f_kx_ky_t, assume_sorted=True, axis=2)
        f_kx_ky_t_interp = f_interp(time_interp)

        # Take time derivative if required
        if time_der:
            f_kx_ky_t_interp = np.gradient(f_kx_ky_t_interp, axis=2)/dt

        # Fourier transform time to omega
        #f_kx_ky_omega = np.fft.fft(f_kx_ky_t, axis=2)
        #omega = np.fft.fftfreq(len(time))
        f_kx_ky_omega = np.fft.fft(f_kx_ky_t_interp, axis=2)
        omega = -np.fft.fftfreq(len(time_interp), d=dt)*(2*np.pi)

        # Take care of ky
        if only_zonal:
            f_kx_omega = f_kx_ky_omega[:,0,:]
        else:
            # Summing over ky
            print("Note! Summing over ky")
            f_kx_omega = np.sum(f_kx_ky_omega, axis=1)

        f_kx_omega     = f_kx_omega[:, (omega<=omega_max) & (omega>=omega_min)]
        omega  = omega[(omega<=omega_max) & (omega>=omega_min)]

 
        if normalise_each_kx:
            for i_kx in range(len(kx)):
                norm = (np.abs(f_kx_omega[i_kx,:])).max()
                if norm != 0:
                    f_kx_omega[i_kx,:] = f_kx_omega[i_kx,:]/norm

        ## Find peak at larger kx
        #f_kx_omega_subset = f_kx_omega[np.abs(kx) > 0.4]
        #kx_subset = kx[np.abs(kx) > 0.4]
        #idx_max = np.argmax( f_kx_omega_subset )
        #idx_max = np.unravel_index(idx_max, f_kx_omega_subset.shape)
        #print("kx = %.2e, omega = %.2e, omega/kx = %.2e at maximum (kx > 0.4)." % (kx_subset[idx_max[0]], omega[idx_max[1]], omega[idx_max[1]]/kx_subset[idx_max[0]]))

        if not no_plot:
            #Ascending order
            idx_omega = np.argsort(omega)
            idx_kx    = np.argsort(kx)
            omega = omega[idx_omega]
            kx    = kx[idx_kx]
            f_kx_omega = f_kx_omega[idx_kx,:][:,idx_omega]

            if ax is None:
                fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,10))
                plt.subplots_adjust(left=0.15,right=0.95)

            # Rescale with R/a fac
            omega_norm = omega_norm*scale_eps
            f_kx_omega = f_kx_omega/scale_eps
            
            # Plot
            Z    = np.abs(f_kx_omega)
            #Z    = f_kx_omega

            if vmax is None:
                vmax = Z[1:,:].max()
            if logarithmic and vmin is not None:
                vmin = vmin*vmax
            if vmin is None:
                vmin = Z[1:,:].min()
            print("vmin = %e" % (vmin))
            print("vmax = %e" % (vmax))


            if not plot_omega2_kx2:
                X, Y = np.meshgrid(kx, omega/omega_norm)
                dkx = kx[1]-kx[0]
                dom = (omega[1]-omega[0])/omega_norm
                if logarithmic:
                    im = ax.imshow(Z.T, norm=colors.LogNorm(vmin=vmin, vmax=vmax), interpolation='nearest', cmap=cmap, extent=[kx.min()-dkx/2, kx.max()-dkx/2, omega.min()/omega_norm-dom/2, omega.max()/omega_norm-dom/2], aspect='auto', origin='lower')
                    ax.set_xlim([kx.min()-dkx/2, kx.max()+dkx/2])
                    ax.set_ylim([omega.min()/omega_norm-dom/2, omega.max()/omega_norm-dom/2])
                    #im = ax.pcolormesh(X, Y, Z.T, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
                else:
                    im = ax.pcolormesh(X, Y, Z.T, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
                ax.set_xlabel(r"$k_x \rho_i$")
                if scale_eps == 1:
                    ax.set_ylabel(r"$\omega a/v_{Ti}$")
                else:
                    ax.set_ylabel(r"$\omega R/v_{Ti}$")
            else:
                X, Y = np.meshgrid(kx**2, (omega/omega_norm)**2)
                if logarithmic:
                    im = ax.pcolormesh(X, Y, Z.T, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
                else:
                    im = ax.pcolormesh(X, Y, Z.T, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
                ax.set_xlabel(r"$(k_x \rho_i)^2$")
                if scale_eps:
                    ax.set_ylabel(r"$(\omega R/v_T)^2$")
                else:
                    ax.set_ylabel(r"$(\omega a/v_T)^2$")

        else:
            im = None

        return fig, ax, im, kx, omega, f_kx_omega

    def plot_quantity_x_t(self, quantity, fig=None, ax=None, vmin=None, vmax=None, species_idx=0, logarithmic=False, remove_zonal=False, only_zonal=False, time_idx_skip=1, normalise_each_t=False, y_val=None, cmap='inferno', kx_order=0, zed_val=None, zed_idx=None, mult_zed=None, time_min=0, time_max=1e10, nx=None, kxmin_filter=1e4, kxmax_filter=-1, par_der_order=0, scale_eps=1, return_avg=False, mult=1):

        time_all    =  self.get_time_array(GX_big=True)
        kx, ky, zed = self.get_kx_ky_zed()
        time_idx_min = np.argmin(np.abs(time_all-time_min))
        time_idx_max = np.argmin(np.abs(time_all-time_max))
        time      = time_all[time_idx_min:time_idx_max:time_idx_skip]
        time_idxs = range(time_idx_min, time_idx_max, time_idx_skip)
        assert(len(time)==len(time_idxs))

        if nx is None:
            nx = len(kx)

        f_t_x_y = np.zeros( (len(time_idxs), nx, 2*len(ky)-1) )
        for i_idx, time_idx in enumerate(time_idxs):
            print("Evaluating time_idx %.6i/%i..." % (i_idx+1, len(time_idxs)), end="\r")
            f_x_y, x, y, _ = self.get_quantity_x_y(quantity=quantity, remove_zonal=remove_zonal, only_zonal=only_zonal, time_idx=time_idx, kx_order=kx_order, zed_val=zed_val, zed_idx=zed_idx, mult_zed=mult_zed, nx=nx, kxmin_filter=kxmin_filter, kxmax_filter=kxmax_filter, par_der_order=par_der_order)
            f_t_x_y[i_idx] = f_x_y*mult

        if only_zonal:
            f_t_x = f_t_x_y[:,:,0]
        elif y_val is None:
            # Integrate over y
            print("Note! Integrating over y")
            dy = y[1]-y[0]
            f_t_x = np.sum(f_t_x_y, axis=2)*dy
        else:
            yval_idx = np.argmin( np.abs(y-y_val) )
            f_t_x = f_t_x_y[:,:,yval_idx]
 
#        f_t_zed_kx_ky = f_t_zed_kx_ky_ri[::time_idx_skip,:,:,:,0] + 1j*f_t_zed_kx_ky_ri[::time_idx_skip,:,:,:,1]

#        # Filter zonal if requested
#        if remove_zonal:
#            f_t_zed_kx_ky[:,:,:,0]= 0
#        if only_zonal:
#            f_t_zed_kx_ky[:,:,:,1:]= 0
#
#        # Average in zed
#        dl_over_B_avg = self.dl_over_B_avg()
#        f_t_kx_ky = np.sum(f_t_zed_kx_ky * dl_over_B_avg[None,:,None,None], axis=1)
#
#        # FT in kx, sum over ky
#        x = np.fft.fftshift(np.fft.fftfreq(len(kx),d=(kx[1]-kx[0])/(2*np.pi)))
#        f_t_x = np.sum( np.real(np.fft.ifft(f_t_kx_ky, axis=1)), axis=2)
#        #f_t_x = np.sum( np.abs(np.fft.ifft(f_t_kx_ky, axis=1)), axis=2)

        time  = time*scale_eps
        f_t_x = f_t_x/scale_eps

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,10))
            plt.subplots_adjust(left=0.15,right=0.95)

        X, Y = np.meshgrid(time, x)
        Z    = f_t_x.T

        if normalise_each_t:
            for time_idx in range(len(time)):
                Z[:,time_idx] = Z[:,time_idx]/max(np.abs(Z[:,time_idx]))

#        if logarithmic and vmax == "auto":
#            maxabs = np.abs(Z).max()
#            vmax =  maxabs
#            vmin = vmin*maxabs
#        if vmin == "sym" or vmax == "auto":
#            maxabs = np.abs(Z).max()
#            vmin = -maxabs
#            vmax =  maxabs
#            cmap = 'coolwarm'
#        if vmax is None:
#            vmax = Z.max()
#        if vmin is None:
#            vmin = Z.min()
        if vmax is None:
            vmax = Z.max()
        if vmax == "last":
            vmax = np.abs(Z[:,-1]).max()
        if vmin == "symm":
            vmin = -vmax

        #print(vmin)
        #print(vmax)

        if logarithmic:
            im = ax.pcolormesh(X, Y, np.abs(Z), norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
        else:
            im = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap, rasterized=True)
            #dx = x[1]-x[0]; xmin=x[0]-dx/2; xmax=x[-1]+dx/2
            #dt = time[1]-time[0]; tmin=time[0]-dt/2; tmax=time[-1]+dt/2
            #im = ax.imshow(Z, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest', aspect='auto', origin='lower', extent=[tmin, tmax, xmin, xmax])
            ax.set_xlim([time[0], time[-1]])
            ax.set_ylim([x[0],    x[-1]])

        if scale_eps == 1:
            ax.set_xlabel(r"$t v_T/a$")
        else:
            ax.set_xlabel(r"$t v_T/R$")
        ax.set_ylabel(r"$x/\rho_i$")

        if return_avg:
            f_t_mean = np.mean(f_t_x, axis=1)
            Lx = x[-1]-x[0]
            f_t_mean_norm = f_t_mean/np.abs(f_t_mean).max() * Lx/4
            return fig, ax, im, time, f_t_mean_norm

        else:
            return fig, ax, im, X, Y, Z


    def plot_quantity_x_zed(self, quantity="phi", fig=None, ax=None, time_idx=-1, vmin=None, vmax=None, logarithmic=False, remove_zonal=False, only_zonal=False, avg_norm=None, nx=None, ny=None, species_idx=0, cmap='inferno', kx_order=0, ky_order=0, kxmin_filter=1000, kxmax_filter=0, polar_plot=False, idx_x_shift=None, mult_zed=None, mult_fac=1, xlim_box=None):

        # Figure
        if ax is None:
            if polar_plot:
        #        x += x[-1]/2
                fig, ax = plt.subplots(figsize=(12,10), subplot_kw=dict(projection='polar'))
                ax.set_rorigin(-2*x[-1])
            else:
                fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,10))
        #    plt.subplots_adjust(left=0.15,right=0.95)

        # Load data
        if isinstance(quantity, str):
            f_zed_x_y, zed, x, y, time_eval = self.get_quantity_zed_x_y(quantity, time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, nx=nx, ny=ny, kx_order=kx_order, ky_order=ky_order, kxmin_filter=kxmin_filter, kxmax_filter=kxmax_filter)

            # zed weight and multiplication factor
            zed_weight = self.get_zed_weight(mult_zed, zed)
            f_zed_x_y = f_zed_x_y*zed_weight[:,None,None]*mult_fac

            # Shift in x if desired
            if idx_x_shift and idx_x_shift > 0 and idx_x_shift < len(x)-1:
                idx_sort  = np.concatenate( (range(idx_x_shift,len(x)), range(idx_x_shift)) )
                f_zed_x_y = f_zed_x_y[:,idx_sort,:]

            if only_zonal:
                avg_norm = "zonal"

            # Average over y
            if avg_norm == "abs":
                f_zed_x = np.sum( np.abs(f_zed_x_y), axis=2 )
            elif avg_norm == 2:
                f_zed_x = np.sqrt( np.sum( f_zed_x_y**2, axis=2 ) )
            elif avg_norm == "center":
                f_zed_x = f_zed_x_y[:,:,0]
            else:
                f_zed_x = np.sum( f_zed_x_y, axis=2 )

            # Plot only part of box if desired
            if xlim_box is not None:
                idx_min = np.argmin(np.abs(x-xlim_box[0]))
                idx_max = np.argmin(np.abs(x-xlim_box[1]))
    
                x = x[idx_min:idx_max]
                f_zed_x = f_zed_x[:,idx_min:idx_max]

            fig.suptitle(r"$t v_T/a=%.2f$" % (time_eval))


        else:
            f_zed_x = quantity
            kx, _, zed = self.get_kx_ky_zed()
            xmax = np.pi/(kx[1]-kx[0])
            x = np.linspace(-xmax, xmax, len(f_zed_x[0]), endpoint=False)

        # Plot

        X, Y = np.meshgrid(x, zed)
        Z = f_zed_x

        if vmax is None:
            vmax = np.abs(Z).max()
        if vmax == "last":
            vmax = np.abs(Z[:,-1]).max()
        if vmin == "symm":
            vmin = -vmax
        elif vmin is None:
            if logarithmic:
                vmin = 1e-2*vmax
            else:
                vmin = Z.min()

        if logarithmic:
            im = ax.pcolormesh(Y, X, np.abs(Z), norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)#, rasterized=True)
            #im = ax.pcolormesh(X, Y, np.abs(Z), norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
        else:
            #im = ax.contourf(Y, X, Z)#, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
            im = ax.pcolormesh(Y, X, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)#, rasterized=True)

        im.set_edgecolor('face')

        if not polar_plot:
            ax.set_ylim(ymin=x[0],ymax=x[-1])
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$x/\rho$")

        ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

        # Evaluate zed_avg(x)
        zed_avg_x = np.sum(f_zed_x*zed[:,None], axis=0)/np.sum(np.abs(f_zed_x), axis=0)
        #dl_over_B_avg = self.dl_over_B_avg()
        #zed_avg_x = np.sum(f_zed_x*dl_over_B_avg[:,None], axis=0)

        return fig, ax, im, x, zed, f_zed_x, zed_avg_x, vmin, vmax

    def plot_quantity1_quantity2(self, quantities, fig=None, ax=None, ls="--", c=None, marker='.', time_min=0, time_max=99999, time_idx_skip=1, remove_zonals=[False,False], only_zonals=[False,False], avg_norms=[None,None], nx=None, ny=None, species_idx=0, kx_orders=[0,0], ky_orders=[0,0], mult_zeds=[None, None], time_ders=[False, False], mult_vals=[1,1], all_xs=False):

        # Determine time over which to plot
        time_all   = self.ncdata.variables['t'][:]#[::time_idx_skip]
        time_idx_min = np.argmin(np.abs(time_all-time_min))
        time_idx_max = np.argmin(np.abs(time_all-time_max))
        time_plot    = time_all[time_idx_min:time_idx_max:time_idx_skip]
        time_idxs = range(time_idx_min, time_idx_max, time_idx_skip)
        assert(len(time_plot)==len(time_idxs))

        if nx is None:
            kx, _, _  = self.get_kx_ky_zed()
            nx = len(kx)

        # Load quantities
        if all_xs:
            f12_t = np.zeros((2, len(time_plot)*nx))
        else:
            f12_t = np.zeros((2, len(time_plot)))

        for i_quantity, quantity in enumerate(quantities):
            kx_order = kx_orders[i_quantity]
            ky_order = ky_orders[i_quantity]
            avg_norm = avg_norms[i_quantity]
            mult_zed = mult_zeds[i_quantity]
            only_zonal = only_zonals[i_quantity]
            remove_zonal = remove_zonals[i_quantity]

            for i_idx, time_idx in enumerate(time_idxs):
                print("Evaluating time_idx %.6i/%i..." % (i_idx+1, len(time_idxs)), end="\r")
    
                x_der_taken = False
                y_der_taken = False
                if quantity == "phi-phi":
                    phi_x_y,  x, y, time_eval = self.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, remove_zonal=True, only_zonal=False, nx=nx, ny=ny, mult_zed=mult_zed)
                    f_x_y = phi_x_y**2
    
                elif quantity == "phi-pressure_perp":
                    phi_x_y,  x, y, time_eval = self.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, nx=nx, ny=ny, mult_zed=mult_zed)
                    Pprp_x_y,  x, y, time_eval = self.get_quantity_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, nx=nx, ny=ny, mult_zed=mult_zed)
                    f_x_y = phi_x_y * Pprp_x_y
    
#                elif quantity == "dyphi-T":
#                    dyphi_x_y,  x, y, time_eval = self.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny)
#                    T_x_y,  x, y, time_eval = self.get_quantity_x_y("temperature", time_idx=time_idx, species_idx=species_idx, nx=nx, ny=ny)
#                    f_x_y = dyphi_x_y * T_x_y
    
                elif quantity == "dyphi-dyPprp":
                    dyphi_x_y,  x, y, time_eval = self.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny)
                    dyPprp_x_y,  x, y, time_eval = self.get_quantity_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny)
                    f_x_y = dyphi_x_y * dyPprp_x_y
    
                elif quantity == "dxphi-dyPprp":
                    dxphi_x_y,  x, y, time_eval = self.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, kx_order=1, nx=nx, ny=ny)
                    dyPprp_x_y,  x, y, time_eval = self.get_quantity_x_y("pressure_perp", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny)
                    f_x_y = dxphi_x_y * dyPprp_x_y
    
                elif quantity == "dyphi-dyphi":
                    dyphi_x_y,  x, y, time_eval = self.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny)
                    f_x_y = dyphi_x_y**2
    
                elif quantity == "kx-avg":
                    phi_kx_ky,  kx, ky, time_eval = self.get_quantity_kx_ky("phi", time_idx=time_idx, species_idx=species_idx)
                    kx_avg = np.sum( kx[None,:,None] * np.abs(phi_kx_ky[:,:,1:])**2, axis=(1,2)) / np.sum( np.abs(phi_kx_ky[:,:,1:])**2, axis=(1,2))
    
                    f_x_y = kx_avg[:,None,None]
    
                elif quantity == "dxphi-dyphi":
                    dxphi_x_y,  x, y, time_eval = self.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, kx_order=1, nx=nx, ny=ny)
                    dyphi_x_y,  x, y, time_eval = self.get_quantity_x_y("phi", time_idx=time_idx, species_idx=species_idx, ky_order=1, nx=nx, ny=ny)
                    f_x_y = dxphi_x_y * dyphi_x_y
    
                else:
                    f_x_y,  x, y, time_eval = self.get_quantity_x_y(quantity=quantity, time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, nx=nx, ny=ny, mult_zed=mult_zed)
                    x_der_taken = True
                    y_der_taken = True

    
                # Take derivatives by finite differences if needed
                if not x_der_taken:
                    for i in range(kx_order):
                        f_x_y = np.gradient(f_x_y, axis=0)/(x[1]-x[0])
                if not y_der_taken:
                    for i in range(ky_order):
                        f_x_y = np.gradient(f_x_y, axis=1)/(y[1]-y[0])
    
                if all_xs:
                    # Average over y
                    if avg_norm == "abs":
                        f_x = np.sum( np.abs(f_x_y), axis=1)
                    elif avg_norm == 2:
                        f_x = np.sqrt( np.sum( f_x_y**2, axis=1) )
                    elif avg_norm == "center":
                        f_x = f_x_y[:,0]
                    else:
                        f_x = np.sum( f_x_y , axis=1)
                    f12_t[i_quantity, nx*i_idx:nx*(i_idx+1)] = f_x*mult_vals[i_quantity]
 
                else:
                    # Average over x-y
                    if avg_norm == "abs":
                        f = np.sum( np.abs(f_x_y))
                    elif avg_norm == 2:
                        f = np.sqrt( np.sum( f_x_y**2) )
                    elif avg_norm == "center":
                        f = f_x_y[0,0]
                    elif avg_norm == "zonal_center":
                        f = np.sum(f_x_y[0])
                    else:
                        f = np.sum( f_x_y )
    
                    # Save to array
                    f12_t[i_quantity, i_idx] = f*mult_vals[i_quantity]

            # Time derivative if required
            dt = np.gradient(time_plot)
            if time_ders[i_quantity]:
                f12_t[i_quantity] = np.gradient(f12_t[i_quantity])/dt
 
        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,10))


        if all_xs:
            ax.scatter(f12_t[0], f12_t[1],              marker=marker, s=100, cmap='inferno')
        else:
            ax.plot(f12_t[0], f12_t[1], ls=ls, c=c)
            ax.scatter(f12_t[0], f12_t[1], c=time_plot, marker=marker, s=100, cmap='inferno')
        ax.grid()

        return fig, ax

    def plot_quantity_t_k(self, quantity="phi", fig=None, ax=None, remove_zonal=False, ky_idx=None, only_zonal=False, ls=None, lw=None, log_ax=True, t_min=0, t_max=1e6, ratio_zonal_nonzonal=False, kx_min=-1, kx_idxs=None, time_idx_skip=1, species_idx=0, kx_order=0, ky_order=0, eval_real=False, eval_imag=False, colors=None, marker=None, no_plot=False, norm_plot=False, sum_kx=False, labels=None):
        
        ky           = self.ncdata.variables['ky'][:]
        kx           = self.ncdata.variables['kx'][:] 
        time_all = self.get_time_array()
        time_idx_min = np.argmin(np.abs(time_all-t_min))
        time_idx_max = np.argmin(np.abs(time_all-t_max))
        time = time_all[time_idx_min:time_idx_max:time_idx_skip]
        dl_over_B_avg = self.dl_over_B_avg()

        if quantity=="phi":
            # phi_vs_t(t, tube, zed, theta0, ky, ri)
            f_t_zed_kx_ky_ri = self.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max:time_idx_skip,0,:,:,:,:]
        elif quantity=="phi2":
            # phi2_vs_kxky(t, kx, ky)
            phi2_t_kx_ky = self.ncdata.variables['phi2_vs_kxky'][time_idx_min:time_idx_max:time_idx_skip,:,:]
            f_t_zed_kx_ky = phi2_t_kx_ky[:,None,:,:]
        elif quantity=="density":
            # density(t, species, tube, zed, kx, ky, ri)
            f_t_zed_kx_ky_ri = self.ncdata.variables['density'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
        elif quantity=="upar":
            # upar(t, species, tube, zed, kx, ky, ri)
            f_t_zed_kx_ky_ri = self.ncdata.variables['upar'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
        elif quantity=="temperature":
            # temperature(t, species, tube, zed, kx, ky, ri)
            f_t_zed_kx_ky_ri = self.ncdata.variables['temperature'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
        elif quantity=="pressure_par":
            P_t_zed_kx_ky_ri = self.ncdata.variables['pressure'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
            try:
                Pprp_t_zed_kx_ky_ri = self.ncdata.variables['pressure_perp'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
            except:
                Pprp_t_zed_kx_ky_ri = self.ncdata.variables['pressure_prp'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
            f_t_zed_kx_ky_ri = P_t_zed_kx_ky_ri-0.5*Pprp_t_zed_kx_ky_ri
        elif quantity=="pressure_perp":
            # pressure_perp(t, species, tube, zed, kx, ky, ri)
            try:
                f_t_zed_kx_ky_ri = self.ncdata.variables['pressure_perp'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
            except:
                f_t_zed_kx_ky_ri = self.ncdata.variables['pressure_prp'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
        elif quantity=="qpar":
            # qpar(t, species, tube, zed, kx, ky, ri)
            f_t_zed_kx_ky_ri = self.ncdata.variables['qpar'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
        elif quantity=="qperp":
            # qperp(t, species, tube, zed, kx, ky, ri)
            f_t_zed_kx_ky_ri = self.ncdata.variables['qperp'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:,:]
        elif quantity=="qflx":
            # qflx_kxky(t, species, tube, zed, kx, ky)
            f_t_zed_kx_ky = self.ncdata.variables['qflx_kxky'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:]
        elif quantity=="pflx":
            # pflx_kxky(t, species, tube, zed, kx, ky)
            f_t_zed_kx_ky = np.abs(self.ncdata.variables['pflx_kxky'][time_idx_min:time_idx_max:time_idx_skip,species_idx,0,:,:,:])

        elif quantity in ["Reynolds", "par_mom_transport", "dEZ_par_mom_transport", "pressure_transport"]:

            time_idx_min = np.argmin(np.abs(time-t_min))
            time_idx_max = np.argmin(np.abs(time-t_max))
            time_idx_eval = np.arange(time_idx_min, time_idx_max, time_idx_skip)

            f_t_kx = np.zeros((len(time_idx_eval), int((1+len(kx))/2)), dtype='complex')
            for i_time_idx, time_idx in enumerate(time_idx_eval):
                print("Time idx %4i/%4i" % (i_time_idx, len(time_idx_eval)), end="\r")

                f_zed_kx_ky, _, kx, _, _ = self.get_quantity_zed_kx_ky(quantity=quantity, time_idx=time_idx)

                # Recall int(dy) f(y) = L_y f_{ky=0}/2
                #reynolds_stress_t_kx[i_time_idx] = reynolds_stress_zed_kx_ky[0,:,0]*(y[-1]-y[0])/2
                f_t_kx[i_time_idx] = f_zed_kx_ky[0,:,0]/(2*len(y))

            time = time[time_idx_eval]
            f_t_zed_kx_ky = f_t_kx[:,None,:,None]
            dl_over_B_avg[:] = 1/len(dl_over_B_avg)
            ky_idx = 0

        else:
            print("Did not enter valid quantity to plot (" + str(quantity) + "). Returning")
            return


        # For some quantities, evaluate abs() or real part if desired
        if quantity in ["phi", "density", "upar", "temperature", "pressure_perp", "pressure_par", "qpar", "qperp"]:
            f_t_zed_kx_ky = f_t_zed_kx_ky_ri[:,:,:,:,0] + 1j*f_t_zed_kx_ky_ri[:,:,:,:,1]

#            if eval_real:
#                f_t_zed_kx_ky = f_t_zed_kx_ky_ri[:,:,:,:,0]
#            else:
#                f_t_zed_kx_ky = np.abs(f_t_zed_kx_ky_ri[:,:,:,:,0] + 1j*f_t_zed_kx_ky_ri[:,:,:,:,1])

        # Filter out ky's now if requested to avoid work in summing
        if only_zonal:
            ky_idx = 0
        if ky_idx is not None and not ky_idx == "abs" and not ky_idx == "SB":
            f_t_zed_kx_ky[:,:,:,:ky_idx] = 0
            f_t_zed_kx_ky[:,:,:,ky_idx+1:] = 0

        # x-derivatives
        f_t_zed_kx_ky = f_t_zed_kx_ky * np.abs(kx[None,None,:,None])**kx_order
        #f_t_zed_kx_ky = f_t_zed_kx_ky * np.abs(kx[None,None,:,None]/(kx[1]-kx[0]))**kx_order

        # y-derivatives
        f_t_zed_kx_ky = f_t_zed_kx_ky * np.abs(ky[None,None,None,:])**ky_order
        #f_t_zed_kx_ky = f_t_zed_kx_ky * np.abs(ky[None,None,None,:]/(ky[1]-ky[0]))**ky_order


#        print("Extracting kx's...")
        if kx_idxs is None:
            print("Will plot for all kx")
            kx_idxs = 1e15

        if sum_kx or len(np.shape(kx_idxs)) == 0:
            kx_idxs = [i for i in range(len(kx)) if (np.abs(kx[i]) < kx_idxs and np.abs(kx[i]) > kx_min)]
            #kx_idxs = [i for i in range(len(kx)) if (np.abs(kx[i]) < kx_idxs and kx[i] > 0)]

        kx_plot  = kx[kx_idxs]
        idx_sort = np.argsort(kx_plot)
        kx_sort = kx_plot[idx_sort]
        kx_idxs_sort = np.array(kx_idxs)[idx_sort.astype(int)]
        nx = len(kx_idxs_sort)
        
        if colors is None:
            colors = sns.color_palette("coolwarm", nx)
            #colors = sns.color_palette("rocket", nx)
        elif len(np.shape(colors)) == 0:
            colors = sns.color_palette(colors, nx)

        f_t_zed_kx_ky = f_t_zed_kx_ky[:,:,kx_idxs_sort]

        ## Take zed average
        f_t_kx_ky = np.sum(dl_over_B_avg[None,:,None,None]*f_t_zed_kx_ky, axis=1)

        if ky_idx is None:
            if ratio_zonal_nonzonal:
                f_t_kx =  f_t_kx_ky[:,:,0] / np.sum(f_t_kx_ky[:,:,1:], axis=2)
            elif remove_zonal:
                f_t_kx = np.sum(f_t_kx_ky[:,:,1:], axis=2)
            elif only_zonal:
                f_t_kx = f_t_kx_ky[:,:,0]
            else:
                f_t_kx = np.sum(f_t_kx_ky, axis=2)
        elif ky_idx == "abs":
            f_t_kx = np.sum(np.abs(f_t_kx_ky), axis=2)
        elif ky_idx == "SB":
            f_t_kx = np.sum(f_t_kx_ky*np.exp(1j*np.pi/2* ky[None,None,:]/ky[-1]), axis=2)
            #f_t_zed_kx = np.sum(f_t_zed_kx_ky[:,:,:,::2], axis=3)

            #f_zed_kx_ky_abs = np.abs(f_t_zed_kx_ky[-1,:,:,1:])
            #print(np.shape(f_zed_kx_ky_abs))
            #argmax = np.unravel_index(np.argmax(f_zed_kx_ky_abs), f_zed_kx_ky_abs.shape)
            #print(argmax)
            #ky_idx_SB = argmax[2]
            #f_t_zed_kx = f_t_zed_kx_ky[:,:,:,1+ky_idx_SB]
        else:
            f_t_kx = f_t_kx_ky[:,:,ky_idx]

   #     if ky_idx is None:
   #         if ratio_zonal_nonzonal:
   #             f_t_zed_kx =  f_t_zed_kx_ky[:,:,:,0] / np.sum(f_t_zed_kx_ky[:,:,:,1:], axis=3)
   #         elif remove_zonal:
   #             f_t_zed_kx = np.sum(f_t_zed_kx_ky[:,:,:,1:], axis=3)
   #         elif only_zonal:
   #             f_t_zed_kx = f_t_zed_kx_ky[:,:,:,0]
   #         else:
   #             f_t_zed_kx = np.sum(f_t_zed_kx_ky, axis=3)
   #     elif ky_idx == "abs":
   #         f_t_zed_kx = np.sum(np.abs(f_t_zed_kx_ky), axis=3)
   #     elif ky_idx == "SB":
   #         f_t_zed_kx = np.sum(f_t_zed_kx_ky*np.exp(1j*np.pi/2* ky[None,None,None,:]/ky[-1]), axis=3)
   #         #f_t_zed_kx = np.sum(f_t_zed_kx_ky[:,:,:,::2], axis=3)

   #         #f_zed_kx_ky_abs = np.abs(f_t_zed_kx_ky[-1,:,:,1:])
   #         #print(np.shape(f_zed_kx_ky_abs))
   #         #argmax = np.unravel_index(np.argmax(f_zed_kx_ky_abs), f_zed_kx_ky_abs.shape)
   #         #print(argmax)
   #         #ky_idx_SB = argmax[2]
   #         #f_t_zed_kx = f_t_zed_kx_ky[:,:,:,1+ky_idx_SB]
   #     else:
   #         f_t_zed_kx = f_t_zed_kx_ky[:,:,:,ky_idx]
   #
   #     # Take zed average
   #     f_t_kx = np.sum(dl_over_B_avg[None,:,None]*f_t_zed_kx, axis=1)

        if sum_kx:
            kx_idxs_sort = [0]
            f_t_kx[:,0]  = np.sum(f_t_kx, axis=1)
            f_t_kx[:,1:] = 0

 
        if not no_plot:
            if ax is None:
                fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

            for i_kx, kx_idx in enumerate(kx_idxs_sort):
    
                if eval_real:
                    f_t_plot = np.real(f_t_kx[:,i_kx])
                elif eval_imag:
                    f_t_plot = np.imag(f_t_kx[:,i_kx])
                else:
                    f_t_plot = np.abs(f_t_kx[:,i_kx])

                if norm_plot:
                    f_t_plot = f_t_plot/(np.abs(f_t_plot).max())
        
                if labels is not None:
                    if len(labels) == len(kx_idxs_sort):
                        label = labels[i_kx]
                    elif labels == "firstlast":
                        if i_kx == 0 or i_kx == len(kx_idxs_sort)-1:
                            label = r"$k_x \rho_i = %.3f$" % (kx_sort[i_kx])
                        else:
                            label = None
                    elif labels == "minlast":
                        #if kx_sort[i_kx] == np.abs(kx_sort).min() or i_kx == len(kx_idxs_sort)-1 or i_kx==0:
                        if kx_sort[i_kx] == np.abs(kx_sort[np.abs(kx_sort)>0]).min() or i_kx == len(kx_idxs_sort)-1:
                            label = r"$k_x \rho_i = %.3f$" % (kx_sort[i_kx])
                        else:
                            label = None
                    else:
                        label = None
                else:
                    label = None

                if log_ax:
                    ax.semilogy(time, np.abs(f_t_plot),label=label, ls=ls, c=colors[i_kx], lw=lw, marker=marker)
                else:
                    ax.plot(time, f_t_plot, label=label, ls=ls, c=colors[i_kx], lw=lw, marker=marker)
    
            ax.set_xlabel(r"$t$")
    
            if labels is not None:
                ax.legend()

        return fig, ax, time, kx_sort, f_t_kx


    def plot_phi_t_ky(self, fig=None, ax=None, zed_idx=None, remove_zonal=False, only_zonal=False, label=None, ls=None, c=None, lw=None, log_ax=True, norm_to_t0=False, plot_abs=True, t_max=np.infty, time_avg=1, norm_kperp2=False, ratio_zonal_nonzonal=False):
        

        # phi_vs_t(t, tube, zed, theta0, ky, ri)
        phi_vs_t  = self.ncdata.variables['phi_vs_t'][:,0,:,:,:,:]
        zed       = self.ncdata.variables['zed'][:]
        ky        = self.ncdata.variables['ky'] 
        kx        = self.ncdata.variables['kx'] 
        time      = self.ncdata.variables['t'][:]

        dl_over_B_avg = self.dl_over_B_avg()
         
        # if zed_idx = None, average over tube
        if zed_idx is None:
            phi_t_ky = np.zeros(shape=(len(time),len(ky)))
            for i_zed in range(len(zed)):
                if plot_abs:
                    phi_t_ky = phi_t_ky + np.sum( np.abs(phi_vs_t[:,i_zed,:,:,0]+1j*phi_vs_t[:,i_zed,:,:,1])**2, axis=1)*dl_over_B_avg[i_zed]
                else:
                    phi_t_ky = phi_t_ky + np.sum( phi_vs_t[:,i_zed,:,:,0], axis=1)*dl_over_B_avg[i_zed]
                #phi_t_ky = phi_t_ky + np.sum( np.real(phi_vs_t[:,i_zed,:,:,0]+1j*phi_vs_t[:,i_zed,:,:,1]), axis=1)*dl_over_B_avg[i_zed]
        else:
            if plot_abs:
                phi_t_ky = np.sum( np.abs(phi_vs_t[:,zed_idx,:,:,0]+1j*phi_vs_t[:,zed_idx,:,:,1])**2, axis=1)
            else:
                phi_t_ky = np.sum( dl_over_B_avg[None,:,None,None]*phi_vs_t[:,zed_idx,:,:,0], axis=1)
            #phi_t_ky = np.sum( np.real(phi_vs_t[:,zed_idx,:,:,0]+1j*phi_vs_t[:,zed_idx,:,:,1]), axis=1)

        # Filter zonal if requested
        if ratio_zonal_nonzonal:
            phi_t =  phi_t_ky[:,0] / np.sum(phi_t_ky[:,1:], axis=1)
        elif remove_zonal:
            phi_t = np.sum(phi_t_ky[:,1:], axis=1)
        elif only_zonal:
            phi_t = phi_t_ky[:,0]
        else:
            phi_t = np.sum(phi_t_ky, axis=1)

        # Only keep t<=tmax
        phi_t = phi_t[time < t_max]
        time  = time[time < t_max]

        phi_end = np.mean(phi_t[time > max(0, time[-1]-time_avg)])
        # Normalise by flux-tube averaged kperp2 if desired
        if norm_kperp2:
            kperp2 = self.get_avg_kperp2()
            print(self.input_file + ": <kperp2> = %e" % (kperp2))
            phi_t = phi_t/kperp2

        # Plot
        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        if norm_to_t0:
            phi_t = phi_t / phi_t[0]
        if log_ax:
            ax.semilogy(time, np.abs(phi_t),label=label, ls=ls, c=c, lw=lw)
        else:
            ax.plot(time, phi_t,label=label, ls=ls, c=c, lw=lw)

        ax.set_xlabel(r"$t v_T/a$")
        if norm_to_t0 and plot_abs:
            label = r"$|\varphi(t)/\varphi(t=0)|^2$"
        elif not norm_to_t0 and plot_abs:
            label = r"$|\varphi(t)|^2$"
        elif norm_to_t0 and not plot_abs:
            label = r"$\varphi(t)/\varphi(t=0)$"
        else:
            label = r"$\varphi(t)$"

        if norm_kperp2:
            label = label + r"$/\langle (k_\perp \rho)^2\rangle$"
        ax.set_ylabel(label)

        ax.legend()
        ax.grid()

        return fig, ax, phi_end
        #return fig, ax, phi_end, time, phi_t

    def get_Wenergy_t_zed_kx_ky(self, time_idx_min=None, time_idx_max=None, time_min=0, time_max=10000, time_idx_skip=1, tite=1):

        if time_idx_min is None:
            time_idx_min = self.get_time_idx(time_min)
        if time_idx_max is None:
            time_idx_max = self.get_time_idx(time_max)
        time = self.ncdata.variables['t'][time_idx_min:time_idx_max:time_idx_skip]
        kx   = self.ncdata.variables['kx'][:]
        ky   = self.ncdata.variables['ky'][:]
        zed  = self.ncdata.variables['zed'][:]

        # Energy in phi
        # phi_vs_t(t, tube, zed, theta0, ky, ri)
        phi_t_zed_kx_ky_ri  = self.ncdata.variables['phi_vs_t'][time_idx_min:time_idx_max:time_idx_skip,0,:,:,:,:]
        phi_t_zed_kx_ky = phi_t_zed_kx_ky_ri[:,:,:,:,0] + 1j*phi_t_zed_kx_ky_ri[:,:,:,:,1]
        dl_over_B_avg = self.dl_over_B_avg()
        kperp2 = self.ncdata.variables['kperp2'][:,0,:,:]
        bmag   = self.ncdata.variables['bmag'][:,0]
        Gamma0_arg = kperp2/ ((2*bmag**2)[:,None,None])
        Gamma0 = specialfunc.iv(0, Gamma0_arg) * np.exp(-Gamma0_arg)
        phiZ_t_kx = np.sum(phi_t_zed_kx_ky[:,:,:,0]*dl_over_B_avg[None,:,None], axis=1)

        Wenergy_phi_e_t_zed_kx_ky = tite*(np.abs(phi_t_zed_kx_ky-phiZ_t_kx[:,None,:,None]*(1-np.heaviside(ky,0))[None,None,None,:] )**2) /2
        
        Wenergy_phi_i_t_zed_kx_ky = ( (1-Gamma0)[None,:,:,:]*np.abs(phi_t_zed_kx_ky)**2 )/2

        # Energy in g
        #double Wenergy_g(t, species, tube, zed, kx, ky) ;
        Wenergy_g_t_zed_kx_ky  = self.ncdata['Wenergy_g'][time_idx_min:time_idx_max:time_idx_skip,0,0,:,:,:]

        return Wenergy_g_t_zed_kx_ky, Wenergy_phi_e_t_zed_kx_ky, Wenergy_phi_i_t_zed_kx_ky, time, zed, kx, ky
        

    def get_gvpa_gmu(self, time_idx=-1, species_idx=0, remove_zonal=False, only_zonal=False):

        if self.code == "stella":
            # gvmus(t, species, mu, vpa)
            if only_zonal:
                gvmus  = self.ncdata.variables['gvmus_Z'][time_idx,species_idx]
            elif remove_zonal:
                gvmus  = self.ncdata.variables['gvmus_NZ'][time_idx,species_idx]
            else:
                gvmus  = self.ncdata.variables['gvmus'][time_idx,species_idx]
    
            vpa    = self.ncdata.variables['vpa'][:]
            mu     = self.ncdata.variables['mu'][:]
            time   = self.ncdata.variables['t'][time_idx]
    
            bmag = 1
            integrand = gvmus
    
        elif self.code == "GX":
            vpa = np.linspace(-3,3,50)
            mu  = np.linspace(0,5,100)
            time   = self.ncdata.variables['time'][time_idx]
            from scipy import special

            Wml  = self.ncdata['Spectra']['Wlmst'][time_idx,species_idx]
            nhermite  = self.ncdata['nhermite'].getValue()
            nlaguerre = self.ncdata['nlaguerre'].getValue()

            print("Wml(2,0)/Wml(0,0) = %e" % (Wml[2,0]/Wml[0,0]))
            print("Wml(0,1)/Wml(0,0) = %e" % (Wml[0,1]/Wml[0,0]))
            #Wml[:,:] = 0
            #Wml[0,1] = 0

            Wvmus = np.zeros((len(mu),len(vpa)))
            for i_hermite in range(nhermite):
                hermite_pol = special.hermitenorm(i_hermite)
                for i_laguerre in range(nlaguerre):
                    laguerre_pol = special.laguerre(i_laguerre)

                    Wvmus = Wvmus + Wml[i_hermite,i_laguerre] * hermite_pol(vpa)[None,:] * laguerre_pol(mu)[:,None] * (-1)**(i_laguerre) * np.exp(-mu)[:,None] * np.exp(-vpa**2/2)[None,:] * (np.math.factorial(i_hermite))**(-1/2)
                    #Wvmus = Wvmus + Wml[i_hermite,i_laguerre] * hermite_pol(vpa)[None,:] * laguerre_pol(mu)[:,None] * (-1)**(i_laguerre) * np.exp(-mu)[:,None] * np.exp(-vpa**2/2)[None,:] * (np.math.factorial(i_hermite))**(-1/2)

                    #Wvmus = Wvmus + Wml[ i_hermite, i_laguerre] * hermite_pol(vpa)[None,:] * laguerre_pol(mu)[:,None] * np.exp(-mu)[:,None] * np.exp(-vpa**2/2)[None,:] / np.sqrt( np.math.factorial(i_hermite) )
                    #Wvmus = Wvmus + Wml[i_hermite,i_laguerre] * hermite_pol(vpa)[None,:] * laguerre_pol(mu)[:,None] * (-1)**(i_laguerre) * np.exp(-mu)[:,None] * np.exp(-vpa**2/2)[None,:] / np.sqrt( np.math.factorial(i_hermite) )

            integrand = Wvmus


        dmu  = np.abs( mu[1]- mu[0])
        dvpa = np.abs(vpa[1]-vpa[0])
        gmu  = np.sum( integrand, axis=1) * dvpa
        gvpa = np.sum( integrand, axis=0) * dmu


        return gmu, gvpa, mu, vpa, time

    def get_Evpa_Emu(self, time_idx=-1, species_idx=0):

        # gvmus(t, species, mu, vpa)
        gvmus  = self.ncdata.variables['gvmus'][time_idx,species_idx]
        vpa    = self.ncdata.variables['vpa'][:]
        mu     = self.ncdata.variables['mu'][:]
        time   = self.ncdata.variables['t'][time_idx]

        bmag = 1
        integrand = gvmus * (vpa[None,:]**2 + 2*mu[:,None]*bmag)/2

        dmu  = np.abs( mu[1]- mu[0])
        dvpa = np.abs(vpa[1]-vpa[0])
        Emu  = np.sum( integrand, axis=1) * dvpa
        Evpa = np.sum( integrand, axis=0) * dmu

        return Emu, Evpa, mu, vpa, time

    def get_n_T_vpa_mu(self, time_idx=-1, species_idx=0):

        # gvmus(t, species, mu, vpa)
        gvmus  = self.ncdata.variables['gvmus'][time_idx,species_idx]
        vpa    = self.ncdata.variables['vpa'][:]
        mu     = self.ncdata.variables['mu'][:]
        time   = self.ncdata.variables['t'][time_idx]

        bmag = 1
        integrand_n = gvmus
        integrand_T = gvmus * (vpa[None,:]**2 + 2*mu[:,None]*bmag - 3/2)

        dmu  = np.abs( mu[1]- mu[0])
        dvpa = np.abs(vpa[1]-vpa[0])
        nmu  = np.sum( integrand_n, axis=1) * dvpa
        nvpa = np.sum( integrand_n, axis=0) * dmu
        Tmu  = np.sum( integrand_T, axis=1) * dvpa
        Tvpa = np.sum( integrand_T, axis=0) * dmu

        return nmu, nvpa, Tmu, Tvpa, mu, vpa, time

    def plot_contour_gvmu_vpa(self, fig=None, ax=None, time_idx=-1, vmin=None, vmax=None, logarithmic=False, cmap='inferno', plot_diff=False, zonal=False, nozonal=False, species_idx=0, kx_min=None, kx_max=None, dt_avg=None):
        
        if dt_avg is None:
            time_idx_eval = time_idx
        else:
            time_all = self.get_time_array()
            time_eval = time_all[time_idx]
            time_min = time_eval-dt_avg#/2
            time_max = time_eval#+dt_avg/2
            time_idx_min = self.get_time_idx(time_min)
            time_idx_max = self.get_time_idx(time_max)
            time_idx_eval = np.arange(time_idx_min, time_idx_max)
            dt_vals = np.gradient(time_all[time_idx_eval])

        # gvmus(t, species, mu, vpa)
        if kx_min is None and kx_max is None:

            if zonal:
                try:
                    gvmus  = self.ncdata.variables['gvmus_Z'][time_idx_eval,species_idx]
                except:
                    gvmus_tot  = self.ncdata.variables['g2_vs_vpamus'][time_idx_eval,species_idx]
                    gvmus_NZ   = self.ncdata.variables['g2nozonal_vs_vpamus'][time_idx_eval,species_idx]
                    gvmus = gvmus_tot - gvmus_NZ
            else:
                try:
                    gvmus  = self.ncdata.variables['gvmus'][time_idx_eval,species_idx]
                except:
                    if nozonal:
                        gvmus  = self.ncdata.variables['g2nozonal_vs_vpamus'][time_idx_eval,species_idx]
                    else:
                        gvmus  = self.ncdata.variables['g2_vs_vpamus'][time_idx_eval,species_idx]

        else:
            if kx_min is None:
                kx_min = 0
            if kx_max is None:
                kx_max = 1e20

            kx = self.ncdata.variables['kx'][:]

            if zonal:
               gkxvmus_tot = self.ncdata.variables['g2_vs_kxvpamus'][time_idx_eval,species_idx]
               gkxvmus_NZ  = self.ncdata.variables['g2nozonal_vs_kxvpamus'][time_idx_eval,species_idx]
               gkxvmus = gkxvmus_tot - gkxvmus_NZ
            elif nozonal:
               gkxvmus     = self.ncdata.variables['g2nozonal_vs_kxvpamus'][time_idx_eval,species_idx]
            else:
               gkxvmus     = self.ncdata.variables['g2_vs_kxvpamus'][time_idx_eval,species_idx]

            # Sum over kx's within desired range 
            if dt_avg is None:
                gvmus = np.sum(gkxvmus[  :,:, ( (np.abs(kx) >= kx_min) & (np.abs(kx) <= kx_max) )], axis=2)
            else:
                gvmus = np.sum(gkxvmus[:,:,:, ( (np.abs(kx) >= kx_min) & (np.abs(kx) <= kx_max) )], axis=3)

        vpa    = self.ncdata.variables['vpa']
        mu     = self.ncdata.variables['mu']
        time   = self.ncdata.variables['t'][time_idx]

        if dt_avg is not None:
            gvmus = np.sum(gvmus*dt_vals[:,None,None], axis=0)/np.sum(dt_vals)

        if plot_diff:
            gvmus_init = self.ncdata.variables['gvmus'][0,species_idx]
            gvmus = gvmus - gvmus_init

        X, Y = np.meshgrid(vpa, mu)
        Z = gvmus

        if fig is None and ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        if logarithmic:
            Z = np.abs(Z)

        if vmin is None:
            vmin = Z.min()
        if vmax is None:
            vmax = Z.max()

        if vmin=='symm' or vmax=='symm':
            vmax = np.abs(np.nanmax(Z))
            if not logarithmic:
                vmin = -vmax
            else:
                if Z.min() < 1e-15:
                    Z = Z+1e-14
                    vmax = vmax+1e-14
                vmin = vmax/1e4

        if logarithmic:
            try:
                im = ax.pcolormesh(X, Y, np.abs(Z), norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
                #im = ax.contourf(X, Y, np.abs(Z), norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap, levels=100)
            except:
                logarithmic = False

        if not logarithmic:
            im = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)

        ax.set_xlabel(r"$v_\parallel/v_T$")
        ax.set_ylabel(r"$\mu B_\mathrm{max}/T$")
        fig.suptitle(r"$t v_T/a = %.2f$" % (time))

        return fig, ax, im

    def plot_contour_gzvs(self, fig=None, ax=None, time_idx=-1, vmin=None, vmax=None, logarithmic=False, cmap='inferno', plot_diff=False, zonal=False, nozonal=False):
        

        # gzvs(t, species, vpa, zed, tube)
        try:
            gzvs_tot = self.ncdata.variables['gzvs'][time_idx,0, :, :, 0]
        except:
            gzvs_tot = self.ncdata.variables['g2_vs_zvpas'][time_idx,0, :, :, 0]
            gzvs_NZ  = self.ncdata.variables['g2nozonal_vs_zvpas'][time_idx,0, :, :, 0]

        if zonal:
            gzvs = gzvs_tot-gzvs_NZ
        elif nozonal:
            gzvs = gzvs_NZ
        else:
            gzvs = gzvs_tot

        vpa  = self.ncdata.variables['vpa']
        zed       = self.ncdata.variables['zed'][:]
        time = self.ncdata.variables['t'][time_idx]

        if plot_diff:
            gzvs_init = self.ncdata.variables['gzvs'][0,0, :, :, 0]
            gzvs = gzvs - gzvs_init

        X, Y = np.meshgrid(vpa, zed)
        Z = gzvs.T

        if fig is None and ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        if vmin=='symm' or vmax=='symm':
            vmax = np.abs(Z.max())
            if not logarithmic:
                vmin = -vmax
            else:
                if Z.min() < 1e-15:
                    Z = Z+1e-14
                    vmax = vmax+1e-14
                vmin = vmax/1e4


        if vmin is None:
            vmin = Z.min()
        if vmax is None:
            vmax = Z.max()

        if logarithmic:
            im = ax.pcolormesh(X, Y, np.abs(Z), norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap=cmap)
        else:
            im = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)

        ax.set_xlabel(r"$v_\parallel/v_T$")
        ax.set_ylabel(r"$\zeta$")
        ax.set_title(r"$t v_T/a = %.2f$" % (time))

        return fig, ax, im

    def evolve_markers_2D(self, t_min=0, t_max=np.infty, x0=[0], y0=[0], only_zonal_vEx=False, only_zonal_vEy=False, remove_zonal=False, zed_val=0, nx=None, ny=None, kxmax_filter=-1):

        # Time interval
        time_all = self.get_time_array()
        time_idx_min = np.argmin(np.abs(time_all-t_min))
        time_idx_max = np.argmin(np.abs(time_all-t_max))
        time_eval = time_all[time_idx_min:time_idx_max]

        Nmarkers = len(x0)
        assert(len(x0)==len(y0))
        x_t_vals = np.zeros((Nmarkers, len(time_eval)))
        y_t_vals = np.zeros((Nmarkers, len(time_eval)))
        x_t_vals[:,0] = x0[:]
        y_t_vals[:,0] = y0[:]

        # Evaluate velocity as a function of (x,y,t)
        for i_t in range(len(time_eval)-1):
            print("Marker evolution: time step %i/%i..." % (1+i_t, len(time_eval)), end="\r")

            # Evaluate velocity field
            vEx_x_y, x, y, _ = self.get_quantity_x_y(quantity="phi", zed_val=zed_val, time_idx=time_idx_min+i_t, remove_zonal=remove_zonal, only_zonal=only_zonal_vEx, ky_order=1, nx=nx, ny=ny, kxmax_filter=kxmax_filter)
            vEx_x_y = -vEx_x_y
            vEy_x_y, x, y, _ = self.get_quantity_x_y(quantity="phi", zed_val=zed_val, time_idx=time_idx_min+i_t, remove_zonal=remove_zonal, only_zonal=only_zonal_vEy, kx_order=1, nx=nx, ny=ny, kxmax_filter=kxmax_filter)

            # Shift for easier periodicity
            x = x-x[0]
            y = y-y[0]

            # Interpolating function 
            vEx_x_y_interp = interp2D(points=(x,y), values=vEx_x_y)
            vEy_x_y_interp = interp2D(points=(x,y), values=vEy_x_y)

            # Time step
            dt = time_eval[i_t+1]-time_eval[i_t]
            for i_m in range(Nmarkers):
                x_per = x_t_vals[i_m,i_t] % x[-1]
                y_per = y_t_vals[i_m,i_t] % y[-1]
                dx = dt*vEx_x_y_interp( (x_per, y_per) )
                dy = dt*vEy_x_y_interp( (x_per, y_per) )
                x_t_vals[i_m,i_t+1] = x_t_vals[i_m,i_t] + dx
                y_t_vals[i_m,i_t+1] = y_t_vals[i_m,i_t] + dy

        return time_eval, x_t_vals, y_t_vals

def plot_y_over_zed(ax, zed, y, ylabel=None, label=None, set_xlim=False, ls=None, color=None, no_xticks=False, lw=1, norm=False, xlim=None, alpha=1):
    if norm:
        y = y/y.max()

    if xlim is not None:
        y   = y[  zed >= xlim[ 0]]
        zed = zed[zed >= xlim[ 0]]
        y   = y[  zed <= xlim[-1]]
        zed = zed[zed <= xlim[-1]]

    ax.plot(zed, y, label=label, ls=ls, c=color, lw=lw, alpha=alpha)
    ax.set_ylabel(ylabel)
    #if set_xlim:
    #    #ax.set_xlim([-np.pi, np.pi])
    #    #ax.set_xticks([])
    #    ax.set_xticks([-np.pi,0,np.pi])
    #    ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

    if no_xticks:
        #ax.set_xticks([-np.pi,0,np.pi])
        ax.set_xticklabels([])

    else:
        ax.set_xlabel(r"$\chi$")

def get_avg_stddev_timetrace(time, quantity, timeavg, timemax=None):
    if timemax is None:
        timemax = max(time)
    timemin = max(0, timemax-timeavg)

    quantity_to_avg = quantity[(timemin<time) & (time<timemax)]
    time_to_avg     = time[(timemin<time) & (time<timemax)]
    if len(time_to_avg) < 2:
        time_to_avg     = time[-2:]
        quantity_to_avg = quantity[-2:]

    Delta_t = np.gradient(time_to_avg)

    quantity_mean = np.mean(quantity_to_avg*Delta_t)/np.mean(Delta_t)
    quantity_std  = np.sqrt( np.mean( Delta_t*(quantity_to_avg-quantity_mean)**2 )/np.mean(Delta_t) )

    return quantity_mean, quantity_std

def get_convergence_quantity(time, quantity, timeavg_array, timemax=None):
    quantity_avg = np.zeros_like(timeavg_array)
    quantity_std = np.zeros_like(timeavg_array)
    for i_avg, timeavg in enumerate(timeavg_array):
        quantity_avg[i_avg], quantity_std[i_avg] = get_avg_stddev_timetrace(time, quantity, timeavg, timemax)

    return quantity_avg, quantity_std

def plot_convergence_quantity(time, quantity, timeavg_array, timemax=None, fig=None, ax=None, c=None, marker=None, ls=None, label=None):

    if ax is None:
       fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    quantity_avg, quantity_std = get_convergence_quantity(time, quantity, timeavg_array, timemax)

    ax.errorbar(timeavg_array, quantity_avg, yerr=quantity_std, label=label, c=c, marker=marker, ls=ls)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_xlim(xmin=0)

    return fig, ax

####### Get growth rate from time trace of quantity
def extract_growth_rate(time, quantity):

    assert(len(time)==len(quantity))

    # Instantaneous growth rate, assuming f~f0*exp(i*omega*t)
    gamma = np.zeros(len(time), dtype='complex')
    gamma[0] = 1/(time[1]-time[0])*np.log(quantity[1]/quantity[0])
    gamma[1:] = 1/(time[1:]-time[:-1])*np.log(quantity[1:]/quantity[:-1])

    return gamma

# List of "true" (actual flux-surface) flux_norm values for some configurations
def get_true_flux_norm(configuration):

    if configuration == "precise QA":
        # from /scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2022-04-04_ITG_scan_precise_QA/configuration/evaluate_true_flux_surface_averages/precise_QA
        return 1.2911204260996358e+00

    elif configuration == "precise QH":
        # from /scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2022-04-04_ITG_scan_precise_QA/configuration/evaluate_true_flux_surface_averages/precise_QH
        return 1.3731390294068104e+00

# Fourier transform to real space
def get_fft_real_space(f_kx_ky, kx, ky, nx=None, ny=None):

    #f_kx_ky[0,0] = 1

    if nx is None:
        nx = len(kx)
    else:
        # Padding in middle of array if necessary
        f_kx_ky = np.array( np.concatenate( (f_kx_ky[:int((len(kx)+1)/2),:], np.zeros((nx-len(kx), len(ky))), f_kx_ky[int((len(kx)+1)/2):,:]), axis=0) )
        
    if ny is None:
        ny = len(ky)

    f_x_y = np.fft.irfft((np.fft.ifft(f_kx_ky, axis=0)), n=2*ny-1, axis=1)*(nx)*(ny)

    #f_x_y = np.fft.irfft((np.fft.ifft(f_kx_ky, axis=0, n=n)), n=2*len(ky)-1, axis=1)
    #f_x_y = np.fft.irfft((np.fft.ifft(f_kx_ky, axis=0)), n=2*ny-1, axis=1)*(nx)*(2*ny-1)
    #f_x_y = np.fft.irfft((np.fft.ifft(f_kx_ky, axis=0)), n=2*ny-1, axis=1)*(nx/len(kx))*((2*ny-1)/(2*len(ky)-1))

    #x = np.fft.fftshift(np.fft.fftfreq(n,d=(kx[1]-kx[0])/(2*np.pi)))
    #x = np.fft.fftshift(np.fft.fftfreq(len(kx),d=(kx[1]-kx[0])/(2*np.pi)))

    if len(kx) == 1:
        xmax = np.pi
    else:
        xmax = np.pi/(kx[1]-kx[0])
    x    = np.linspace(-xmax, xmax, nx, endpoint=False)

    if len(ky) == 1:
        ymax = np.pi
    else:
        ymax = np.pi/(ky[1]-ky[0])
    y = np.linspace(-ymax, ymax, 2*ny-1, endpoint=False)

    #print(ny/(2*ny-1))
    #VERIFY INTEGRAL OVER dxdy = f_{kx=0, ky=0}*L_x*L_y/2
    #integral = np.sum(f_x_y)*(x[1]-x[0])*(y[1]-y[0])
    #print("\nIntegral_xy: %e, k=0: %e" % (integral, np.real(f_kx_ky[0,0])*(x[-1]-x[0])*(y[-1]-y[0])*ny/(ny-1)/2))

    return f_x_y, x, y
    #return f_x_y*(2*np.pi)**2, x, y
    #return f_x_y * len(kx)*len(ky)*(2*np.pi)**2, x, y
    #return f_x_y * len(kx)*len(ky)/(4*xmax*ymax), x, y


# 1D Fourier transform real quantity to k-space
def get_fft_k(f_x, x):
    kx = np.fft.rfftfreq(len(f_x), d=(x[1]-x[0])/(2*np.pi))
    kmin = kx[1]-kx[0]
    f_kx = np.fft.rfft(f_x)/len(x)

    #kx = np.fft.fftfreq(len(f_x), d=(x[1]-x[0])/(2*np.pi))
    #kmin = kx[1]-kx[0]
    #f_kx = np.fft.fft(f_x)/(len(x))
    #f_kx = np.fft.fft(f_x)/(2*len(x))

    ##VERIFY INTEGRAL OVER dx f(x) = L_x f_{kx=0}
    #integral_x = np.sum(f_x)*(x[1]-x[0])
    #print("\nIntegral_x: %e, kx=0: %e" % (integral_x, np.real(f_kx[0])*(x[-1]-x[0])))

    return f_kx, kx

# Wigner transform
def plot_Wigner_t_omega(f_t, ts, fig=None, ax=None):

    tfr = WignerVilleDistribution(f_t, timestamps=ts)
    tfr_wvd, t_wvd, f_wvd = tfr.run()

    dt = ts[1]-ts[0]
    f_wvd = np.fft.fftshift(np.fft.fftfreq(tfr_wvd.shape[0], d=2 * dt))
    df_wvd = f_wvd[1]-f_wvd[0]  # the frequency step in the WVT
    sig_Wig = np.fft.fftshift(np.abs(tfr_wvd), axes=0)
    
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

    im = ax.imshow(sig_Wig, aspect='auto', origin='lower',\
           extent=((ts[0] - dt/2), (ts[-1] + dt/2),\
        (f_wvd[0]-df_wvd/2)*2*np.pi, (f_wvd[-1]+df_wvd/2)*2*np.pi ))

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\omega$")
#    #    return 

#        tfr.plot(kind='contour', show_ft=True)

    return im, fig, ax


# Wigner transform
def get_Wigner_x_kx(f_x):


    tfr = WignerVilleDistribution(f_x)
    tfr.run()
    tfr.plot(kind='contour', show_ft=True)

# Laplace transform t->omega
def Laplace_transform(times, f_t, omega_r, omega_i):

    X, Y = np.meshgrid(omega_r, omega_i)

    f_t_interp_real = interp(times, np.real(f_t))
    f_t_interp_imag = interp(times, np.imag(f_t))

    def Laplace_integrand_real(t, omega):
        return np.real( (f_t_interp_real(t)+1j*f_t_interp_imag(t)) * np.exp(1j*omega*t) )
    def Laplace_integrand_imag(t, omega):
        return np.imag( (f_t_interp_real(t)+1j*f_t_interp_imag(t)) * np.exp(1j*omega*t) )

    omega = X + 1j*Y
    Z = np.zeros_like(omega)
    
    N_r = len(omega_r)
    N_i = len(omega_i)
    for i_r in range(N_r):
        for i_i in range(N_i):
            print("Laplace: evaluating omega %i/%i..." % (1+i_i+i_r*N_i, N_i*N_r), end="\r")
            Z_real = integrate.quad(Laplace_integrand_real, times[0], times[-1], args=(omega[i_i,i_r],))[0]
            Z_imag = integrate.quad(Laplace_integrand_imag, times[0], times[-1], args=(omega[i_i,i_r],))[0]
            #print(Z_real)
            Z[i_i, i_r] = (Z_real + 1j*Z_imag)#*np.exp(np.imag(omega[i_i,i_r])*times[-1])

    return X, Y, Z

def estimate_omega_gamma_signal(f_t, t, ignore_omega=False):
    #argmaxs = argrelextrema(np.abs((f_t)), np.greater)[0]
    argmaxs = argrelextrema(np.abs(np.real(f_t)), np.greater)[0]
    if len(argmaxs)<=1 or ignore_omega:
        omega = 0
        gamma = np.log(np.abs(f_t)[-1]/np.abs(f_t)[0])/(t[-1]-t[0])
        omega_stddev = 0
        gamma_stddev = 0
    else:
        t_maxs = t[argmaxs]
        f_maxs = f_t[argmaxs]
        avg_period = np.mean(np.gradient(t_maxs))
        period_stddev = np.std(np.gradient(t_maxs))
        omega = 2*np.pi/avg_period/2
        omega_stddev = omega * period_stddev/avg_period
        gamma_all = np.log(np.abs(f_maxs[1:])/np.abs(f_maxs[:-1]))/np.diff(t_maxs)
        print(gamma_all)
        gamma = np.mean(gamma_all)
        gamma_stddev = np.std(gamma_all)

    return omega, gamma, omega_stddev, gamma_stddev

#######  Plot qflx(tprim) from runs of given directory
def plot_qflx_tprim_qinp_dir(dirname, filename, rundir_str_exclude=None, rundir_str_beg="run_", rundir_str_end="", species_idx=0, time_max=1e10, time_avg=None, norm=True, configuration=None, fig=None, ax=None, label=None, ls=None, c=None, lw=None, marker='.', code="stella", scale_tprim=1, scale_Q=1, tprim_qinp="both", load_from_file=False, tprim_max=None):

    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(9,5))

    rundirs = glob(dirname+"/"+rundir_str_beg+"*"+rundir_str_end)

    if rundir_str_exclude is not None:
        rundirs_new = []
        for i_dir, rundir in enumerate(rundirs):
            if rundir_str_exclude not in rundir:
                rundirs_new.append(rundir)
        rundirs = rundirs_new
    
    Nr_dirs = len(rundirs)
    tprim_qinp_vals = np.zeros(Nr_dirs)
    qflx_vals  = np.zeros(Nr_dirs)
    qflx_stddev_vals  = np.zeros(Nr_dirs)

    for i_dir in range(Nr_dirs):
        filename_data = rundirs[i_dir]+"/data_qflx_" + tprim_qinp + ".dat"

        if load_from_file and exists(filename_data):
            # Load from file if desired
            tprim_qinp_vals[i_dir], qflx_vals[i_dir], qflx_stddev_vals[i_dir] = np.loadtxt(filename_data)
        else:

            try:
                diagObj = stellaDiagnostics(rundirs[i_dir]+"/"+filename, code=code)
                _, _, qflx, time = diagObj.get_fluxes_over_time(species_idx=species_idx, norm=norm, configuration=configuration)#, delta_t=1.5*time_avg)
                qflx = qflx[time<time_max]
                time = time[time<time_max]
    
                if time_avg == "auto":
                    # Time average between t=2*t(Qmax) and t_end
                    idx_Qmax = np.argmax(qflx)
                    double_t_Qmax = 2*time[idx_Qmax]
                    time_start_avg = min( double_t_Qmax, time[-2])
                    time_avg = time[-1]-time_start_avg
                    print("Automatically determine t_avg for " + rundirs[i_dir] +" = %e" % (time_avg))
    
                if time_avg is not None:
                    qflx, qflx_stddev = get_avg_stddev_timetrace(time, qflx, time_avg)
                else:
                    qflx = qflx[-1]
                    qflx_stddev = 0
    
                qflx        = qflx       *scale_Q
                qflx_stddev = qflx_stddev*scale_Q
        
                if code == "GX":
                    tprim = diagObj.ncdata['Inputs']['Species']['T0_prime'][species_idx]
                    geo_data = np.loadtxt(diagObj.geo_file, max_rows=1, skiprows=1)
                    qinp  = geo_data[-1]
                else:
                    tprim = diagObj.ncdata.variables['tprim'][species_idx]
                    if code == "GS2":
                        qinp  = diagObj.ncdata.variables['qval'].getValue()
                    else:
                        qinp  = diagObj.ncdata.variables['q'].getValue()
        
                tprim = tprim*scale_tprim
    
                qflx_vals[i_dir]  = qflx
                qflx_stddev_vals[i_dir]  = qflx_stddev
    
                if tprim_qinp == "tprim":
                    tprim_qinp_vals[i_dir] = tprim
                elif tprim_qinp == "qinp":
                    tprim_qinp_vals[i_dir] = qinp
                elif tprim_qinp == "both":
                    tprim_qinp_vals[i_dir] = qinp*tprim

                np.savetxt(filename_data, (tprim_qinp_vals[i_dir], qflx_vals[i_dir], qflx_stddev_vals[i_dir]))
    
            except:
                print("Failed reading " + rundirs[i_dir])
                continue

    try:
        idx_sort = np.argsort(tprim_qinp_vals)
        tprim_qinp_vals = tprim_qinp_vals[idx_sort]
        qflx_vals  =  qflx_vals[idx_sort]
        qflx_stddev_vals  =  qflx_stddev_vals[idx_sort]
    
        if tprim_max is not None:
            qflx_vals        = qflx_vals[       tprim_qinp_vals<tprim_max*scale_tprim]
            qflx_stddev_vals = qflx_stddev_vals[tprim_qinp_vals<tprim_max*scale_tprim]
            tprim_qinp_vals  = tprim_qinp_vals[ tprim_qinp_vals<tprim_max*scale_tprim]

        if len(qflx_vals)>=1:
            ax.errorbar(tprim_qinp_vals, qflx_vals, yerr=qflx_stddev_vals, ls=ls, label=label, c=c, lw=lw, marker=marker)
        #ax.plot(tprim_vals, qflx_vals, ls=ls, label=label, c=c, lw=lw, marker=marker)
    except:
        print("It seems like none of the directories in " + dirname + " could be read.")

    return fig, ax

####### Get statistics from a time signal
def get_statistics(time, f_t, dt):

    # Make sure dt is not smaller than minimum timestep size
    dt_min = 2*np.min(time[1:]-time[:-1])
    if dt < dt_min:
        print("dt for statistics was taken to be too small.")
        dt = dt_min

    # Get data on equal time intervals
    time_intervalled = np.arange(time[0]+dt/2, time[-1]-dt/2, dt)
    f_t_intervalled = np.zeros_like(time_intervalled)
    for i_interval, time_interval in enumerate(time_intervalled):
        time_min_integrate = time_interval-dt/2
        time_max_integrate = time_interval+dt/2
        time_idx_min = np.argmin(np.abs(time-time_min_integrate))
        time_idx_max = np.argmin(np.abs(time-time_max_integrate))
        f_t_intervalled[i_interval] = np.mean(f_t[time_idx_min:time_idx_max])

    # Compute mean, rms, etc
    f_t_mean = np.mean(f_t_intervalled)
    f_t_rms = np.sqrt( np.mean(f_t_intervalled**2) )
    f_t_std = np.std(f_t_intervalled)

    return time_intervalled, f_t_intervalled, f_t_mean, f_t_rms, f_t_std

###### Get correlation function from a 1D signal
def get_correlation_func_1D(x, y, ref_point="middle", dx_max=None, Nr_dx=10, Nr_x_ref=10):
    assert(len(x)==len(y))

    if ref_point=="middle":
        corr_func = np.zeros_like(x)
        idx_mid = int(len(x)/2)
        y_mid   = y[idx_mid]
        mult_mid = np.conj(y_mid)/np.abs(y_mid)**2
        Delta_x   = x-x[idx_mid]

        for i in range(len(x)):
            corr_func[i] = np.real(y[i]*mult_mid)

    elif ref_point=="avg":
        # Interpolate in case data is not equally spaced
        y_interp_real = interp(x, np.real(y))
        y_interp_imag = interp(x, np.imag(y))

        # Determine dx_max and ensure it is lower than 1/2 length of data
        if dx_max is None:
            dx_max = (x[-1]-x[0])/2

        Delta_x = np.linspace(-dx_max, dx_max, Nr_dx, endpoint=True)
        corr_func = np.zeros_like(Delta_x)

        for i_dx, dx in enumerate(Delta_x):

            xmin_ref = max( x[0],  x[0] -dx)
            xmax_ref = min( x[-1], x[-1]-dx)
            xvals_ref = np.linspace(xmin_ref*1.001, xmax_ref*0.999, Nr_x_ref)

            y_interp_xvals_ref = y_interp_real(xvals_ref) + 1j*y_interp_imag(xvals_ref)
            y_interp_dx        = y_interp_real(xvals_ref+dx) + 1j*y_interp_imag(xvals_ref+dx)
            norm = np.mean( np.abs(y_interp_xvals_ref)**2 )
            corr_func[i_dx] = np.mean( np.real(y_interp_xvals_ref * np.conj(y_interp_dx))) / norm

    return Delta_x, corr_func


###### Get correlation function from a 2D signal, relative to middle point
def get_correlation_func_2D(x1, x2, y, idx_ref1=None, idx_ref2=None, ref_point="middle", x2_window=None):
#ref_point="middle", dx1_max=None, dx2_max=None, Nr_dx1=10, Nr_dx2=10, Nr_x1_ref=10, Nr_x2_ref=10):
    assert(len(x1)==np.shape(y)[0])
    assert(len(x2)==np.shape(y)[1])
    corr_func = np.zeros_like(y)

    if ref_point=="single":
        if idx_ref1 is None:
            idx_ref1 = int(len(x1)/2)
        if idx_ref2 is None:
            idx_ref2 = int(len(x2)/2)
        y_ref   = y[idx_ref1, idx_ref2]
        mult_ref = np.conj(y_ref)/np.abs(y_ref)**2
        
        Delta_x1  = x1-x1[idx_ref1]
        Delta_x2  = x2-x2[idx_ref2]
        
        for i1 in range(len(x1)):
            for i2 in range(len(x2)):
                corr_func[i1, i2] = np.real(y[i1,i2]*mult_ref)

    elif ref_point=="avg1":
        # Note: assumes equally spaced data in first index, and periodic (e.g. x or y)
        if idx_ref2 is None:
            idx_ref2 = int(len(x2)/2)

        idx_mid1  = int(len(x1)/2)+1
        Delta_x1  = x1-np.mean(x1)
        Delta_x2  = x2-x2[idx_ref2]
        corr_func = np.zeros((len(x1), len(x2)))
        yvals_ref = y[:,idx_ref2]
        norm_ref  = np.mean( np.abs(yvals_ref)**2 )

        for i_Delta_x1 in range(len(Delta_x1)):
            idxs_1 = ( idx_mid1 + np.arange(len(Delta_x1)) + i_Delta_x1) % len(Delta_x1)
            for i_Delta_x2 in range(len(Delta_x2)):
                corr_func[i_Delta_x1, i_Delta_x2] = np.mean( np.real(y[idxs_1, i_Delta_x2]*np.conj(yvals_ref)) ) / norm_ref


    elif ref_point=="avg":
        # Note: assumes equally spaced data, and periodic in first index
        idx_mid1  = int(len(x1)/2)+1
        idx_mid2  = int(len(x2)/2)+1
        Delta_x1  = x1-np.mean(x1)
        Delta_x2  = x2-x2[0]
        idxs2 = np.arange(len(Delta_x2))
        idxs2_window = idxs2[Delta_x2<=x2_window]
        idxs2_ref    = idxs2[Delta_x2<=Delta_x2[-1]-x2_window]
        #print(len(idxs2))
        #print(len(idxs2_window))
        #print(len(idxs2_ref))
        assert(len(idxs2)>=len(idxs2_window)+len(idxs2_ref))
        corr_func = np.zeros((len(x1), len(idxs2_window)))


        for i_Delta_x1 in range(len(Delta_x1)):
            idxs_1 = ( idx_mid1 + np.arange(len(Delta_x1)) + i_Delta_x1) % len(Delta_x1)

            for i_Delta_x2 in range(len(idxs2_window)):
                corr_func[i_Delta_x1, i_Delta_x2] = np.sum( np.real(y[idxs_1[:,None],idxs2_ref[None,:]+i_Delta_x2]*np.conj(y[:,idxs2_ref])) ) / np.sum( np.abs(y[:,idxs2_ref])**2 )

        Delta_x2 = x2[idxs2_window]-x2[0]

#    elif ref_point == "interpolate":
#        # Note: interpolates in both indices, for real input
#        #from scipy.interpolate import RectBivariateSpline as interp2d
#        from scipy.interpolate import RegularGridInterpolator as interp2d
#        from scipy.integrate   import dblquad  as dblquad
#
#        Delta_x1  = x1-np.mean(x1)
#        Delta_x2  = np.linspace(0, x2_window, len(x2))
#        corr_func = np.zeros((len(Delta_x1), len(Delta_x2)))
#
#        # Create interpolation function
#        #y_interp = interp2d(x1, x2, np.real(y))
#        y_interp = interp2d(points=[x1, x2], values=np.real(y))
#
#        def integrand_denominator_C(x1val, x2val):
#            x1_sv = x1[0] + (x1val-x1[0])%(x1[-1]-x1[0])
#            return y_interp(x1_sv, x2val)**2
#
#        # Evaluate correlation
#        for i_Delta_x1, Delta_x1_val in enumerate(Delta_x1):
#            for i_Delta_x2, Delta_x2_val in enumerate(Delta_x2):
#                print("Evaluating Deltax1-x2 %5i/%5i" % (i_Delta_x1*len(Delta_x2)+i_Delta_x2+1, len(Delta_x2)*len(Delta_x1)))
#                def integrand_numerator_C(x2val, x1val):
#                    x1_sv = x1[0] + (x1val-x1[0])%(x1[-1]-x1[0])
#                    x1_sv_Delta = x1[0] + (x1val+Delta_x1_val-x1[0])%(x1[-1]-x1[0])
#                    #assert(x2val >= 0)
#                    #assert(x2val <= x2[-1])
#                    #assert(x2val+Delta_x2_val >= 0)
#                    #assert(x2val+Delta_x2_val <= x2[-1])
#                    #print(x1_sv)
#                    #print(x1_sv_Delta)
#                    #print(x1[0])
#                    #print(x1[-1])
#                    #print(x2val)
#                    #print(x2val+Delta_x2_val)
#                    #print(x2[0])
#                    #print(x2[-1])
#                    return y_interp([x1_sv, x2val])*y_interp([x1_sv_Delta, x2val+Delta_x2_val])
#
#                corr_func[i_Delta_x1, i_Delta_x2] = dblquad( integrand_numerator_C, a=x1[0], b=x1[-1], gfun=x2[0], hfun=x2[-1]-Delta_x2_val, epsrel=1e-2)[0]

            

#    elif ref_point=="avg":
#
#        # Determine dx_max and ensure it is lower than 1/2 length of data
#        if dx1_max is None:
#            dx1_max = (x1[-1]-x1[0])/2
#        if dx2_max is None:
#            dx2_max = (x2[-1]-x2[0])/2
#
#        Delta_x = np.linspace(-dx_max, dx_max, Nr_dx, endpoint=True)
#        corr_func = np.zeros_like(Delta_x)
#
#        for i_dx1, dx1 in enumerate(Delta_x1):
#
#            x1min_ref = max( x1[0],  x1[0] -dx1)
#            x1max_ref = min( x1[-1], x1[-1]-dx2)
#            x1vals_ref = np.linspace(xmin_ref*1.001, xmax_ref*0.999, Nr_x_ref)
#
#            # Interpolate in case data is not equally spaced
#            X_2D, _ = np.meshgrid(x1, x2)
#
#            y_interp_real_ref = interpn(X_2D, np.real(y), xvals_ref)
#            y_interp_imag_ref = interpn(X_2D, np.imag(y), 
#
#
#            y_interp_xvals_ref = y_interp_real(xvals_ref) + 1j*y_interp_imag(xvals_ref)
#            y_interp_dx        = y_interp_real(xvals_ref+dx) + 1j*y_interp_imag(xvals_ref+dx)
#            norm = np.mean( np.abs(y_interp_xvals_ref)**2 )
#            corr_func[i_dx] = np.mean( np.real(y_interp_xvals_ref * np.conj(y_interp_dx))) / norm

    return Delta_x1, Delta_x2, corr_func

