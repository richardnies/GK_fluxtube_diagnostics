import numpy as np
import stellaDiagnostics
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from os.path import exists
import scipy.special as specialfunc

class loadStellaScan:

    def __init__(self, filenames_base, labels=None, codes=None, ls=None):

        self.filenames_base = filenames_base
        self.Nr_files       = len(filenames_base)

        # Initialise StellaDiagnostics objects
        self.list_dataObj = []
        self.list_labels  = []
        self.list_ls  = []
        for i_file, filename_base in enumerate(self.filenames_base):
            if codes is None:
                code = "stella"
            else:
                code = codes[i_file]

            if ls is None:
                self.list_ls.append("-")
            else:
                self.list_ls.append(ls[i_file])

            try:
                dataObj = stellaDiagnostics.stellaDiagnostics(filename_base, code=code)
                self.list_dataObj.append( dataObj )
                if labels is None:
                    self.list_labels.append(None)
            except:
                print("Couldn't load " + filename_base)
                if labels is not None:
                    del labels[i_file]
                continue

        if labels is not None:
            self.list_labels = labels

        assert(len(self.list_dataObj) == len(self.list_labels))

    #def load_omegas(self, timestep=-1, om_avg=True, delta_t_avg=None, t_val=None, check_convergence=False):
    def load_omegas(self, timestep=-1, om_avg=True, delta_t_avg=None, t_val=None, check_convergence=True):

        self.om_time = [] 
        self.ky      = [] 
        self.kx      = [] 
        self.omega_r = [] 
        self.omega_i = [] 

        for i, dataObj in enumerate(self.list_dataObj):
            try:
                om_time, ky, kx, omega_r, omega_i = \
                         dataObj.read_data_omega_k(timestep=timestep, om_avg=om_avg, delta_t_avg=delta_t_avg, t_val=t_val, check_convergence=check_convergence)
            except:    
                om_time = ky = kx = omega_r = omega_i = [[0]]

            self.om_time.append(om_time)
            self.ky.append(ky)
            self.kx.append(kx)
            self.omega_r.append(omega_r)
            self.omega_i.append(omega_i)

        self.om_time = np.array(self.om_time)
        self.ky      = np.array(self.ky)
        self.kx      = np.array(self.kx)
        self.omega_r = np.array(self.omega_r)
        self.omega_i = np.array(self.omega_i)

        return self.om_time, self.ky, self.kx, self.omega_r, self.omega_i

    def load_phi_vs_zed(self):

        list_phi_vs_t = []
        list_zed      = []
        for i, dataObj in enumerate(self.list_dataObj):

            phi_vs_t, zed = dataObj.read_phi_vs_zed()
            list_phi_vs_t.append(phi_vs_t)
            list_zed.append(zed)

        return list_phi_vs_t, list_zed

    def plot_comparison_flux_tube_geometry(self, plot_phi=False, zed_times_nfield_periods=False, load_from_nc=True, normalise_bmag=False, colors=None, fig=None, axs=None, norm_gradpar=False):

        if fig is None and axs is None:
            fig, axs = plt.subplots(nrows=3,ncols=4, figsize=(24,10))
            #fig, axs = plt.subplots(nrows=3,ncols=4, figsize=(24,18))
            plt.subplots_adjust(hspace=0,left=0.08,right=0.95,top=0.9,bottom=0.1,wspace=0.45)

        for i, dataObj in enumerate(self.list_dataObj):
            if colors is None:
                color = None
            else:
                color=colors[i]
            dataObj.plot_flux_tube_geometry(axs=axs, label=self.list_labels[i], plot_phi=plot_phi, zed_times_nfield_periods=zed_times_nfield_periods, load_from_nc=load_from_nc, normalise_bmag=normalise_bmag, color=color, ls=self.list_ls[i], norm_gradpar=norm_gradpar)

        return fig, axs

    def plot_omega_ky(self, fig=None, axs=None, label=None, ls=None, color=None, markersize=10, marker='o', gamma_min=-np.inf, delta_t_avg=None, t_val=None, kx_idx=0, check_convergence=True, rescale_vT=1, rescale_omega=1):

        try:
            omega = self.omega_r
            gamma = self.omega_i
            ky    = self.ky
        except:
            self.load_omegas(delta_t_avg=delta_t_avg, t_val=t_val, check_convergence=check_convergence)

        # Pick kx index with maximum gamma for each ky
        if kx_idx =="max":
            omega_r_plt = []
            omega_i_plt = []
            ky_plt      = []

            for i in range(np.shape(self.ky)[0]):
                for j in range(np.shape(self.ky)[1]):
                    idx_kx_max = np.argmax(self.omega_i[i,j])
                    print("kx_max = %.3f for ky = %.3f" % (self.kx[i,j,idx_kx_max], self.ky[i,j,idx_kx_max]))
                    omega_r_plt.append(self.omega_r[i,j,idx_kx_max])
                    omega_i_plt.append(self.omega_i[i,j,idx_kx_max])
                    ky_plt.append(self.ky[i,j,idx_kx_max])
                
        else:
            omega_r_plt = (self.omega_r[:,:,kx_idx]).flatten()
            omega_i_plt = (self.omega_i[:,:,kx_idx]).flatten()
            ky_plt      = (self.ky[:,:,kx_idx]).flatten()
            print("kx val = %.3f" % (self.kx[0,0,kx_idx]))

        # Order in ky
        idx_sort_ky = np.argsort(ky_plt)
        omega_r_plt = np.array(omega_r_plt)[idx_sort_ky]
        omega_i_plt = np.array(omega_i_plt)[idx_sort_ky]
        ky_plt      = np.array(ky_plt)[idx_sort_ky]

        if axs is None:
            fig, axs = plt.subplots(nrows=2,ncols=1, figsize=(12,9))

        axs[0].plot(ky_plt[omega_i_plt>gamma_min]/rescale_vT, omega_i_plt[omega_i_plt>gamma_min]*rescale_vT*rescale_omega, ls=ls, c=color, label=label, marker=marker, markersize=markersize, markerfacecolor='None')
        #axs[0].plot(self.ky[self.omega_i>gamma_min], self.omega_i[self.omega_i>gamma_min], ls=ls, c=color, label=label, marker=marker)
        axs[0].set_ylabel(r"$\gamma a/v_T$")
        axs[0].set_xticklabels([])

        axs[1].plot(ky_plt[omega_i_plt>gamma_min]/rescale_vT, omega_r_plt[omega_i_plt>gamma_min]*rescale_vT*rescale_omega, ls=ls, c=color, label=label, marker=marker, markersize=markersize, markerfacecolor='None')
        #axs[1].plot(self.ky[self.omega_i>gamma_min], self.omega_r[self.omega_i>gamma_min], ls=ls, c=color, label=label, marker=marker)
        axs[1].set_ylabel(r"$\omega_r a/v_T$")

        axs[1].set_xlabel(r"$k_y \rho_i$")

        return fig, axs, omega_r_plt, omega_i_plt, ky_plt

    def plot_omega_kx(self, axs=None, label=None, ls=None, color=None, marker='o', gamma_min=-np.inf, delta_t_avg=None, t_val=None, ky_idx=0):

        try:
            omega = self.omega_r
            gamma = self.omega_i
            kx    = self.kx
        except:
            self.load_omegas(delta_t_avg=delta_t_avg, t_val=t_val)

        # Pick ky index with maximum gamma for each kx
        if ky_idx =="max":
            omega_r_plt = []
            omega_i_plt = []
            kx_plt      = []

            for i in range(np.shape(self.kx)[0]): # Loop over directories
                for j in range(np.shape(self.kx)[1]): # Loop over kx values
                    idx_ky_max = np.argmax(self.omega_i[i])
                    omega_r_plt.append(self.omega_r[i,idx_ky_max,j])
                    omega_i_plt.append(self.omega_i[i,idx_ky_max,j])
                    kx_plt.append(self.kx[i,idx_ky_max,j])
                
        else:
            omega_r_plt = (self.omega_r[:,ky_idx,:]).flatten()
            omega_i_plt = (self.omega_i[:,ky_idx,:]).flatten()
            kx_plt      = (self.kx[:,ky_idx,:]     ).flatten()

        # Order in kx
        idx_sort_kx = np.argsort(kx_plt)
        omega_r_plt = np.array(omega_r_plt)[idx_sort_kx]
        omega_i_plt = np.array(omega_i_plt)[idx_sort_kx]
        kx_plt      = np.array(kx_plt)[idx_sort_kx]

        if axs is None:
            fig, axs = plt.subplots(nrows=2,ncols=1, figsize=(12,9))

        axs[0].plot(kx_plt[omega_i_plt>gamma_min], omega_i_plt[omega_i_plt>gamma_min], ls=ls, c=color, label=label, marker=marker)
        axs[0].set_ylabel(r"$\gamma a/v_T$")
        axs[0].set_xticklabels([])

        axs[1].plot(kx_plt[omega_i_plt>gamma_min], omega_r_plt[omega_i_plt>gamma_min], ls=ls, c=color, label=label, marker=marker)
        axs[1].set_ylabel(r"$\omega_r a/v_T$")

        axs[1].set_xlabel(r"$k_x \rho_i$")

        return axs


    def plot_contour_gamma_kx_ky(self, ax=None, delta_t_avg=None, t_val=None):

        try:
            omega = self.omega_r
            gamma = self.omega_i
            kx    = self.kx
        except:
            self.load_omegas(delta_t_avg=delta_t_avg, t_val=t_val)

        omega_i_plt = self.omega_i[0,:,:]
        kx_plt      = self.kx[0,0,:]
        ky_plt      = self.ky[0,:,0]

        # Order in kx
        idx_sort_kx = np.argsort(kx_plt)
        omega_i_plt = np.array(omega_i_plt)[:,idx_sort_kx]
        kx_plt      = np.array(kx_plt)[idx_sort_kx]

        X, Y = np.meshgrid(kx_plt, ky_plt)
        Z = omega_i_plt

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='magma', vmin=0)

        ax.set_title(r"$\gamma a/v_T$")
        ax.set_xlabel(r"$k_x \rho_i$")
        ax.set_ylabel(r"$k_y \rho_i$")

        return ax, im

    def plot_phi_vs_zed(self, ax=None, zed_times_nfield_periods=False):

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        for i, dataObj in enumerate(self.list_dataObj):
            dataObj.plot_phi_vs_zed(ax=ax, label=self.list_labels[i], zed_times_nfield_periods=zed_times_nfield_periods)

        return ax

    def plot_contour_phi_vs_zed_theta0(self, fig=None, ax=None, normalise_phi=False, logarithmic=False, vmin=None, vmax=None):

        # Load values
        list_phi, list_zed = self.load_phi_vs_zed()
        array_phi = np.asarray(list_phi)

        list_theta0 = []
        for i, dataObj in enumerate(self.list_dataObj):
            theta0 = dataObj.read_basic_params()['theta0'][0]
            list_theta0.append(theta0)

        # Normalise phi if desired
        if normalise_phi:
            for i_t in range(len(list_theta0)):
                array_phi[i_t, :] = np.abs(array_phi[i_t, :]) / np.max(np.abs(array_phi[i_t, :]))

        # Convert to plottable arrays
        X, Y = np.meshgrid(np.asarray(list_theta0), np.asarray(list_zed[0]) )
        Z = array_phi.T

        # Plot
        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,9))

        if vmin is None:
            vmin = Z.min()
        if vmax is None:
            vmax = Z.max()

        if logarithmic:
            im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto', cmap='plasma')
        else:
            im = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading='auto', cmap='plasma')

        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\zeta$")

        return fig, ax, im

    def plot_phi_k_spectrum(self, plot_kx, fig=None, ax=None, time_idx=-2, ls_list=None, marker_list=None, color_list=None, tprim_norm_list=None, qinp_norm_list=None, xdrift_norm_list=None, delta_t_avg=None, only_zonal=False, remove_zonal=False, scale_kmin=False, k_exp=0, alpha_kx_O=1, beta_kx_O=0, lw=None, no_label=False, scaling_theory="GCB", W_instead_of_phi=False, scale_fac_vals=None, zonal_stationary=False, load_from_file=False, mult_k=False, plot_alpha_spectrum=False, plot_RH_phi_spectrum=False, alpha_plot=1, markersize=3):

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(9,6))


        for i, dataObj in enumerate(self.list_dataObj):

            filename_data = self.filenames_base[i]
            if W_instead_of_phi:
                filename_data += "_W-instead-of-phi"

            if plot_kx:
                if only_zonal:
                    if plot_RH_phi_spectrum:
                        filename_data += "_Ephi_RH_kx_zonal.dat"
                    else:
                        filename_data += "_Ephi_kx_zonal.dat"
                else:
                    filename_data += "_Ephi_kx.dat"
            else:
                filename_data += "_Ephi_ky.dat"

            ####### LOAD DATA
            if exists(filename_data) and load_from_file:
                phi2_k, k = np.loadtxt(filename_data)

            else:
                time = dataObj.get_time_array()
                time_max = time[time_idx]
                if delta_t_avg is not None:
                    time_min = time_max-delta_t_avg
                else:
                    time_min = time_max-10

                if plot_RH_phi_spectrum:
                    E_RH_t_kx, RH_time, RH_kx = dataObj.get_E_RH_t_kx(time_min=time_min, time_max=time_max)
                    phi2_k = np.average(E_RH_t_kx[:,RH_kx>0], weights=np.gradient(RH_time), axis=0)*2
                    k      = RH_kx[      RH_kx>0]
                    phi2_k_stddev = np.zeros_like(phi2_k)

                else:
                    if not zonal_stationary:
                        if W_instead_of_phi:
                            phi2_t_kx_ky, time, kx, ky = dataObj.read_W_spectra(time_min=time_min, time_max=time_max)
                        else:
                            phi2_t_kx_ky, time, kx, ky = dataObj.read_phi2_spectra(time_min=time_min, time_max=time_max)

                        delta_kx = kx[1]-kx[0]
                        delta_ky = ky[1]-ky[0]
    
                        phi2_t_kx_ky[np.isnan(phi2_t_kx_ky)]=0
    
                        if delta_t_avg is None:
                            phi2_kx_ky = phi2_t_kx_ky[-1]
                            phi2_kx_ky_stddev = np.zeros_like(phi2_kx_ky)
                            print("Evaluating at t = %.2f" % (time[-1]))
                        else:
                            phi2_kx_ky = np.average(    phi2_t_kx_ky, weights=np.gradient(time), axis=0)
                            phi2_kx_ky_stddev = np.std( phi2_t_kx_ky,                            axis=0)
    
                        if plot_kx:
    
                            if only_zonal:
                                phi2_kx_ky[:,1:] = 0
                                phi2_kx_ky[0,0]  = 0
                                phi2_kx_ky_stddev[:,1:] = 0
                                phi2_kx_ky_stddev[0,0]  = 0
                                kx = np.array(kx)
                                phi2_kx_ky = phi2_kx_ky[kx > 0, :]
                                phi2_kx_ky_stddev = phi2_kx_ky_stddev[kx > 0, :]
                                #phi2_kx_ky = phi2_kx_ky[kx > 0, :]
                                kx = kx[kx>0]
                                
                            if remove_zonal:
                                phi2_kx_ky[:,0]  = 0
                                phi2_kx_ky_stddev[:,0]  = 0


                            phi2_k = np.sum( phi2_kx_ky, axis=1)
                            phi2_k_stddev = np.sum( phi2_kx_ky_stddev, axis=1)
                            idx_sort = np.argsort(kx)
                            k = kx[idx_sort]
                            phi2_k = phi2_k[idx_sort]
                            phi2_k_stddev = phi2_k_stddev[idx_sort]
                            k = np.abs(k)
    #                        print("phi2(kx=0) = %e" % (phi2_k[0]))
    #                        phi2_k = phi2_k[k>0]
    #                        k = k[k>0]

                        else:
                            phi2_k = np.sum( phi2_kx_ky[:,1:], axis=0)
                            phi2_k_stddev = np.sum( phi2_kx_ky_stddev[:,1:], axis=0)
                            k = ky[1:]

                    # Stationary zonal flows
                    else:
                        omega, kx, EZ_omega_kx = dataObj.get_EZ_omega_kx(quantity="phi", time_min=-delta_t_avg)
                        #phi2_k = np.sum(EZ_omega_kx[:, kx>0], axis=0)*2
                        k = kx[kx>0]
                        #phi2_k = EZ_omega_kx[0, kx>0]*2
                        phi2_k = EZ_omega_kx[0, kx>0]*2/k**2

                # multiply phi2_k with k power if desired
                phi2_k = phi2_k * k**(k_exp)
                phi2_k_stddev = phi2_k_stddev * k**(k_exp)

                if scale_kmin:
                    print("Rescaling phi2 with kmin (to be able to compare sims with different x0,y0)")
                    phi2_k = phi2_k / np.abs(k[1]-k[0])
                    phi2_k_stddev = phi2_k_stddev / np.abs(k[1]-k[0])

                # Save data to file
                np.savetxt(filename_data, (phi2_k, k))
                np.savetxt(filename_data[:-4]+"_stddev.dat", (phi2_k_stddev, k))

            ### RESCALE DATA
            if tprim_norm_list is not None:
                try:
                    aspectratio = dataObj.aspect_ratio
                    print("Aspect ratio = %.2f" % (aspectratio))
                except:
                    aspectratio = 2.778
                    print("Setting aspect ratio to %.2f" % (aspectratio))

                try:
                    safetyfactor = qinp_norm_list[i]
                except:
                    print("Setting safety factor to 1")
                    safetyfactor = 1

                phi2_k = phi2_k* 2*aspectratio**2

                kappa  = tprim_norm_list[i]*aspectratio
                print("A = %.4f, kappa = %.4f, q = %.4f" % (aspectratio, kappa, safetyfactor))
                #phi2_k = phi2_k / np.abs(safetyfactor**3 * kappa**5)
                #k      = k[:] * np.abs(safetyfactor*kappa)
                #phi2_k = phi2_k / np.abs(safetyfactor**(2/3) * kappa**(8/3))
                #k      = k[:]

                #alpha_kx_O = -1/2
                #alpha_kx_O = 1  # isotropy perpendicular to B
                #alpha_kx_O = 0  # cut-off at kx ~ 1
                #print("alpha_k_O = %.2f" % (alpha_kx_O))
                #print("beta_k_O = %.2f"  % (beta_kx_O))
                #kx_O = (safetyfactor*kappa)**(-alpha_kx_O)
                #kx_O_phi = (safetyfactor*kappa)**(-beta_kx_O)
                #phi2_k = phi2_k / np.abs(safetyfactor**(2/3) * kappa**(8/3) * kx_O_phi**(-7/3))
                #k      = k[:]/kx_O

                #k = k*np.sqrt(kappa)
                #phi2_k = phi2_k / (kappa**(7/2))

#                if plot_kx:
#                    k = k*np.sqrt(kappa)
#                    phi2_k = phi2_k / (kappa**(7/2))
#                else:
#                    k = k*kappa
#                    phi2_k = phi2_k / (kappa**(4))

                if scaling_theory == "CB":
                    k = k*kappa*safetyfactor
                    phi2_k = phi2_k / (kappa**5 * safetyfactor**3)

                elif scaling_theory == "GCB":
                    if plot_kx:
#                        k = k*safetyfactor**0.5
#                        phi2_k = phi2_k / (kappa**2 * safetyfactor**1)
                         k = k*safetyfactor
                         phi2_k = phi2_k / (kappa**2 * safetyfactor**3)
                    else:
#                        k = k*kappa*safetyfactor
#                        phi2_k = phi2_k / (kappa**3 * safetyfactor**2)
                        k = k*kappa*safetyfactor
                        phi2_k = phi2_k / (kappa**3 * safetyfactor**3)

                elif scaling_theory == "MGCB":
                    if plot_kx:
                        k = k*safetyfactor**0.5
                        phi2_k = phi2_k / (kappa**2 * safetyfactor**2.5)
                    else:
                        k = k*kappa*safetyfactor
                        phi2_k = phi2_k / (kappa**3 * safetyfactor**3)

                elif scaling_theory == "zonal_diffusive":
                    phi2_k = phi2_k / (kappa**2 * safetyfactor)

                elif scaling_theory == "heuristic_T":
                    if plot_kx:
                        k = k*kappa*safetyfactor
                        phi2_k = phi2_k / (kappa**4 * safetyfactor**3)
                    else:
                        k = k*kappa*safetyfactor
                        phi2_k = phi2_k / (kappa**4 * safetyfactor**3)



            if xdrift_norm_list is not None:
                print("Scaling with xdrift_norm")
    
                norm = xdrift_norm_list[i]

                if plot_kx:
                    if only_zonal:
                        kexp_alphaD = 0
                        phi2exp_alphaD = 1
                    else:
                        if scaling_theory == "GCB":
                            kexp_alphaD = 1
                            phi2exp_alphaD = 2
                        else:
                            kexp_alphaD = 1/2
                            phi2exp_alphaD = 3/2
                        #kexp_alphaD = 2/3
                        #phi2exp_alphaD = 5/3
                else:
                    kexp_alphaD = 0
                    phi2exp_alphaD = 1

                phi2_k = phi2_k / (norm**(phi2exp_alphaD))
                k = k*norm**(kexp_alphaD)

                #k = k*np.sqrt(norm)
                #phi2_k = phi2_k / (norm**2)

            if scale_fac_vals is not None:
                phi2_k = phi2_k*scale_fac_vals[i]


            ####### PLOT
            try:
                ls = ls_list[i]
            except:
                ls = None
            try:
                marker = marker_list[i]
            except:
                marker = '.'
                if only_zonal:
                    marker = 's'
            try:
                color = color_list[i]
            except:
                color = None

            if no_label:
                label = None
            else:
                label = self.list_labels[i]
 
            if only_zonal and not W_instead_of_phi and not plot_RH_phi_spectrum:
                Gamma0 = specialfunc.iv(0, k**2/2) * np.exp(-k**2/2)
                phi2_k = phi2_k*(1-Gamma0)
            elif mult_k:
                phi2_k = phi2_k*np.abs(k)

            if plot_alpha_spectrum:
                idx_k_0 = np.argmin(k)
                k[:idx_k_0] *= -1
                k_mid, alpha_k = get_alpha_spectrum(k, phi2_k)
                k_mid = np.abs(k_mid)
                ax.semilogx(k_mid, alpha_k, label=label, ls='None', marker=marker, color=color, lw=lw, alpha=alpha_plot)
                #ax.plot(k_mid, alpha_k, label=label, ls=ls, marker=marker, color=color, lw=lw)
                ax.set_ylabel(r"$\alpha$")
            else:
                ax.loglog(k[np.abs(k)>0], phi2_k[np.abs(k)>0], label=label, ls=ls, marker=marker, color=color, lw=lw, markersize=markersize, alpha=alpha_plot)
                #print(np.trapz(y=phi2_k[(np.abs(k)>0.3) & , x=
#                idx_k_max = np.argmin(np.abs(phi2_k-phi2_k.max()))
#                ax.scatter([k[idx_k_max]], [phi2_k[idx_k_max]], marker=marker, color=color, s=200)
#            if plot_kx:
#                plt.xscale('symlog', linthresh=kx[1]-kx[0])

            if not plot_alpha_spectrum:
                if i == len(self.list_dataObj)-1:
                    # Plot theoretical -7/3 scaling (Barnes et al. 2011)
                    idx_phi2_max = np.argmax(phi2_k[1:]) + 1
                    k_plot = np.linspace(1,10,10)*k[idx_phi2_max]
#                    if only_zonal and plot_RH_phi_spectrum:
#                        phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-4)
#                        ax.plot(k_plot, phi2_k_theory, c='0.5', ls=':')
#                        phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-2)
#                        ax.plot(k_plot, phi2_k_theory, c='0.5', ls=':')

#                    if not only_zonal:
#                        if plot_kx:
#                            phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-7/3)
#                            ax.plot(k_plot*20, phi2_k_theory, c='0.5', lw=4)
#                            ax.text(k_plot[2]*3, phi2_k_theory[2]*20, r"$\sim k^{-7/3}$", c='0.5')
#                            #ax.text(k_plot[2]*20.2, phi2_k_theory[2], r"$\sim k^{-7/3}$", c='0.5')
#                    #        phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-2)
#                    #        ax.plot(k_plot*20, phi2_k_theory, c='0.5', lw=4)
#                    #        ax.text(k_plot[2]*20.2, phi2_k_theory[2], r"$\sim k^{-2}$", c='0.5')
#                            phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-1)
#                            ax.plot(k_plot, 8*phi2_k_theory, ls='-', c='0.5', label=r"$\sim k^{-1}$", lw=4)
#                            #phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-1/2)
#                            #ax.plot(k_plot/3, 4*phi2_k_theory, ls='--', c='g', label=r"$\sim k^{-1/2}$", lw=4)
#                            #idx_min = np.argmin(k)
#                            #k_plot = np.linspace(1,3,100)*k[idx_min+1]
#                            #phi2_k_theory = phi2_k[idx_min+1]*np.ones_like(k_plot)
#                            #ax.plot(k_plot, phi2_k_theory, ls='--', c='g', label=r"$\sim k^{0}$", lw=4)
#                        else:
#                            phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-7/3)
#                            ax.plot(k_plot*10, phi2_k_theory, c='0.5', lw=4)
#                            ax.text(k_plot[2]*1.5, phi2_k_theory[2]*20, r"$\sim k^{-7/3}$", c='0.5')
#                            #ax.text(k_plot[2]*10.1, phi2_k_theory[2], r"$\sim k^{-7/3}$", c='0.5')
#                            #phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-3)
#                            #ax.plot(k_plot*10, phi2_k_theory, c='0.5', lw=4)
#                            #ax.text(k_plot[2]*10.1, phi2_k_theory[2], r"$\sim k^{-3}$", c='0.5')
#                            #phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-5/3)
#                            #ax.plot(k_plot*10, phi2_k_theory, c='0.5', lw=4)
#                            #ax.text(k_plot[2]*10.1, phi2_k_theory[2], r"$\sim k^{-5/3}$", c='0.5')
#
#                            #phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-5/3)
#                            #ax.plot(k_plot/4, 10*phi2_k_theory, ls='--', c='g', label=r"$\sim k^{-5/3}$", lw=4)
#                            #phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-4/3)
#                            #ax.plot(k_plot/4, 10*phi2_k_theory, ls='--', c='g', label=r"$\sim k^{-4/3}$", lw=4)
#                            #k_plot = np.linspace(1,3,100)*k[0]
#                            #phi2_k_theory = phi2_k[0]*(k_plot/k_plot[0])
#                            #ax.plot(k_plot, phi2_k_theory, ls='--', c='g', label=r"$\sim k^{1}$", lw=4)
#                        #phi2_k_theory = phi2_k[idx_phi2_max] * (k_plot/k[idx_phi2_max])**(-1/2)
#                        #ax.plot(k_plot, phi2_k_theory, ls='--', c='g', label=r"$\sim k^{-1/2}$")
##                    if only_zonal:
##                        phi2_k_theory = phi2_k[idx_phi2_max]*2 * (k_plot/k[idx_phi2_max])**(-10/3)
##                        ax.plot(k_plot, phi2_k_theory, ls='--', c='g', label=r"$\sim k^{-10/3}$")

#        if W_instead_of_phi:
#            ylabel_base = r"$W$"
#        else:
#            ylabel_base = r"$\Phi^2$"

        if plot_kx:
            if plot_alpha_spectrum:
                ylabel_base = r"$\alpha_{k_x}$"
            else:
                if W_instead_of_phi:
                    ylabel_base = r"$W_{k_x}$"
                else:
                    if remove_zonal:
                        ylabel_base = r"$\left(\frac{Z_i e\delta\varphi^\mathrm{NZ}_{k_x}}{T_i} \frac{R}{\rho_i}\right)^2$"
                        #ylabel_base = r"$(e_i\delta\varphi^\mathrm{NZ}_{k_x}/T_i\; R/\rho_i)^2$"
                    elif only_zonal:
                        ylabel_base = r"$\left(\frac{Z_i e\delta\varphi^\mathrm{Z}_{k_x}}{T_i} \frac{R}{\rho_i}\right)^2$"
                        #ylabel_base = r"$(e_i\delta\varphi^\mathrm{Z}_{k_x}/T_i\; R/\rho_i)^2$"
                    else:
                        ylabel_base = r"$\left(\frac{Z_i e\delta\varphi_{k_x}}{T_i} \frac{R}{\rho_i}\right)^2$"
                        #ylabel_base = r"$(e_i\delta\varphi_{k_x}/T_i\; R/\rho_i)^2$"
                    #ylabel_base = r"$\Phi_{k_x}^2$"
            xlabel = r"$k_x \rho_i$"
            #xlabel = r"$|k_x| \rho_i$"
        else:
            if plot_alpha_spectrum:
                ylabel_base = r"$\alpha_{k_y}$"
            else:
                if W_instead_of_phi:
                    ylabel_base = r"$W_{k_y}$"
                else:
                    ylabel_base = r"$\left(\frac{Z_i e\delta\varphi_{k_y}}{T_i} \frac{R}{\rho_i}\right)^2$"
                    #ylabel_base = r"$(e_i\varphi_{k_y}/T_i\; R/\rho_i)^2$"
                    #ylabel_base = r"$\Phi_{k_y}^2$"
            xlabel = r"$k_y \rho_i$"

        #if delta_t_avg is not None:
        #    ylabel_base = r"$\langle$" + ylabel_base + r"$\rangle_{\Delta t = %i}$" % (delta_t_avg)

        if only_zonal and not W_instead_of_phi:
            ylabel = ylabel_base + r"$(1-\Gamma_0)$"
        elif mult_k:
            ylabel = ylabel_base + r"$k/k_\mathrm{scale}$"
        else:
            ylabel = ylabel_base

        if k_exp != 0:
            ylabel = ylabel + r"$k^{%i}$" % (k_exp)
        else:
            ylabel = ylabel# + r"$(t=%.1f)$" % (time[time_idx])

        if tprim_norm_list is not None:
            #xlabel = xlabel + r"$(q \kappa)^{%.1f}$" % (-alpha_kx_O)
            #ylabel = ylabel + r"$ / (q^3 \kappa^5 (q\kappa)^{7/3(%.1f-1)} )$" % (beta_kx_O)
            if scaling_theory == "CB":
                xlabel = xlabel + r"$q \kappa$"
                ylabel = ylabel + r"$ / q^3 \kappa^{5}$"
            elif scaling_theory == "GCB":
                if plot_kx:
                    #xlabel = xlabel + r"$\kappa^{1/2}$"
                    #ylabel = ylabel + r"$ / \kappa^{7/2}$"
                    xlabel = xlabel + r"$q$"
                    ylabel = ylabel + r"$ / q^3 \kappa^{2}$"
                else:
                    xlabel = xlabel + r"$q \kappa$"
                    ylabel = ylabel + r"$ / q^3 \kappa^{3}$"

            elif scaling_theory == "MGCB":
                if plot_kx:
                    xlabel = xlabel + r"$q^{1/2}$"
                    ylabel = ylabel + r"$ / q^2 \kappa^{5/2}$"
                else:
                    xlabel = xlabel + r"$q \kappa$"
                    ylabel = ylabel + r"$ / q^3 \kappa^{3}$"

            elif scaling_theory == "zonal_diffusive":
                ylabel = ylabel + r"$ / q \kappa^2$"

            elif scaling_theory == "heuristic_T":
                xlabel = xlabel + r"$q \kappa$"
                ylabel = ylabel + r"$ / q^4 \kappa^4$"

        add_arb_units = False
        #add_arb_units = True
        if add_arb_units:
            ylabel+=r" (arb. units)"

        if xdrift_norm_list is not None:
            xlabel = xlabel + r"$\alpha_D^{%.2f}$" % (kexp_alphaD)
            ylabel = ylabel + r"$ / \alpha_D^{%.2f}$" % (phi2exp_alphaD)

        ax.set_xlabel(xlabel)
        if plot_alpha_spectrum:
            ax.set_ylabel(ylabel_base)
        else:
            ax.set_ylabel(ylabel)
#        plt.gca().xaxis.grid(True, which='minor')
#        plt.gca().yaxis.grid(True, which='minor')
        ax.grid()

        #return fig, ax, time
        return fig, ax

    def plot_Q_k_spectrum(self, plot_kx, species_idx=0, tube=0, fig=None, ax=None, time_idx=-1, ls_list=None, marker_list=None, color_list=None, delta_t_avg=None, zed_val=None, scale_k=False, scale_kmin=True, kfilter_vals=None, plot_k_qk=False):

        if ax is None:
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(10,8))

        for i, dataObj in enumerate(self.list_dataObj):

            #if dataObj.code == "stella":
            qflx_t_zed_kx_ky, time, zed, kx, ky = dataObj.read_flux_spectra(species_idx=species_idx, tube=tube)
            delta_kx = kx[1]-kx[0]
            delta_ky = ky[1]-ky[0]
    
            if delta_t_avg is None:
                qflx_zed_kx_ky = qflx_t_zed_kx_ky[time_idx]
            else:
                qflx_zed_kx_ky = np.average( qflx_t_zed_kx_ky[time > time[time_idx]-delta_t_avg], axis=0)
    
            if zed_val is None:
                dl_over_B_avg = dataObj.dl_over_B_avg()
                qflx_kx_ky = np.sum( dl_over_B_avg[:,None,None] * qflx_zed_kx_ky, axis=0)
            else:
                zed_idx = np.argmin( np.abs( zed[:] - zed_val ) )
                qflx_kx_ky = qflx_zed_kx_ky[zed_idx]
    
            if plot_kx:
                qflx_k = np.sum( qflx_kx_ky, axis=1)
                idx_sort = np.argsort(kx)
                k = kx[idx_sort]
                qflx_k = qflx_k[idx_sort]
    #                    #    k = np.abs(kx)
            else:
            #    qflx_k = qflx_k[k>0]
            #    k = k[k>0]
                qflx_k = np.sum( qflx_kx_ky, axis=0)
                k = ky

            #elif dataObj.code == "GX":
            #    if plot_kx:
            #        qflx_kx = dataObj.ncdata['Spectra']['Qkxst'][-1,0,:] / 2**(3/2)
            #        kx      = dataObj.ncdata.variables['kx'][:]
            #        idx_sort = np.argsort(kx)
            #        k = kx[idx_sort]
            #        qflx_k = qflx_kx[idx_sort]
            #    else:
            #        qflx_k  = dataObj.ncdata['Spectra']['Qkyst'][-1,0,:] / 2**(3/2)
            #        k       = dataObj.ncdata.variables['ky'][:]
            #else:
            #    print("WARNING! Invalid code entered.")

            if scale_k and zed_idx is not None:
                _, _, gds2, _, gds22, _ = dataObj.get_FLR()
                if plot_kx:
                    k = k*np.sqrt(gds22[zed_idx])
                else:
                    k = k*np.sqrt(gds2[zed_idx])

            if scale_kmin:
                print("Rescaling Q with kmin (to be able to compare sims with different x0,y0)")
                qflx_k = qflx_k / np.abs(k[1]-k[0])

#            # Determine maxima of Q(k)
#            print("\nFor " + self.filenames_base[i] + ", k at which Q=Qmax (local) is:")
#            Nr_comp = 4
#            for i_k in range(len(k)-Nr_comp):
#                i_count = int(i_k+Nr_comp/2)
#                is_local_max = True
#                for i_comp in range(int(Nr_comp/2)):
#                    if qflx_k[i_count] < qflx_k[i_count+i_comp] or qflx_k[i_count] < qflx_k[i_count-i_comp]:
#                        is_local_max = False
#                        continue
#                if is_local_max:
#                    print("k = %e" % (k[i_count]))
#
#            ## Check
#            #print("Qflx(t=%e) = %e" % (time[time_idx], np.sum(qflx_k)))

            # Evaluate and print integrated heat flux for some kfilter_vals
            if kfilter_vals is not None:
                if plot_kx:
                    str_k = "(k=kx)"
                else:
                    str_k = "(k=ky)"
                print("\n"+self.filenames_base[i]+str_k+":")
                Q_integrated = np.sum(qflx_k)
                print("    sum_k Q_k = %e" % (Q_integrated))
                for kfilter_val in kfilter_vals:
                    Q_integrated_filter = np.sum(qflx_k[k>kfilter_val])
                    print("    sum_k Q_k (k>%.4f) = %e (%.2f percent)" % (kfilter_val, Q_integrated_filter, Q_integrated_filter/Q_integrated*100))

            try:
                ls = ls_list[i]
            except:
                ls = None
            try:
                marker = marker_list[i]
            except:
                marker = '.'
            try:
                color = color_list[i]
            except:
                color = None

            if plot_kx:
                plt.xscale('symlog', linthresh=k[1]-k[0])
                k = np.abs(k)

            if plot_k_qk:
                qflx_k = qflx_k*k

            ax.loglog(k, qflx_k, label=self.list_labels[i], ls=ls, marker=marker, color=color)

            if i == len(self.list_dataObj)-1:
                # Plot theoretical -7/3 scaling (Barnes et al. 2011)
                idx_Q_max = np.argmax(qflx_k[1:]) + 12
                k_plot = np.linspace(1,10,10)*k[1+idx_Q_max]
                if plot_kx:
                    qflx_k_theory = qflx_k[idx_Q_max] * (k_plot/k[idx_Q_max])**(-7/3)
                    #ax.plot(k_plot*2, qflx_k_theory, ls='--', c='g', label=r"$\sim k^{-7/3}$", lw=4)
                    #qflx_k_theory = qflx_k[idx_Q_max] * (k_plot/k[idx_Q_max])**(-1)
                    #ax.plot(k_plot/3, 4*qflx_k_theory, ls='--', c='g', label=r"$\sim k^{-1}$", lw=4)
                    ##qflx_k_theory = qflx_k[idx_Q_max] * (k_plot/k[idx_Q_max])**(-1/2)
                    ##ax.plot(k_plot/3, 4*qflx_k_theory, ls='--', c='g', label=r"$\sim k^{-1/2}$", lw=4)
                    #idx_min = np.argmin(k)
                    #k_plot = np.linspace(1,3,100)*k[idx_min+1]
                    #qflx_k_theory = qflx_k[idx_min+1]*np.ones_like(k_plot)
                    #ax.plot(k_plot, qflx_k_theory, ls='--', c='g', label=r"$\sim k^{0}$", lw=4)
                else:
                    qflx_k_theory = qflx_k[idx_Q_max] * (k_plot/k[idx_Q_max])**(-4/3)
                    ax.plot(k_plot/4, qflx_k_theory, ls='--', c='g', label=r"$\sim k^{-4/3}$", lw=4)
                    #qflx_k_theory = qflx_k[idx_Q_max] * (k_plot/k[idx_Q_max])**(-5/3)
                    #ax.plot(k_plot/4, 10*qflx_k_theory, ls='--', c='g', label=r"$\sim k^{-5/3}$", lw=4)
                    #qflx_k_theory = qflx_k[idx_Q_max] * (k_plot/k[idx_Q_max])**(-4/3)
                    #ax.plot(k_plot/4, 10*qflx_k_theory, ls='--', c='g', label=r"$\sim k^{-4/3}$", lw=4)
                    #k_plot = np.linspace(1,3,100)*k[0]
                    #qflx_k_theory = qflx_k[0]*(k_plot/k_plot[0])**2
                    #ax.plot(k_plot, qflx_k_theory, ls='--', c='g', label=r"$\sim k^{2}$", lw=4)
 
        if plot_kx:
            ylabel_base = r"$Q_{k_x}$"
            if plot_k_qk:
                ylabel_base += r"$k_x$"
            xlabel = r"$k_x \rho_i$"
            if scale_k:
                xlabel = xlabel + r"$|\nabla x|$"
            #xlabel = r"$|k_x| \rho_i$"
        else:
            ylabel_base = r"$Q_{k_y}$"
            if plot_k_qk:
                ylabel_base += r"$k_y$"
            xlabel = r"$k_y \rho_i$"
            if scale_k:
                xlabel = xlabel + r"$|\nabla y|$"

        #if delta_t_avg is not None:
        #    ylabel_base = r"$\langle$" + ylabel_base + r"$\rangle_{\Delta t = %i}$" % (delta_t_avg)

        ylabel = ylabel_base# + r"$(t=%.1f)$" % (time[time_idx])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
#        plt.gca().xaxis.grid(True, which='minor')
#        plt.gca().yaxis.grid(True, which='minor')
        ax.grid()
        #ax.legend()

        return fig, ax

# alpha is logarithmic slope (e.g. alpha=-5/3 for Kolmogorov)
def get_alpha_spectrum(k, f_k):
    k_mid = 0.5*(k[1:]+k[:-1])
    f_k_mid = 0.5*(f_k[1:]+f_k[:-1])
    alpha = (f_k[1:]-f_k[:-1])/(k[1]-k[0])*k_mid/f_k_mid
    #alpha = ( np.log(f_k[1:])-np.log(f_k[:-1]) )/(k[1]-k[0])
    return k_mid, alpha
