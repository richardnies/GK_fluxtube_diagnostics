"""StellaRun: a single STELLA (or GX/GS2) simulation output.

code="stella" (default) / "GX" / "GS2" selects which
simulation code produced the output, since file naming and
netCDF variable layout differ between them (see
stella_diagnostics.io.codes). Most methods below are thin
wrappers that delegate to a free function of the same name in
one of the stella_diagnostics submodules (grid, quantities,
physics, spectral, plotting) -- see that function for the
actual implementation and docstring.
"""

import warnings
import numpy as np
import netCDF4 as nc4
from os.path import exists

import stella_diagnostics.grid as grid
import stella_diagnostics.physics.correlations as physics_correlations
import stella_diagnostics.physics.energy_transfer as physics_energy_transfer
import stella_diagnostics.physics.fluxes as physics_fluxes
import stella_diagnostics.physics.rosenbluth_hinton as physics_rosenbluth_hinton
import stella_diagnostics.physics.velocity_space as physics_velocity_space
import stella_diagnostics.physics.zonal_energy as physics_zonal_energy
import stella_diagnostics.plotting.flux_plots as plotting_flux_plots
import stella_diagnostics.plotting.kspace_plots as plotting_kspace_plots
import stella_diagnostics.plotting.realspace_plots as plotting_realspace_plots
import stella_diagnostics.plotting.zed_plots as plotting_zed_plots
import stella_diagnostics.quantities.realspace as quantities_realspace
import stella_diagnostics.quantities.registry as quantities_registry
import stella_diagnostics.spectral.omega as spectral_omega
import stella_diagnostics.spectral.stats as spectral_stats

# Placeholder aspect_ratio used when a run's geometry file exists but is in
# the Miller-geometry format this parser can't fully read (the real
# rhoc/dxdXcoord computation below is disabled -- it's never produced a
# sane value here). NOT computed from the run's own geometry -- an
# unverified stand-in (roughly CBC-like), used only so downstream tprim
# scaling theories (scan.spectrum_scan.plot_phi_k_spectrum) have some
# aspect ratio rather than crashing. If you need a physically correct
# aspect_ratio for a Miller-geometry run, fix the parsing above instead of
# trusting this.
FALLBACK_ASPECT_RATIO = 2.8


class StellaRun:

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
                self.aspect_ratio  = FALLBACK_ASPECT_RATIO #float(inputdata1[0]) / float(inputdata1[6]) # rhoc/dxdXcoord
                #self.aspect_ratio  = 1/5.55#float(inputdata1[0]) / float(inputdata1[6]) # rhoc/dxdXcoord
                # print, not warnings.warn: this class turns off UserWarning
                # (see filterwarnings('ignore', ...) above), which would
                # otherwise swallow it silently.
                print(
                    "%s: using unverified placeholder aspect_ratio=%.3f (Miller geometry file "
                    "parsing doesn't compute a real value here) -- see io.run.FALLBACK_ASPECT_RATIO."
                    % (self.filename_base, self.aspect_ratio)
                )

            except Exception as e:
                #print("Warning:", type(e).__name__) 
                print("Warning! Geometry file for " + self.filename_base + " do not exist?")

    def read_basic_params(self):
        return grid.read_basic_params(self)

    def get_kx_ky_zed(self):
        return grid.get_kx_ky_zed(self)

    def get_time_array(self, GX_big=False):
        return grid.get_time_array(self, GX_big=GX_big)

    def get_time_idx(self, time_val):
        return grid.get_time_idx(self, time_val=time_val)

    def get_zed_weight(self, mult_zed, zed=None):
        return grid.get_zed_weight(self, mult_zed=mult_zed, zed=zed)

    def read_avg_ky_rhoi(self, time_idx_jump=1, avg_qflx=False, normal_mean=False, take_max=False):
        return spectral_stats.read_avg_ky_rhoi(self, time_idx_jump=time_idx_jump, avg_qflx=avg_qflx, normal_mean=normal_mean, take_max=take_max)

    def read_avg_kx_rhoi(self, time_idx_jump=1, avg_qflx=False, normal_mean=False, take_max=False, only_zonal=False, remove_zonal=True):
        return spectral_stats.read_avg_kx_rhoi(self, time_idx_jump=time_idx_jump, avg_qflx=avg_qflx, normal_mean=normal_mean, take_max=take_max, only_zonal=only_zonal, remove_zonal=remove_zonal)

    def dl_over_B_avg(self):
        return grid.dl_over_B_avg(self)

    def flux_norm(self):
        return physics_fluxes.flux_norm(self)

    def evaluate_net_radial_drift(self, B_bounce=0.9):
        return plotting_flux_plots.evaluate_net_radial_drift(self, B_bounce=B_bounce)

    def plot_net_radial_drift(self, fig=None, ax=None, label=None, ls=None, color=None):
        return plotting_flux_plots.plot_net_radial_drift(self, fig=fig, ax=ax, label=label, ls=ls, color=color)

    def read_avg_kperp_rhoi(self, exclude_zonal=True, only_zonal=False, time_idx_jump=1):
        return spectral_stats.read_avg_kperp_rhoi(self, exclude_zonal=exclude_zonal, only_zonal=only_zonal, time_idx_jump=time_idx_jump)

    def read_data_omega_k(self, timestep=-1, om_avg=True, check_convergence=True, nonconverged_to_none=True, time_avg=None, time_val_avg=None):
        return spectral_omega.read_data_omega_k(self, timestep=timestep, om_avg=om_avg, check_convergence=check_convergence, nonconverged_to_none=nonconverged_to_none, time_avg=time_avg, time_val_avg=time_val_avg)

    def read_omega_t(self, time_avg=None):
        return spectral_omega.read_omega_t(self, time_avg=time_avg)

    def read_phi_vs_zed(self, time_avg=None, time_idx=-1, normalise_phi=True, kx_idx=0, ky_idx=0, eval_real=True, squared=False, remove_zonal=False):
        return plotting_zed_plots.read_phi_vs_zed(self, time_avg=time_avg, time_idx=time_idx, normalise_phi=normalise_phi, kx_idx=kx_idx, ky_idx=ky_idx, eval_real=eval_real, squared=squared, remove_zonal=remove_zonal)

    def read_g_vs_zed(self, time_idx=-1, species_idx=0, vpa_index=None, normalise=True):
        return physics_velocity_space.read_g_vs_zed(self, time_idx=time_idx, species_idx=species_idx, vpa_index=vpa_index, normalise=normalise)

    def read_flux_spectra(self, species_idx=0, tube=0):
        return physics_fluxes.read_flux_spectra(self, species_idx=species_idx, tube=tube)

    def read_phi2_spectra(self, time_min=0, time_max=10000000000.0, time_idx_skip=1):
        return physics_fluxes.read_phi2_spectra(self, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip)

    def read_W_spectra(self, time_min=0, time_max=10000000000.0, time_idx_skip=1):
        return physics_fluxes.read_W_spectra(self, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip)

    def read_phi_zonal_spectra(self):
        return physics_fluxes.read_phi_zonal_spectra(self)

    def read_phi2_vs_t_zed(self, tube=0, remove_zonal=False, only_zonal=False, kx_zonal=True, time_min=0, time_max=1000000.0):
        return physics_fluxes.read_phi2_vs_t_zed(self, tube=tube, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_zonal=kx_zonal, time_min=time_min, time_max=time_max)

    def read_phi2_vs_t(self, tube=0):
        return physics_fluxes.read_phi2_vs_t(self, tube=tube)

    def get_avg_kperp2(self, ky_idx=0, kx_idx=0):
        return grid.get_avg_kperp2(self, ky_idx=ky_idx, kx_idx=kx_idx)

    def get_RH_inertia(self, species_idx='sum', kx_max=100000.0, idxs_kx=None):
        return physics_rosenbluth_hinton.get_RH_inertia(self, species_idx=species_idx, kx_max=kx_max, idxs_kx=idxs_kx)

    def get_RH_fluxes(self, species_idx='sum', passing_trapped='both', time_min=0, time_max=10000000000.0, time_idx_skip=1, kx_max=100000.0, idxs_kx=None, fphi=1, fapar=1, fbpar=1, fcoll=1):
        return physics_rosenbluth_hinton.get_RH_fluxes(self, species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll)

    def get_RH_phi_I(self, species_idx='sum', time_min=0, time_max=10000000000.0, time_idx_skip=1, kx_max=100000.0, idxs_kx=None):
        return physics_rosenbluth_hinton.get_RH_phi_I(self, species_idx=species_idx, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx)

    def get_RH_fluxes_t_kx(self, species_idx='sum', passing_trapped='both', time_min=0, time_max=10000000000.0, time_idx_skip=1, kx_max=100000.0, idxs_kx=None, fphi=1, fapar=1, fbpar=1, fcoll=1):
        return physics_rosenbluth_hinton.get_RH_fluxes_t_kx(self, species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll)

    def get_RH_phi_I_t_kx(self, species_idx='sum', time_min=0, time_max=10000000000.0, time_idx_skip=1, kx_max=100000.0, idxs_kx=None):
        return physics_rosenbluth_hinton.get_RH_phi_I_t_kx(self, species_idx=species_idx, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx)

    def get_E_RH_t_kx(self, species_idx='sum', time_min=0, time_max=10000000000.0, kx_max=100000.0, idxs_kx=None):
        return physics_rosenbluth_hinton.get_E_RH_t_kx(self, species_idx=species_idx, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)

    def get_P_RH(self, species_idx='sum', passing_trapped='both', time_min=0, time_max=10000000000.0, time_idx_skip=1, kx_max=100000.0, idxs_kx=None, fphi=1, fapar=1, fbpar=1, fcoll=1):
        return physics_rosenbluth_hinton.get_P_RH(self, species_idx=species_idx, passing_trapped=passing_trapped, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, kx_max=kx_max, idxs_kx=idxs_kx, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll)

    def get_P_RH_coll_over_vnew_E_RH_t(self, vnew=None, species_idx='sum', time_min=0, time_max=10000000000.0, kx_max=100000.0, idxs_kx=None):
        return physics_rosenbluth_hinton.get_P_RH_coll_over_vnew_E_RH_t(self, vnew=vnew, species_idx=species_idx, time_min=time_min, time_max=time_max, kx_max=kx_max, idxs_kx=idxs_kx)

    def plot_E_RH(self, fig=None, ax=None, time_min=0, time_max=10000000000.0, idxs_kx=None, kx_max=100000.0, colors=None):
        return physics_rosenbluth_hinton.plot_E_RH(self, fig=fig, ax=ax, time_min=time_min, time_max=time_max, idxs_kx=idxs_kx, kx_max=kx_max, colors=colors)

    def plot_RH_phi_I(self, fig=None, axs=None, time_min=0, time_max=10000000000.0, idxs_kx=None, kx_max=100000.0, colors=None, colors_sim=None):
        return physics_rosenbluth_hinton.plot_RH_phi_I(self, fig=fig, axs=axs, time_min=time_min, time_max=time_max, idxs_kx=idxs_kx, kx_max=kx_max, colors=colors, colors_sim=colors_sim)

    def plot_RH_fluxes(self, fig=None, axs=None, time_min=0, time_max=10000000000.0, species_idx='sum', passing_trapped='both', idxs_kx=None, kx_max=100000.0, colors=None, fphi=1, fapar=1, fbpar=1, fcoll=1):
        return physics_rosenbluth_hinton.plot_RH_fluxes(self, fig=fig, axs=axs, time_min=time_min, time_max=time_max, species_idx=species_idx, passing_trapped=passing_trapped, idxs_kx=idxs_kx, kx_max=kx_max, colors=colors, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll)

    def plot_P_RH(self, fig=None, axs=None, time_min=0, time_max=10000000000.0, species_idx='sum', passing_trapped='both', idxs_kx=None, kx_max=100000.0, colors=None, fphi=1, fapar=1, fbpar=1, fcoll=1, D_hyper=None, combine_fields=False, combine_even_odd=False):
        return physics_rosenbluth_hinton.plot_P_RH(self, fig=fig, axs=axs, time_min=time_min, time_max=time_max, species_idx=species_idx, passing_trapped=passing_trapped, idxs_kx=idxs_kx, kx_max=kx_max, colors=colors, fphi=fphi, fapar=fapar, fbpar=fbpar, fcoll=fcoll, D_hyper=D_hyper, combine_fields=combine_fields, combine_even_odd=combine_even_odd)

    def get_RH_integrand_mu_vpa_zed_kx(self, species_idx=0):
        return physics_rosenbluth_hinton.get_RH_integrand_mu_vpa_zed_kx(self, species_idx=species_idx)

    def get_FLR(self, ky_idx=0, kx_idx=0):
        return grid.get_FLR(self, ky_idx=ky_idx, kx_idx=kx_idx)

    def get_omega_s_k(self, ky_idx=0, kx_idx=0):
        return quantities_registry.get_omega_s_k(self, ky_idx=ky_idx, kx_idx=kx_idx)

    def get_Gamma0(self, ky_idx=0, kx_idx=0):
        return quantities_registry.get_Gamma0(self, ky_idx=ky_idx, kx_idx=kx_idx)

    def plot_phi_vs_zed(self, ax=None, label=None, ls=None, color=None, zed_times_nfield_periods=False, time_idx=-1, normalise_phi=True, kx_idx=None, ky_idx=None):
        return plotting_zed_plots.plot_phi_vs_zed(self, ax=ax, label=label, ls=ls, color=color, zed_times_nfield_periods=zed_times_nfield_periods, time_idx=time_idx, normalise_phi=normalise_phi, kx_idx=kx_idx, ky_idx=ky_idx)

    def plot_phi2_vs_t_zed(self, tube=0, ax=None, label=None, zed_times_nfield_periods=False, remove_zonal=False):
        return plotting_zed_plots.plot_phi2_vs_t_zed(self, tube=tube, ax=ax, label=label, zed_times_nfield_periods=zed_times_nfield_periods, remove_zonal=remove_zonal)

    def plot_flux_tube_geometry(self, fig=None, axs=None, label=None, plot_phi=True, zed_times_nfield_periods=False, load_from_nc=True, normalise_bmag=False, color=None, ls='-', xlim=None, norm_gradpar=False):
        return plotting_zed_plots.plot_flux_tube_geometry(self, fig=fig, axs=axs, label=label, plot_phi=plot_phi, zed_times_nfield_periods=zed_times_nfield_periods, load_from_nc=load_from_nc, normalise_bmag=normalise_bmag, color=color, ls=ls, xlim=xlim, norm_gradpar=norm_gradpar)

    def get_zonal_shearing_kx(self, time_min=0, time_max=100000.0):
        return physics_zonal_energy.get_zonal_shearing_kx(self, time_min=time_min, time_max=time_max)

    def get_fluxes_over_time(self, species_idx=0, norm=True, configuration=None, delta_t=None, load_from_nc=False):
        return physics_fluxes.get_fluxes_over_time(self, species_idx=species_idx, norm=norm, configuration=configuration, delta_t=delta_t, load_from_nc=load_from_nc)

    def get_dt_par_mom_pressure_transport(self, time_min=0, time_max=10000000000.0, time_idx_skip=1, nx=None, ny=None, kx_lowpass_cutoff=np.inf, ky_lowpass_cutoff=np.inf, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1):
        return physics_zonal_energy.get_dt_par_mom_pressure_transport(self, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)

    def get_dt_par_mom_pressure_transport_x(self, time_idx=-1, nx=None, ny=None, kx_lowpass_cutoff=np.inf, ky_lowpass_cutoff=np.inf, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1):
        return physics_zonal_energy.get_dt_par_mom_pressure_transport_x(self, time_idx=time_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)

    def get_dt_zonal_energy_contributions(self, time_min=0, time_max=10000000000.0, time_idx_skip=1, nx=None, ny=None, kx_lowpass_cutoff=np.inf, ky_lowpass_cutoff=np.inf, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1, separate_Reynolds=True):
        return physics_zonal_energy.get_dt_zonal_energy_contributions(self, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, separate_Reynolds=separate_Reynolds)

    def get_Reynolds_NZ_spectrum(self, time_min=0, time_max=99999, time_idx_skip=1):
        return physics_zonal_energy.get_Reynolds_NZ_spectrum(self, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip)

    def get_Reynolds_kz_kxNZ_spectrum(self, time_min=0, time_max=99999, time_idx_skip=1):
        return physics_zonal_energy.get_Reynolds_kz_kxNZ_spectrum(self, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip)

    def get_time_avg_zonal_energy_contributions_kx(self, time_min=0, time_max=10000000000.0, time_idx_skip=1, alt_slow_eval=False, omega_min=None, omega_max=None):
        return physics_zonal_energy.get_time_avg_zonal_energy_contributions_kx(self, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, alt_slow_eval=alt_slow_eval, omega_min=omega_min, omega_max=omega_max)

    def get_dt_zonal_energy_contributions_x(self, time_idx=-1, nx=None, ny=None, kx_lowpass_cutoff=np.inf, ky_lowpass_cutoff=np.inf, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1):
        return physics_zonal_energy.get_dt_zonal_energy_contributions_x(self, time_idx=time_idx, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff)

    def get_EZ_omega_x(self, quantity, time_min=0, time_max=99999, time_idx_skip=1, nx=None):
        return physics_zonal_energy.get_EZ_omega_x(self, quantity=quantity, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, nx=nx)

    def get_EZ_omega_kx(self, quantity, time_min=0, time_max=99999, time_idx_skip=1, nx=None):
        return physics_zonal_energy.get_EZ_omega_kx(self, quantity=quantity, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, nx=nx)

    def get_EZ_omega(self, quantity='phi', time_min=0, time_max=99999, time_idx_skip=1, nx=None):
        return physics_zonal_energy.get_EZ_omega(self, quantity=quantity, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, nx=nx)

    def get_energies_over_time(self, species_idx=0):
        return physics_fluxes.get_energies_over_time(self, species_idx=species_idx)

    def get_moments2_over_time(self, species_idx=0, remove_zonal=True):
        return physics_fluxes.get_moments2_over_time(self, species_idx=species_idx, remove_zonal=remove_zonal)

    def plot_flux_over_time(self, axs=None, label=None, species_idx=None, ls='-', color=None, marker=None, timeavg=None, timemax=np.inf, log=False):
        return plotting_flux_plots.plot_flux_over_time(self, axs=axs, label=label, species_idx=species_idx, ls=ls, color=color, marker=marker, timeavg=timeavg, timemax=timemax, log=log)

    def plot_flux_spectra(self, fig=None, ax=None, species_idx=0, tube=0, time_idx=-1, kx_idx=0):
        return plotting_flux_plots.plot_flux_spectra(self, fig=fig, ax=ax, species_idx=species_idx, tube=tube, time_idx=time_idx, kx_idx=kx_idx)

    def plot_flux_spectra_kx_ky(self, fig=None, ax=None, species_idx=0, tube=0, time_idx=-1, normalise_ky=False):
        return plotting_flux_plots.plot_flux_spectra_kx_ky(self, fig=fig, ax=ax, species_idx=species_idx, tube=tube, time_idx=time_idx, normalise_ky=normalise_ky)

    def plot_quantities_over_zed(self, fig=None, ax=None, mult_zed=1, zed_times_nfield_periods=False, time_idx=-1, ls=None, color=None, norm_all=False, **kwargs):
        return plotting_zed_plots.plot_quantities_over_zed(self, fig=fig, ax=ax, mult_zed=mult_zed, zed_times_nfield_periods=zed_times_nfield_periods, time_idx=time_idx, ls=ls, color=color, norm_all=norm_all, **kwargs)

    def plot_quantity_zed_t(self, quantity, fig=None, ax=None, vmin=None, vmax=None, species_idx=0, logarithmic=False, remove_zonal=False, only_zonal=False, sideband=False, time_idx_skip=1, normalise_each_t=False, cmap='inferno', kx_order=0, ky_order=0, nx=None, ny=None, avg_norm=None, time_min=0, time_max=99999, mult_zed=None, kx_lowpass_cutoff=np.inf, plot_zed_avg=True):
        return plotting_zed_plots.plot_quantity_zed_t(self, quantity=quantity, fig=fig, ax=ax, vmin=vmin, vmax=vmax, species_idx=species_idx, logarithmic=logarithmic, remove_zonal=remove_zonal, only_zonal=only_zonal, sideband=sideband, time_idx_skip=time_idx_skip, normalise_each_t=normalise_each_t, cmap=cmap, kx_order=kx_order, ky_order=ky_order, nx=nx, ny=ny, avg_norm=avg_norm, time_min=time_min, time_max=time_max, mult_zed=mult_zed, kx_lowpass_cutoff=kx_lowpass_cutoff, plot_zed_avg=plot_zed_avg)

    def plot_parallel_correlation_function(self, quantity='phi', time_idx=-1, time_avg=0, fig=None, ax=None, zeta_max=False, k_min=None, k_max=None, no_plot=False, kx_instead_of_ky=False, keep_only_zonal=False, vmin=None, vmax=None):
        return physics_correlations.plot_parallel_correlation_function(self, quantity=quantity, time_idx=time_idx, time_avg=time_avg, fig=fig, ax=ax, zeta_max=zeta_max, k_min=k_min, k_max=k_max, no_plot=no_plot, kx_instead_of_ky=kx_instead_of_ky, keep_only_zonal=keep_only_zonal, vmin=vmin, vmax=vmax)

    def get_parallel_correlation_function_kx_ky(self, quantity='phi', time_idx=-1, zeta_max=False, k_min=None):
        return physics_correlations.get_parallel_correlation_function_kx_ky(self, quantity=quantity, time_idx=time_idx, zeta_max=zeta_max, k_min=k_min)

    def get_perp_correlation_function(self, quantity="phi", remove_zonal=True, time_idx=-1, sum_other=True):
        return physics_correlations.get_perp_correlation_function(self, quantity=quantity, remove_zonal=remove_zonal, time_idx=time_idx, sum_other=sum_other)

    def get_energy_transfer_kx_ky(self, time_min, time_max, time_idx_skip=1):
        return physics_energy_transfer.get_energy_transfer_kx_ky(self, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip)

    def get_quantity_zed_kx_ky(self, quantity, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, alt_slow_eval=False):
        return quantities_registry.get_quantity_zed_kx_ky(self, quantity=quantity, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, alt_slow_eval=alt_slow_eval)

    def get_quantity_kx_ky(self, quantity, zed_val=None, zed_idx=None, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, mult_zed=None, par_der_order=0, mean_delt_zed=None, alt_slow_eval=False, sort_kx=False):
        return quantities_registry.get_quantity_kx_ky(self, quantity=quantity, zed_val=zed_val, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, mult_zed=mult_zed, par_der_order=par_der_order, mean_delt_zed=mean_delt_zed, alt_slow_eval=alt_slow_eval, sort_kx=sort_kx)

    def get_quantity_zed_x_y(self, quantity, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, kx_lowpass_cutoff=np.inf, ky_lowpass_cutoff=np.inf, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1, abs_squared=False, quantity_mult=None):
        return quantities_realspace.get_quantity_zed_x_y(self, quantity=quantity, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, nx=nx, ny=ny, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, abs_squared=abs_squared, quantity_mult=quantity_mult)

    def get_quantity_x_y(self, quantity, zed_val=None, zed_idx=None, time_idx=-1, species_idx=0, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, mult_zed=None, kx_lowpass_cutoff=10000000000.0, ky_lowpass_cutoff=10000000000.0, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1, par_der_order=0, abs_squared=False):
        return quantities_realspace.get_quantity_x_y(self, quantity=quantity, zed_val=zed_val, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, nx=nx, ny=ny, mult_zed=mult_zed, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, par_der_order=par_der_order, abs_squared=abs_squared)

    def plot_quantity_3d_torus(self, quantity='phi', fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, vmin=None, vmax=None, cmap=None, torus_rmax=0.6, torus_rmin=0.25, Delta_zeta=np.pi / 3, nzeta=50, xlim=np.inf, lighting=True, ikymin=0, ikymax=None):
        return plotting_realspace_plots.plot_quantity_3d_torus(self, quantity=quantity, fig=fig, ax=ax, species_idx=species_idx, time_idx=time_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, vmin=vmin, vmax=vmax, cmap=cmap, torus_rmax=torus_rmax, torus_rmin=torus_rmin, Delta_zeta=Delta_zeta, nzeta=nzeta, xlim=xlim, lighting=lighting, ikymin=ikymin, ikymax=ikymax)

    def plot_quantity_poloidal_ring(self, quantity='phi', fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, vmin=None, vmax=None, cmap=None, xmin=None, xmax=None, ymin=None, ymax=None, rorigin_fac=2, zed_idx_skip=1, kyfilter_fac=None, ky_lowpass_cutoff=np.inf):
        return plotting_realspace_plots.plot_quantity_poloidal_ring(self, quantity=quantity, fig=fig, ax=ax, species_idx=species_idx, time_idx=time_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, nx=nx, ny=ny, vmin=vmin, vmax=vmax, cmap=cmap, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, rorigin_fac=rorigin_fac, zed_idx_skip=zed_idx_skip, kyfilter_fac=kyfilter_fac, ky_lowpass_cutoff=ky_lowpass_cutoff)

    def plot_quantity_box_zed_x_y(self, quantity='phi', fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, symm=False, vmin=None, vmax=None, kx_lowpass_cutoff=np.inf, ky_lowpass_cutoff=np.inf, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1, cmap=None, xmin=None, xmax=None, ymin=None, ymax=None, zed_neg=True):
        return plotting_realspace_plots.plot_quantity_box_zed_x_y(self, quantity=quantity, fig=fig, ax=ax, species_idx=species_idx, time_idx=time_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, nx=nx, ny=ny, symm=symm, vmin=vmin, vmax=vmax, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, cmap=cmap, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zed_neg=zed_neg)

    def plot_quantity_x_y(self, quantity='phi', fig=None, ax=None, zed_val=None, zed_idx=None, mult_zed=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, show_iota_x=False, kx_order=0, ky_order=0, time_avg=None, nx=None, ny=None, symm=False, vmin=None, vmax=None, kx_lowpass_cutoff=np.inf, ky_lowpass_cutoff=np.inf, kx_highpass_cutoff=-1, ky_highpass_cutoff=-1, cmap=None, xmin=None, xmax=None, ymin=None, ymax=None, interpolation=False, projection_3d=False, plot_contours=False, suptitle=True, xy_layout=True):
        return plotting_realspace_plots.plot_quantity_x_y(self, quantity=quantity, fig=fig, ax=ax, zed_val=zed_val, zed_idx=zed_idx, mult_zed=mult_zed, species_idx=species_idx, time_idx=time_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, show_iota_x=show_iota_x, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, nx=nx, ny=ny, symm=symm, vmin=vmin, vmax=vmax, kx_lowpass_cutoff=kx_lowpass_cutoff, ky_lowpass_cutoff=ky_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, ky_highpass_cutoff=ky_highpass_cutoff, cmap=cmap, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, interpolation=interpolation, projection_3d=projection_3d, plot_contours=plot_contours, suptitle=suptitle, xy_layout=xy_layout)

    def plot_spectrum2(self, quantity, kx_or_ky, fig=None, ax=None, species_idx=0, time_idx=-1, time_val=None, remove_zonal=False, only_zonal=False, kx_order=0, ky_order=0, time_avg=None, c=None, lw=None, label=None, marker='.', scale_kmin=True, scale_CB=False, zed_val=None, zed_idx=None, ls='-', mult_zed=None):
        return plotting_kspace_plots.plot_spectrum2(self, quantity=quantity, kx_or_ky=kx_or_ky, fig=fig, ax=ax, species_idx=species_idx, time_idx=time_idx, time_val=time_val, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, ky_order=ky_order, time_avg=time_avg, c=c, lw=lw, label=label, marker=marker, scale_kmin=scale_kmin, scale_CB=scale_CB, zed_val=zed_val, zed_idx=zed_idx, ls=ls, mult_zed=mult_zed)

    def plot_Q_x_y(self, fig=None, ax=None, zed_idx=None, time_idx=-1, species_idx=0, time_val=None):
        return plotting_realspace_plots.plot_Q_x_y(self, fig=fig, ax=ax, zed_idx=zed_idx, time_idx=time_idx, species_idx=species_idx, time_val=time_val)

    def plot_quantity_x(self, quantity='phi', species_idx=0, fig=None, ax=None, zed_idx=None, time_idx=-1, label=None, ls=None, color=None, marker=None, normalise=False, time_avg=None, nx=None, mult_zed=None, kx_order=0, kx_lowpass_cutoff=100000.0, mult=1, plot_factor=1):
        return plotting_realspace_plots.plot_quantity_x(self, quantity=quantity, species_idx=species_idx, fig=fig, ax=ax, zed_idx=zed_idx, time_idx=time_idx, label=label, ls=ls, color=color, marker=marker, normalise=normalise, time_avg=time_avg, nx=nx, mult_zed=mult_zed, kx_order=kx_order, kx_lowpass_cutoff=kx_lowpass_cutoff, mult=mult, plot_factor=plot_factor)

    def plot_quantity_zonal(self, quantity='phi', species_idx=0, fig=None, axs=None, zed_idx=None, time_idx=-1, label=None, ls=None, color=None, marker=None, substract_background_temp=False, normalise=False, time_avg=None, nx=None, sum_nonzonal=False, mult_zed=None, kx_order_min=0, kx_lowpass_cutoff=100000.0, mult=1):
        return plotting_kspace_plots.plot_quantity_zonal(self, quantity=quantity, species_idx=species_idx, fig=fig, axs=axs, zed_idx=zed_idx, time_idx=time_idx, label=label, ls=ls, color=color, marker=marker, substract_background_temp=substract_background_temp, normalise=normalise, time_avg=time_avg, nx=nx, sum_nonzonal=sum_nonzonal, mult_zed=mult_zed, kx_order_min=kx_order_min, kx_lowpass_cutoff=kx_lowpass_cutoff, mult=mult)

    def get_quantity_omega_zed_kx(self, quantity, time_min, time_max, time_idx_skip=1, species_idx=0, remove_zonal=False, only_zonal=False, kx_order=0, omega_min=-np.inf, omega_max=np.inf, alt_slow_eval=True):
        return spectral_omega.get_quantity_omega_zed_kx(self, quantity=quantity, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, kx_order=kx_order, omega_min=omega_min, omega_max=omega_max, alt_slow_eval=alt_slow_eval)

    def get_quantity_filtered_in_omega(self, f_t, time, omega_min=-np.inf, omega_max=np.inf):
        return spectral_omega.get_quantity_filtered_in_omega(self, f_t=f_t, time=time, omega_min=omega_min, omega_max=omega_max)

    def plot_quantity_kx_omega(self, quantity, time_min, time_max, time_idx_skip=1, fig=None, ax=None, vmin=None, vmax=None, species_idx=0, logarithmic=False, remove_zonal=False, only_zonal=False, cmap='inferno', kx_order=0, par_der_order=0, mult_zed=None, zed_val=None, no_plot=False, omega_min=-np.inf, omega_max=np.inf, time_der=False, plot_omega2_kx2=False, mean_delt_zed=None, alt_slow_eval=False, append_mirror=False, normalise_each_kx=False, omega_norm=1, scale_eps=1):
        return spectral_omega.plot_quantity_kx_omega(self, quantity=quantity, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, fig=fig, ax=ax, vmin=vmin, vmax=vmax, species_idx=species_idx, logarithmic=logarithmic, remove_zonal=remove_zonal, only_zonal=only_zonal, cmap=cmap, kx_order=kx_order, par_der_order=par_der_order, mult_zed=mult_zed, zed_val=zed_val, no_plot=no_plot, omega_min=omega_min, omega_max=omega_max, time_der=time_der, plot_omega2_kx2=plot_omega2_kx2, mean_delt_zed=mean_delt_zed, alt_slow_eval=alt_slow_eval, append_mirror=append_mirror, normalise_each_kx=normalise_each_kx, omega_norm=omega_norm, scale_eps=scale_eps)

    def plot_quantity_x_t(self, quantity, fig=None, ax=None, vmin=None, vmax=None, species_idx=0, logarithmic=False, remove_zonal=False, only_zonal=False, time_idx_skip=1, normalise_each_t=False, y_val=None, cmap='inferno', kx_order=0, zed_val=None, zed_idx=None, mult_zed=None, time_min=0, time_max=10000000000.0, nx=None, kx_lowpass_cutoff=10000.0, kx_highpass_cutoff=-1, par_der_order=0, scale_eps=1, return_avg=False, mult=1):
        return plotting_realspace_plots.plot_quantity_x_t(self, quantity=quantity, fig=fig, ax=ax, vmin=vmin, vmax=vmax, species_idx=species_idx, logarithmic=logarithmic, remove_zonal=remove_zonal, only_zonal=only_zonal, time_idx_skip=time_idx_skip, normalise_each_t=normalise_each_t, y_val=y_val, cmap=cmap, kx_order=kx_order, zed_val=zed_val, zed_idx=zed_idx, mult_zed=mult_zed, time_min=time_min, time_max=time_max, nx=nx, kx_lowpass_cutoff=kx_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, par_der_order=par_der_order, scale_eps=scale_eps, return_avg=return_avg, mult=mult)

    def plot_quantity_x_zed(self, quantity='phi', fig=None, ax=None, time_idx=-1, vmin=None, vmax=None, logarithmic=False, remove_zonal=False, only_zonal=False, avg_norm=None, nx=None, ny=None, species_idx=0, cmap='inferno', kx_order=0, ky_order=0, kx_lowpass_cutoff=np.inf, kx_highpass_cutoff=-1, polar_plot=False, idx_x_shift=None, mult_zed=None, mult_fac=1, xlim_box=None):
        return plotting_zed_plots.plot_quantity_x_zed(self, quantity=quantity, fig=fig, ax=ax, time_idx=time_idx, vmin=vmin, vmax=vmax, logarithmic=logarithmic, remove_zonal=remove_zonal, only_zonal=only_zonal, avg_norm=avg_norm, nx=nx, ny=ny, species_idx=species_idx, cmap=cmap, kx_order=kx_order, ky_order=ky_order, kx_lowpass_cutoff=kx_lowpass_cutoff, kx_highpass_cutoff=kx_highpass_cutoff, polar_plot=polar_plot, idx_x_shift=idx_x_shift, mult_zed=mult_zed, mult_fac=mult_fac, xlim_box=xlim_box)

    def plot_quantity1_quantity2(self, quantities, fig=None, ax=None, ls='--', c=None, marker='.', time_min=0, time_max=99999, time_idx_skip=1, remove_zonals=[False, False], only_zonals=[False, False], avg_norms=[None, None], nx=None, ny=None, species_idx=0, kx_orders=[0, 0], ky_orders=[0, 0], mult_zeds=[None, None], time_ders=[False, False], mult_vals=[1, 1], all_xs=False):
        return plotting_kspace_plots.plot_quantity1_quantity2(self, quantities=quantities, fig=fig, ax=ax, ls=ls, c=c, marker=marker, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, remove_zonals=remove_zonals, only_zonals=only_zonals, avg_norms=avg_norms, nx=nx, ny=ny, species_idx=species_idx, kx_orders=kx_orders, ky_orders=ky_orders, mult_zeds=mult_zeds, time_ders=time_ders, mult_vals=mult_vals, all_xs=all_xs)

    def plot_quantity_t_k(self, quantity='phi', fig=None, ax=None, remove_zonal=False, ky_idx=None, only_zonal=False, ls=None, lw=None, log_ax=True, t_min=0, t_max=1000000.0, ratio_zonal_nonzonal=False, kx_min=-1, kx_idxs=None, time_idx_skip=1, species_idx=0, kx_order=0, ky_order=0, eval_real=False, eval_imag=False, colors=None, marker=None, no_plot=False, norm_plot=False, sum_kx=False, labels=None):
        return plotting_kspace_plots.plot_quantity_t_k(self, quantity=quantity, fig=fig, ax=ax, remove_zonal=remove_zonal, ky_idx=ky_idx, only_zonal=only_zonal, ls=ls, lw=lw, log_ax=log_ax, t_min=t_min, t_max=t_max, ratio_zonal_nonzonal=ratio_zonal_nonzonal, kx_min=kx_min, kx_idxs=kx_idxs, time_idx_skip=time_idx_skip, species_idx=species_idx, kx_order=kx_order, ky_order=ky_order, eval_real=eval_real, eval_imag=eval_imag, colors=colors, marker=marker, no_plot=no_plot, norm_plot=norm_plot, sum_kx=sum_kx, labels=labels)

    def plot_phi_t_ky(self, fig=None, ax=None, zed_idx=None, remove_zonal=False, only_zonal=False, label=None, ls=None, c=None, lw=None, log_ax=True, norm_to_t0=False, plot_abs=True, t_max=np.inf, time_avg=1, norm_kperp2=False, ratio_zonal_nonzonal=False):
        return plotting_kspace_plots.plot_phi_t_ky(self, fig=fig, ax=ax, zed_idx=zed_idx, remove_zonal=remove_zonal, only_zonal=only_zonal, label=label, ls=ls, c=c, lw=lw, log_ax=log_ax, norm_to_t0=norm_to_t0, plot_abs=plot_abs, t_max=t_max, time_avg=time_avg, norm_kperp2=norm_kperp2, ratio_zonal_nonzonal=ratio_zonal_nonzonal)

    def get_Wenergy_t_zed_kx_ky(self, time_idx_min=None, time_idx_max=None, time_min=0, time_max=10000, time_idx_skip=1, tite=1):
        return physics_zonal_energy.get_Wenergy_t_zed_kx_ky(self, time_idx_min=time_idx_min, time_idx_max=time_idx_max, time_min=time_min, time_max=time_max, time_idx_skip=time_idx_skip, tite=tite)

    def get_gvpa_gmu(self, time_idx=-1, species_idx=0, remove_zonal=False, only_zonal=False):
        return physics_velocity_space.get_gvpa_gmu(self, time_idx=time_idx, species_idx=species_idx, remove_zonal=remove_zonal, only_zonal=only_zonal)

    def get_Evpa_Emu(self, time_idx=-1, species_idx=0):
        return physics_velocity_space.get_Evpa_Emu(self, time_idx=time_idx, species_idx=species_idx)

    def get_n_T_vpa_mu(self, time_idx=-1, species_idx=0):
        return physics_velocity_space.get_n_T_vpa_mu(self, time_idx=time_idx, species_idx=species_idx)

    def plot_contour_gvmu_vpa(self, fig=None, ax=None, time_idx=-1, vmin=None, vmax=None, logarithmic=False, cmap='inferno', plot_diff=False, zonal=False, nozonal=False, species_idx=0, kx_min=None, kx_max=None, time_avg=None):
        return physics_velocity_space.plot_contour_gvmu_vpa(self, fig=fig, ax=ax, time_idx=time_idx, vmin=vmin, vmax=vmax, logarithmic=logarithmic, cmap=cmap, plot_diff=plot_diff, zonal=zonal, nozonal=nozonal, species_idx=species_idx, kx_min=kx_min, kx_max=kx_max, time_avg=time_avg)

    def plot_contour_gzvs(self, fig=None, ax=None, time_idx=-1, vmin=None, vmax=None, logarithmic=False, cmap='inferno', plot_diff=False, zonal=False, nozonal=False):
        return physics_velocity_space.plot_contour_gzvs(self, fig=fig, ax=ax, time_idx=time_idx, vmin=vmin, vmax=vmax, logarithmic=logarithmic, cmap=cmap, plot_diff=plot_diff, zonal=zonal, nozonal=nozonal)

    def evolve_markers_2D(self, t_min=0, t_max=np.inf, x0=[0], y0=[0], only_zonal_vEx=False, only_zonal_vEy=False, remove_zonal=False, zed_val=0, nx=None, ny=None, kx_highpass_cutoff=-1):
        return physics_velocity_space.evolve_markers_2D(self, t_min=t_min, t_max=t_max, x0=x0, y0=y0, only_zonal_vEx=only_zonal_vEx, only_zonal_vEy=only_zonal_vEy, remove_zonal=remove_zonal, zed_val=zed_val, nx=nx, ny=ny, kx_highpass_cutoff=kx_highpass_cutoff)

