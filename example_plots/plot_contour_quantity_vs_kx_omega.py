import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    #"font.size": 36, 
    "font.size": 24, 
    #"axes.titlepad": 35,
    "axes.titlepad": 15,
})

write_text = False
#write_text = True

import stellaDiagnostics as sD

filename_base = "CBC"

eps = 1

plot_qinps = False

dirname0 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0/"
dirname1 = "2026-06-26_scan_qinp-1.4_shat-0.8_rmaj-1.000_rhoc-0.18_fprim-2.2_vnew-0.0001/"
qinp_vals    = [dirname0+"run_tprim-4.9000", dirname1+"run_tprim-4.9000", dirname0+"run_tprim-6.3000", dirname1+"run_tprim-6.3000"]
labels = [r"$\nu=0, R/L_T=4.9$", r"$\nu=10^{-4}, R/L_T=4.9$", r"$\nu=0, R/L_T=6.3$", r"$\nu=10^{-4}, R/L_T=6.3$"]

figname_add = "_coll_comparison"

#quantity = "pressure"
#quantity = "pressure_transport"
#quantity = "par_mom_transport"
#quantity = "temperature"
#quantity = "upar"
#quantity = "dyphiPprp-dxphi"
quantity = "phi"
#quantity = "dyphi-T"
#quantity = "qperp"
#quantity = "dyphi2"

kx_order = 1
par_der_order = 0#1

mult_zed = None
#mult_zed = "pos"
#if quantity == "pressure":
#mult_zed = "vdriftx"

#zed_val = 0
#zed_val = -np.pi/2
#zed_val = np.pi/2
zed_val = None

plot_omega2_kx2 = False
#plot_omega2_kx2 = True

#time_min_vals = np.array([300, 300, 100, 100, 60])*2.8
time_min_vals = np.ones(len(qinp_vals))*500
#time_max_vals = np.ones(len(qinp_vals))*300
time_max_vals = np.ones(len(qinp_vals))*1e5

#qinp_vals    = ["/scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2022-04_01_ITG_scans_stellarators/2022-04-04_ITG_scan_precise_QA/nonlinear_physics/2022-07-20_NL_scans/2022-09-03_aLn-2_phase_shift/run_tube_pos-0.0000_tprim_val-4.0000_phaseshift-0.226317"]
#filename_base = "precise_QA_NL"
#write_text = False
#time_min_vals = np.ones(len(qinp_vals))*150

remove_zonal = False
only_zonal = True
#remove_zonal = True
#only_zonal = False 

logarithmic = True
#logarithmic = False

vmin = 1e-3
#vmax = 50
vmax = None

kx_min =     0#-1
kx_max =   1.0#np.sqrt(0.8)

#if quantity=="pressure_transport":
#    kx_min = 0

#omega_min =   -0.45
#omega_max =   0.45 
#omega_min =    0-0.85
#omega_max =    0.85
omega_min =    -0.1
omega_max =    2

normalise_GAM = False
#normalise_GAM = True

omega_GAM = np.sqrt(7/4+1)

overlay_secondary = False
#overlay_secondary = True
vExP_secondary = [0.35]

fig, axs = plt.subplots(ncols=len(qinp_vals), figsize=(6.5*len(qinp_vals),5), sharey=True)
#fig, axs = plt.subplots(ncols=len(qinp_vals), figsize=(4.5*len(qinp_vals),4.5), sharey=True)

for i_qinp, qinp in enumerate(qinp_vals):

    print(qinp)

    if quantity == "phi":
        title_ax = r"$\big|(\hat v_E^\mathrm{ZF} R/v_{Ti}\rho_i)_{k_x, \omega} \big|$"

    elif quantity == "temperature":
        if kx_order == 1:
            if zed_val == -np.pi/2:
                title_ax = r"$\big|(a \partial_x \delta T_i^\mathrm{Z}/T_i)_{k_x, \omega} (\theta=-\pi/2) \big|$"
            if zed_val == np.pi/2:
                title_ax = r"$\big|(a \partial_x \delta T_i^\mathrm{Z}/T_i)_{k_x, \omega} (\theta=\pi/2) \big|$"
            if zed_val == 0:
                title_ax = r"$\big|(a \partial_x \delta T_i^\mathrm{Z}/T_i)_{k_x, \omega} (\theta=0) \big|$"
        else:
            if zed_val == -np.pi/2:
                title_ax = r"$\big|(\rho_{*i} \delta T_i^\mathrm{Z}/T_i)_{k_x, \omega} (\theta=-\pi/2) \big|$"
            if zed_val == np.pi/2:
                title_ax = r"$\big|(\rho_{*i} \delta T_i^\mathrm{Z}/T_i)_{k_x, \omega} (\theta=\pi/2) \big|$"
            if zed_val == 0:
                title_ax = r"$\big|(\rho_{*i} \delta T_i^\mathrm{Z}/T_i)_{k_x, \omega} (\theta=0) \big|$"

    elif quantity == "dyphi-T":
        if zed_val == -np.pi/2:
            title_ax = r"$(a/\rho_i)^2 \big|(\langle v_{Ex} \delta T_i \rangle_y / v_{Ti}T_i )_{k_x, \omega} (\theta=-\pi/2) \big|$"
        if zed_val == np.pi/2:
            title_ax = r"$(a/\rho_i)^2 \big|(\langle v_{Ex} \delta T_i \rangle_y / v_{Ti}T_i )_{k_x, \omega} (\theta=\pi/2) \big|$"
        if zed_val == 0:
            title_ax = r"$(a/\rho_i)^2 \big|(\langle v_{Ex} \delta T_i \rangle_y / v_{Ti}T_i )_{k_x, \omega} (\theta=0) \big|$"

    elif quantity == "pressure_transport":
        if zed_val == -np.pi/2:
            #title_ax = r"$(a/\rho_i)^2 \big| (Q_\mathrm{NL} / (n_i T_i v_{Ti}) )_{k_x, \omega} (\theta=-\pi/2) \big|$"
            title_ax = r"$\big| (\hat{Q}_\mathrm{NL})_{k_x, \omega} (\theta=-\pi/2) \big|$"
        if zed_val == np.pi/2:
            title_ax = r"$\big| (\hat{Q}_\mathrm{NL})_{k_x, \omega} (\theta=\pi/2) \big|$"
        if zed_val == 0:
            title_ax = r"$\big| (\hat{Q}_\mathrm{NL})_{k_x, \omega} (\theta=0) \big|$"
            #title_ax = r"$(a/\rho_i)^2 \big| (Q_\mathrm{NL} / (n_i T_i v_{Ti}) )_{k_x, \omega} (\theta=0) \big|$"
            #title_ax = r"$(a/\rho_i)^2 \big|(\langle v_{Ex} \delta T_i \rangle_y / v_{Ti}T_i )_{k_x, \omega} (\theta=0) \big|$"

    else:
        title_ax = quantity

    
    title_ax = qinp

    #title_ax = r"$\big|(v_E^\mathrm{Z}/v_{Ti})_{k_x, \omega} \big| R / \rho_i$"
    if plot_qinps:
        dirname =  basedir+"/run_qinp-%.4f" % (qinp)
        if len(qinp_vals)>1:
            title_ax += r" $(q = %.1f)$" % (qinp)
        #title_ax = r"$E_r (q = %.1f)$" % (qinp)
    else:
        dirname =  qinp

    fig.suptitle(title_ax)
    #title_ax = qinp
    title_ax = labels[i_qinp]

    time_min = time_min_vals[i_qinp]
    time_max = time_max_vals[i_qinp]

    #try:
    filename = dirname + "/" + filename_base
    StellaObj = sD.stellaDiagnostics(filename)
    
    if len(qinp_vals) == 1:
        ax = axs
    else:
        ax = axs[i_qinp]
    
    fig, ax, im, kx, omega, f_kx_omega = StellaObj.plot_quantity_kx_omega(quantity=quantity, time_min=time_min, time_max=time_max, fig=fig, ax=ax, remove_zonal=remove_zonal, only_zonal=only_zonal, vmin=vmin, vmax=vmax, logarithmic=logarithmic, kx_order=kx_order, zed_val=zed_val, mult_zed=mult_zed, omega_min=omega_min, omega_max=omega_max, plot_omega2_kx2=plot_omega2_kx2, par_der_order=par_der_order, alt_slow_eval=False, scale_eps=eps, cmap='inferno')#, omega_norm=omega_GAM)

    np.savetxt(dirname+"/data_kx.dat", kx)
    np.savetxt(dirname+"/data_omega.dat", omega)
    np.savetxt(dirname+"/data_f_kx_omega.dat", f_kx_omega)

    ax.set_title(title_ax)#, fontsize=fontsize_text)
    fig.colorbar(im, ax=ax)

    if quantity == "phi" and write_text:
        props = dict(boxstyle='round', facecolor='white')#, alpha=0.5)
        fontsize_text = 20

        #ax.text(0.3175, 1.73, title_ax, verticalalignment='top', bbox=props, fontsize=fontsize_text-2)
        #ax.text(0.32, 0.96, title_ax, verticalalignment='top', bbox=props, fontsize=fontsize_text)

        #ax.text( 0.4, 0.5, r"Toroidal"+'\n'+ r"secondary", c='w', rotation=44*(1+eps), rotation_mode='anchor', transform_rotates_text=True, fontsize=fontsize_text)
        #ax.text( 0.27, 0.07/eps, r"Toroidal secondary", c='w', rotation=44*(1+eps), rotation_mode='anchor', transform_rotates_text=True, fontsize=fontsize_text)
        #ax.text( 0.1, 0.4, r"Toroidal secondary", c='w', rotation=46*(1+eps), rotation_mode='anchor', transform_rotates_text=True, fontsize=fontsize_text) # PRL
        ax.text( 0.06, 0.17, r"Toroidal secondary", c='w', rotation=49*(1+eps), rotation_mode='anchor', transform_rotates_text=True, fontsize=fontsize_text) #PPCF

        #ax.text( 0.35, 0.11, r"$\omega \sim k_x v_{Mx}$", c='w', rotation=45*0.9, rotation_mode='anchor', transform_rotates_text=True)
        ax.text( 0.16, -0.11, r"Stationary ZF", c='w', fontsize=fontsize_text) #PRL
        #ax.text( 0.12, 0.012/eps, r"Stationary ZF", c='w', fontsize=fontsize_text) #PPCF
        ax.text( 0.01, 0.82*omega_GAM, r"GAM", c='w', fontsize=fontsize_text)

    plt.subplots_adjust(wspace=0.06)
    ax.set_xticks([0,0.2,0.4,0.6,0.8])
    #ax.set_xlabel(r"$k_x \rho_i$", labelpad=-15)
    ax.set_xlabel(r"$k_x \rho_i$")

    #try:
    if overlay_secondary:   
        ## Plot gamma-omega secondary
        basedir_secondary = "/scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2022-09-28_secondary/2023-06-17_toroidal_neoclassical_secondary/2023-10-16_scan_kxS_systematic_vary_alphaprp"
        for vExP in vExP_secondary:
            #dir_secondary =  basedir_secondary+"/tpar-0.00_tprp-3.00_phase-0.00_fprim-0.00_tprim-0.00_qinp-%.2f_vExP-%.2f_kyP-0.01/" % (4.2, vExP)
            #dir_secondary =  basedir_secondary+"/tpar-0.00_tprp-3.00_phase-0.00_fprim-0.00_tprim-0.00_qinp-%.2f_vExP-%.2f_kyP-0.01/" % (qinp, vExP)
            dir_secondary =  basedir_secondary+"/tpar-0.00_tprp-3.00_phase-0.00_fprim-0.00_tprim-0.00_qinp-%.2f_vExP-%.2f_kyP-0.01/" % (2.8, vExP)
            k_sec     = np.loadtxt(dir_secondary+"/data_kx.dat")
            omega_sec = np.loadtxt(dir_secondary+"/data_omegar.dat")
            gamma_sec = np.loadtxt(dir_secondary+"/data_omegai.dat")
            ax.plot(k_sec[1:], omega_sec[1:]*omega_GAM, c='w', ls='--')

        #basedir_secondary = "/scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2022-09-28_secondary/2023-07-03_ITG_primary/2024-07-11_scan_kxS_vExP_HR/fprim-0.80_tprim-5.00_qinp-2.80_kyP-0.01"
        ##basedir_secondary = "/scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2022-09-28_secondary/2023-07-03_ITG_primary/2024-07-11_scan_kxS_vExP_HR/fprim-0.80_tprim-5.00_qinp-2.80_kyP-0.05"
        #vEP_idx = 2
        #kxS_vals = np.linspace(0.0125,0.8,64, endpoint=True)
        #omega_vals = np.loadtxt(basedir_secondary+"/data_omega.dat")[:,vEP_idx]
        #gamma_vals = np.loadtxt(basedir_secondary+"/data_gamma.dat")[:,vEP_idx]
        #omega_vals[gamma_vals<1e-2] = 0
        #ax.plot(kxS_vals, omega_vals/eps, c='w', ls='--')

    #except:
    #    continue

    try:

        if normalise_GAM:
            if not plot_omega2_kx2:
                ax.set_xlim([kx_min, kx_max])
                ax.set_ylim([omega_min/omega_GAM, omega_max/omega_GAM])
                if i_qinp==0:
                    ax.set_ylabel(r"$\omega/\omega_\mathrm{GAM}$")
                else:
                    ax.set_ylabel(None)
            else:
                ax.set_xlim([0, kx_max**2])
                ax.set_ylim([0, (omega_max/omega_GAM)**2])
                ax.set_ylabel(r"$\omega^2/\omega^2_\mathrm{GAM}$")

        else:
            if not plot_omega2_kx2:
                ax.set_xlim([kx_min, kx_max])
                if i_qinp == 0:
                    labelpad = -15
                else:
                    labelpad = 0
                ax.set_ylabel(r"$\omega R/v_{Ti}$", labelpad=labelpad)
                #ax.set_ylim([omega_min, omega_max])
                #omega_th = np.linspace(omega_min, omega_max, 1000)
                #ax.plot(omega_th, 1.25*omega_th, ls='--', c='g', lw=3)
        
                #ax.axhline(omega_GAM, ls='--', c='white', lw=3, alpha=0.5)
                #ax.text(0.4, omega_GAM*1.03, r"$\omega_\mathrm{GAM}$", c='white')#, rotation=90)
        
                #omega_th = np.linspace(0, omega_max, 1000)
                #ax.plot(1.40*omega_th, omega_th, ls='--', c='white', lw=3, alpha=0.5)
    
            else:
                ax.set_xlim([0, kx_max**2])
                ax.set_ylim([0, omega_max**2])
                ax.axhline(omega_GAM**2, ls='--', c='white', lw=3, alpha=0.5)
                ax.text(0.4, omega_GAM**2*1.03, r"$\omega_\mathrm{GAM}$", c='white', alpha=0.5)#, rotation=90)
#
#        omega_GAM = np.sqrt(7/4)/2.8
#        ax.axvline(omega_GAM, ls='--', c='white', lw=3)
    except:
        continue
#    ax.set_xlim(xmin=ommin)
    
    #except:
    #    continue

#if kx_order == 1:
#    title = r"$\partial_x$"
#elif kx_order == 2:
#    title = r"$\partial^2_x$"
#else:
#    title = ""
#
#if quantity == "phi":
#    title += r"$\varphi$"
#elif quantity == "temperature":
#    title += r"$T$"
#elif quantity == "upar":
#    title += r"$u_\parallel$"
#elif quantity == "pressure":
#    title += r"$P$"
#
#if remove_zonal:
#    title = title+r"$_\mathrm{NZ}$"
#elif only_zonal:
#    title = title+r"$_\mathrm{Z}$"
#
#fig.suptitle(title)
#if logarithmic:
#    fig.subplots_adjust(right=0.8)
#    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#    fig.colorbar(im, cax=cbar_ax)

figname = "fig_contours_"
figname = figname+quantity
figname=figname+"_kx_omega"

if kx_order > 0:
    figname = figname + "_kx-order-%i" % (kx_order)

if remove_zonal:
    figname = figname + "_remove_zonal"
if only_zonal:
    figname = figname + "_only_zonal"

if mult_zed is not None:
    figname = figname + "_mult-zed-" + mult_zed

if zed_val is not None:
    figname = figname + "_zed_val-%.2f"  % (zed_val)

if overlay_secondary:
    figname += "_overlay-secondary"

plt.tight_layout()
plt.savefig(figname+figname_add+".pdf")
