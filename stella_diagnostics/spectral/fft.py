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


def get_Wigner_x_kx(f_x):


    tfr = WignerVilleDistribution(f_x)
    tfr.run()
    tfr.plot(kind='contour', show_ft=True)
