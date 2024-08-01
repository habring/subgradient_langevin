import numpy as np
import matplotlib.pyplot as plt
from util import *
from algorithms import *
from PIL import Image
import os

np.random.seed(0)

sigma = 0.01

niter = int(1e6+500000+1)
n_parallel_chains = int(1)
metropolis_check = False
save_every = 5000
check = 0
burnin = 1e6
prox = False

### blur kernel

sz = 5
sig = 1
mu = 0
l1 = np.linspace(-1,1,sz)
l2 = np.linspace(-1,1,sz)
a1,a2 = np.meshgrid(l1,l2)
gauss_kernel = ( 1.0/np.sqrt(2.0*np.pi*sig*sig) )*np.exp( -(np.square(a1-mu) + np.square(a2-mu))/(2.0*sig*sig))
gauss_kernel = gauss_kernel/np.sum(gauss_kernel)

if metropolis_check:
    tau_list = [1e-6]
    niter = int(3e6+1e6+1)
    burnin = 3e6
else:
    tau_list = [1e-6,1e-5]

reg_par_list = [20,30,10,15,50]
reg_par_list = [20]


folder = 'results/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = 'results/image_ex/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = folder + 'deconv_kernelsize_5/'
if not os.path.exists(folder):
    os.makedirs(folder)


data_par = 1/(sigma**2)

if not prox:

    folder = folder + 'grad_subgrad/'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    if metropolis_check:
        folder = folder + 'metro/'
        if not os.path.exists(folder):
            os.makedirs(folder)

    for reg_par in reg_par_list:
        for tau in tau_list:

            for name in ['barbara','peppers_bw']:

                folder_im = folder + name+'/'
                if not os.path.exists(folder_im):
                    os.makedirs(folder_im)

                u0 = np.load('images/'+name+'.npy')

                # blur:
                u0 = scipy.signal.convolve2d(u0,gauss_kernel,mode='same',boundary='wrap')
                u0 = u0+sigma*np.random.normal(size=u0.shape)
                algo = subgradient_Langevin(u0=u0,F_name = '2d_l2blur',blur_kernel=gauss_kernel,niter =niter,n_parallel_chains=n_parallel_chains,
                                        metropolis_check=metropolis_check,folder=folder_im,check=check,save_every=save_every,
                                        reg_par=reg_par, data_par=data_par, tau=tau)

                res = algo.grad_subgrad(burnin=burnin)


else:
    folder = folder + 'prox_subgrad/'
    if not os.path.exists(folder):
        os.makedirs(folder)
        

    for reg_par in reg_par_list:
        for tau in tau_list:

            for name in ['peppers_bw','barbara']:

                folder_im = folder + name+'/'
                if not os.path.exists(folder_im):
                    os.makedirs(folder_im)

                u0 = np.load('images/'+name+'.npy')

                # blur:
                u0 = scipy.signal.convolve2d(u0,gauss_kernel,mode='same',boundary='wrap')
                u0 = u0+sigma*np.random.normal(size=u0.shape)

                algo = subgradient_Langevin(u0=u0,F_name = '2d_l2blur',blur_kernel=gauss_kernel,niter =niter,n_parallel_chains=n_parallel_chains,
                                        metropolis_check=metropolis_check,folder=folder_im,check=check,save_every=save_every,
                                        reg_par=reg_par, data_par=data_par, tau=tau)

                res = algo.prox_subgrad(burnin=burnin)
