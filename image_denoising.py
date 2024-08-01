import numpy as np
import matplotlib.pyplot as plt
from util import *
from algorithms import *
from PIL import Image
import os


sigma = 0.05
data_par = 1/(sigma**2)
niter = int(1e6+100000+1)
n_parallel_chains = int(1)
metropolis_check = False
save_every = 500
check = 0
burnin = 1e6
prox = True

if metropolis_check:
    tau_list = [1e-6]
    niter = int(3e6+1e6+1)
    burnin = 3e6
else:
    tau_list = [1e-6,1e-5,1e-4]

reg_par_list = [10,15,20,30,50]
reg_par_list = [30]




folder = 'results/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = 'results/image_ex/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = folder+'denoising/'
if not os.path.exists(folder):
    os.makedirs(folder)


if not prox:

    folder = folder + 'grad_subgrad/'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    if metropolis_check:
        folder = folder + 'metro/'
        if not os.path.exists(folder):
            os.makedirs(folder)

    for tau in tau_list:
        for reg_par in reg_par_list:

            for name in ['barbara','peppers_bw']:

                folder_im = folder + name+'/'
                if not os.path.exists(folder_im):
                    os.makedirs(folder_im)

                u0 = np.load('images/'+name+'_noisy.npy')

                algo = subgradient_Langevin(u0=u0,niter =niter,n_parallel_chains=n_parallel_chains,
                                        metropolis_check=metropolis_check,folder=folder_im,check=check,save_every=save_every,
                                        reg_par=reg_par, data_par=data_par, tau=tau)

                res = algo.grad_subgrad(burnin=burnin)


else:

    folder = folder + 'prox_subgrad/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for tau in tau_list:
        for reg_par in reg_par_list:

            for name in ['barbara','peppers_bw']:

                folder_im = folder + name+'/'
                if not os.path.exists(folder_im):
                    os.makedirs(folder_im)

                u0 = np.load('images/'+name+'_noisy.npy')

                algo = subgradient_Langevin(u0=u0,niter =niter,n_parallel_chains=n_parallel_chains,
                                        metropolis_check=metropolis_check,folder=folder_im,check=check,save_every=save_every,
                                        reg_par=reg_par, data_par=data_par, tau=tau)


                res = algo.prox_subgrad(burnin=burnin)