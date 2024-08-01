import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from util import *
from algorithms import *


# Algorithmic parameters
niter = int(50000)              # Number of iterations of the Markov chain
n_parallel_chains = int(1e4)    # Number of parallel Markov chains to simulate
metropolis_check = False        # Wether to include a Metropolis correction step which ensures to exactly approximate the target
save_every = 250                # We save the current samples every save_every iterations
check = 0                       # If check is a positive, natural number, every check iterations we plot the current samples
average_distribution = False    # Wether to track the mean distribution according to denoted by Greek nu in the paper


methods = ['prox_subgrad', 'grad_subgrad', 'MYULA', 'MALA'] # subgrad

tau_list = [1e-5,1e-4,1e-3]




# Target distribution parameters
d = 2
gt = np.array([-1,1])
np.random.seed(0)

sigma = 1
u0 = gt

data_par = 1/(sigma**2)
reg_par = 5
L_F = data_par
L_G = reg_par*np.sqrt(d)
K_nrm = 2



folder = 'results/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = 'results/2d_ex/'
if not os.path.exists(folder):
	os.makedirs(folder)

base_folder = folder+'l2/'
if not os.path.exists(folder):
    os.makedirs(folder)


for method in methods:

    folder = base_folder + method + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    if method == 'grad_subgrad':
            
        for tau in tau_list:
            print('TAU')
            print(tau)

            algo = subgradient_Langevin(u0=u0,niter =niter,n_parallel_chains=n_parallel_chains,
                                    metropolis_check=metropolis_check,folder=folder,check=check,save_every=save_every,
                                    reg_par=reg_par, data_par=data_par, tau=tau)

            res = algo.grad_subgrad(average_distribution=average_distribution)


    elif method == 'prox_subgrad':

        for tau in tau_list:
            print('TAU')
            print(tau)

            algo = subgradient_Langevin(u0=u0,niter =niter,n_parallel_chains=n_parallel_chains,
                                    metropolis_check=metropolis_check,folder=folder,check=check,save_every=save_every,
                                    reg_par=reg_par, data_par=data_par, tau=tau)


            res = algo.prox_subgrad(average_distribution=average_distribution)

    elif method == 'subgrad':
        for tau in tau_list:
            algo = subgradient_Langevin(u0=u0,niter=niter,n_parallel_chains=n_parallel_chains,metropolis_check=metropolis_check,
                                        folder=folder,check=check,save_every=save_every,
                                        reg_par=reg_par, data_par=data_par,tau=tau)

            res = algo.subgrad(burnin=0)

    elif method == 'MYULA':
        for tau in tau_list:
            ld = data_par/100
            print('TAU')
            print(tau)
            algo = MYULA(u0=u0,niter =niter, tau = tau, ld = ld,n_parallel_chains=n_parallel_chains,
                                folder=folder,check=check,save_every=save_every,
                                reg_par=reg_par, data_par=data_par)

            res = algo.sample()


    elif method == 'MALA':
        for tau in tau_list:
            print('TAU')
            print(tau)
            algo = MALA(u0=u0,niter =niter, tau = tau,n_parallel_chains=n_parallel_chains,
                                folder=folder,check=check,save_every=save_every,
                                reg_par=reg_par, data_par=data_par)

            res = algo.sample()

    else:
        print('Method not valid')





