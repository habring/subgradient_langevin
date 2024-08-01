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
burnin=0                        # Wether to track the mean distribution according to denoted by Greek nu in the paper


prox = True                     # Use prox-sub or subgrad


tau_list = [1e-5,1e-4,1e-3,1e-2]   # step sizes to use.


# Example parameters
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

folder = folder+'l1/'
if not os.path.exists(folder):
	os.makedirs(folder)


if prox:

    folder = folder + 'prox_subgrad/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for tau in tau_list:
        algo = subgradient_Langevin(u0=u0,F_name = 'l1',niter=niter,n_parallel_chains=n_parallel_chains,
                                metropolis_check=metropolis_check,folder=folder,check=check,save_every=save_every,
                                reg_par=reg_par, data_par=data_par, tau=tau)
        res = algo.prox_subgrad(average_distribution=True,burnin=burnin)
else:

    folder = folder + 'subgrad/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for tau in tau_list:
        algo = subgradient_Langevin(u0=u0,niter=niter,n_parallel_chains=n_parallel_chains,metropolis_check=metropolis_check,
                                    folder=folder,check=check,save_every=save_every,
                                    F_name='l1', 
                                    reg_par=reg_par, data_par=data_par,tau=tau)

        res = algo.subgrad(burnin=burnin)