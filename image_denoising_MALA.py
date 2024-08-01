import numpy as np
import matplotlib.pyplot as plt
from util import *
from algorithms import *
from PIL import Image
import os

sigma = 0.05
data_par = 1/(sigma**2)
niter = int(1e6+100000+1)
save_every = 500
check = 0
burnin = int(1e6)

tau_list = [1e-6,1e-5,1e-4]


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

folder = folder + 'MALA/'
if not os.path.exists(folder):
    os.makedirs(folder)


for tau in tau_list:
    for reg_par in reg_par_list:

        for name in ['barbara','peppers_bw']:

            folder_im = folder + name+'/'
            if not os.path.exists(folder_im):
                os.makedirs(folder_im)

            u0 = np.load('images/'+name+'_noisy.npy')
            #u0 = u0[50:,50:]

            algo = MALA(u0=u0,niter =niter, tau = tau,
                        folder=folder_im,check=check,save_every=save_every,
                        reg_par=reg_par, data_par=data_par)

            res = algo.sample(burnin=burnin)
