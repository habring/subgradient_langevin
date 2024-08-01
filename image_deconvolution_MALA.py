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
save_every = 5000
check = 0
burnin = 1e6

### blur kernel

sz = 5
sig = 1
mu = 0
l1 = np.linspace(-1,1,sz)
l2 = np.linspace(-1,1,sz)
a1,a2 = np.meshgrid(l1,l2)
gauss_kernel = ( 1.0/np.sqrt(2.0*np.pi*sig*sig) )*np.exp( -(np.square(a1-mu) + np.square(a2-mu))/(2.0*sig*sig))
gauss_kernel = gauss_kernel/np.sum(gauss_kernel)

tau_list = [1e-7,1e-6,1e-5]

data_par = 1/(sigma**2)

reg_par_list = [10,15,20,30,50]
reg_par_list = [20]


folder = 'results/'
if not os.path.exists(folder):
    os.makedirs(folder)

folder = 'results/image_ex/'
if not os.path.exists(folder):
    os.makedirs(folder)

folder = folder+'deconv_kernelsize_5/'
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
            
            u0 = np.load('images/'+name+'.npy')

            u0 = scipy.signal.convolve2d(u0,gauss_kernel,mode='same',boundary='wrap')
            u0 = u0+sigma*np.random.normal(size=u0.shape)

            algo = MALA(u0=u0,niter =niter, tau = tau, F_name = '2d_l2blur',blur_kernel=gauss_kernel,
                        folder=folder_im,check=check,save_every=save_every,
                        reg_par=reg_par, data_par=data_par)

            res = algo.sample(burnin=burnin)







































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



tau_list = [1e-7,1e-6,1e-5]

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
