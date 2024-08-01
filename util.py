import numpy as np
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d #For surface plot
import scipy.signal
import sys #For error handling
#For saving data
import copyreg
import types
#import cPickle as pickle
import pickle
import imageio
from scipy import linalg
from scipy.stats import gmean

import matplotlib.colors as clr
from scipy.fftpack import dct, idct
import matplotlib.image as mpimg
import random

import scipy.ndimage.interpolation as intp

from IPython import get_ipython

import copy

import os
#For iterating dicts
from itertools import product

#Initialization
if __name__ == "__main__":
    
    #Set autoreload
    ipython = get_ipython()
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
    
    #Load main libraries (this requires to set the PYTHONPATH, see info.txt)
    import matpy as mp



### Helper functions ####################################################################
#########################################################################################


### Data I/O ############################################################################

#Class for output variables
#Note that res.__dict__ converts to dict
class parout(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class parameter(object):
    
    def __init__(self,adict):
        self.__dict__.update(adict)
    
    #Define print function
    def __str__(self):
        #self.__dict__: Class members as dict
        #.__str__() is theprint function of a class
        
        #Above: standard dict print, below: print every entry in new line
        #return self.__dict__.__str__()
        return "{\n" + "\n".join("{!r}: {!r},".format(k, v) for k, v in self.__dict__.items()) + "\n}"
        
class timing_object(object):
    
    def __init__(self,adict):
        self.__dict__.update(adict)
    
    #Define print function
    def __str__(self):
        return "{\n" + "\n".join("{!r}: {!r},".format(k, np.round(v,4)) for k, v in self.__dict__.items()) + "\n}"

        
class data_input(object):
    
    def __init__(self,adict):
        self.__dict__.update(adict)
    
    #Define print function
    def __str__(self):
        #self.__dict__: Class members as dict
        #.__str__() is theprint function of a class
        return self.__dict__.__str__()




class output(object):
    
    def __init__(self,par=parameter({})):
        
        self.par = par

    def output_name(self,outpars=[],fname='',folder='results'):
    
        #Try to get filename
        if not fname:
            if hasattr(self.par,'imname'):
                fname = self.par.imname[self.par.imname.rfind('/')+1:]
            elif hasattr(self.par,'signal_name'):
                fname = self.par.signal_name[self.par.signal_name.rfind('/')+1:]
            else:
                raise NameError('No filename given')

        #Generate folder if necessary
        if folder:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # The following was commented by Andi, in order to be able to save with names containing '.'
        #Remove ending if necessary
        # pos = fname.find('.')
        # if pos>-1:
        #     fname = fname[:pos]
        
        #Concatenate folder and filename
        outname = fname
        if folder:
            outname = folder + '/' +  outname
            #Remove double //
            outname = outname.replace('//','/')
        
        #Check for keyword DOT
        if outname.find('DOT')>-1:
            raise NameError('Keyword "DOT" not allowd')
        
        #If outpars are not given, try to generate them from par_in
        if 0:# not outpars: OPTION DEACTIVATED
            if hasattr(self,'par_in'):
                for key,val in self.par_in.items():
                    if isinstance(val, (int, float)): #Only including numbers
                        outpars.append(key)
            else:
                print('No parameters for filename given')
                
        #Add outpars to filename
        for par in outpars:
            if hasattr(self.par,par):
                val = self.par.__dict__[par]
                #exec('val = self.par.'+par)
                outname += '__' + par + '_' + num2str(val)
            else:
                raise NameError('Non-existent parameter: ' + par)

        
        return outname        

    def save(self,outpars=[],fname='',folder=''):
    
        #Get name
        outname = self.output_name(outpars,fname,folder)
        #Save
        psave(outname,self)

    def show(self):
    
        print('Function "show" not initialized.')

#Class for parameter testing
class partest(object):

    def __init__(self,method,fixpars={},testpars={},namepars=[],folder=''):
    
        
        self.method = method
        self.fixpars = fixpars
        self.testpars = testpars
        self.namepars = namepars
        self.folder = folder
        
    def run_test(self):
    
        #Check for conflicts
        for key in self.testpars.keys():
            if key in self.fixpars:
                raise NameError('Double assignement of ' + key)
                
        #Get keys
        testkeys = self.testpars.keys()
        #Iterate over all possible combinations
        for valtuple in list(product(*self.testpars.values())):
            
            #Set test values
            for key,val in zip(testkeys,valtuple):
                self.fixpars[key] = val
                
                
            #Print parameter setup
            print('Testing: ')
            print(self.fixpars)
            #Get result
            res = self.method(**self.fixpars)
            #Save
            res.save(outpars=self.namepars,folder=self.folder)
                
def read_file(basename,pars={},folder='.',flist=[]):

    if not flist:
        flist = os.listdir(folder)
    
    flist = [ fl for fl in flist if basename in fl ]
    for key,val in pars.items():
        flist = [fl for fl in flist if '_' + key + '_' + num2str(val) in fl]
    
    if len(flist)>1:
        print('Warning: non-unique file specification. Reading first occurence')
        flist = [flist[0]]
   
    fname = folder + '/' + flist[0]
    #Remove double //
    fname = fname.replace('//','/')    
    return pload(fname)
    
    
#Return all file names with .pkl extension matching a parameter combination
def get_file_list(basename,pars = {},folder = '.'):


    flist = os.listdir(folder)


    
       
    #Remove non-matching filenames
    for fname in flist[:]:
        if (basename not in fname) or ('.pkl' not in fname): #Basename
            flist.remove(fname)
        else:
            for par in pars.keys():
                #Check parameter name
                if '_' + par + '_' not in fname:
                    flist.remove(fname)
                    break
                else:
                    #Check parameter values
                    valcount = len(pars[par])
                    if valcount>0:
                        for val in pars[par]:
                            if '_' + par + '_' + num2str(val) not in fname: #Parameter value pairs
                                valcount -= 1
                        if valcount == 0: #If no parameter is present
                            flist.remove(fname)
                            break


    return flist
                

#Return a list of file names with .pkl extension matching a parameter combination together with the parameters
def get_file_par_list(basename,pars = {},folder = '.'):

    #Get list of files matching pattern
    flist = get_file_list(basename,pars = pars,folder = folder)
    
    parnames = list(pars)
    parvals = []
    for fname in flist:
        parval = []
        for parname in parnames:
            parval.append(read_parval(fname,parname))
        parvals.append(parval[:])
    
    return flist,parnames,parvals

#Get data with best psnr in "folder" mathing a given pattern. Assuming "orig" and "u" to be available
def get_best_psnr(basename,pars={},folder='.',rescaled=True):

    #Get sortet list of filenames, parnames and values
    flist = get_file_list(basename,pars=pars,folder=folder)

    

    opt_psnr = 0.0
    for fname in flist:
    
        fullname = folder + '/' + fname
        fullname = fullname.replace('//','/') 
        
        res = pload(fullname)
        
        c_psnr = psnr(res.u,res.orig,smax = np.abs(res.orig.max()-res.orig.min()),rescaled=rescaled)
        
        if c_psnr > opt_psnr:
            opt_psnr = c_psnr
            opt_fname = fullname
            
    res = pload(opt_fname) 
    
    print('Best psnr: ' + str(np.round(opt_psnr,decimals=2)))
    
    return res

        
#Read value of parameter from file        
def read_parval(fname,parname):

    #Set position of value    
    star = fname.find('_'+parname+'_')+len('_'+parname+'_')
    #Set end position of value
    end = fname[star:].find('__')
    if end == -1:
        end = fname[star:].find('.')
    end += star 
    
    return str2num(fname[star:end])
            

#Convert number to string and reverse
def num2str(x):
    return str(x).replace('.','DOT')


def str2num(s):
    return float(s.replace('DOT','.'))


#Function to parse the arguments from par_in        
#Take a par_in dict and a list of parameter classes as input
#Sets the class members all elements of parlist according to par_in
#Raises an error when trying to set a non-existing parameter
def par_parse(par_in,parlist):
    
    if not isinstance(par_in,dict):
        par_in = par_in.__dict__
    
    for key,val in par_in.items():
        foundkey = False
        for par in parlist:
            if key in par.__dict__:
                par.__dict__[key] = val
                foundkey = True
        if not foundkey:
            raise NameError('Unknown parameter: ' + key)

#Data storage
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
        
    return func.__get__(obj, cls)

#Currently not used, only relevant as noted in psave
def pickle_get_current_class(obj):
    name = obj.__class__.__name__
    module_name = getattr(obj, '__module__', None)
    obj2 = sys.modules[module_name]
    for subpath in name.split('.'): obj2 = getattr(obj2, subpath)
    return obj2

def psave(name,data):
    
    #This might potentially fix erros with pickle and autoreload...try it next time the error ocurs
    data.__class__ = pickle_get_current_class(data)
    
    copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
    
    if not name[-4:] == '.pkl':
        name = name + '.pkl'
        
    output = open(name,'wb')
    # Pickle the list using the highest protocol available.
    pickle.dump(data, output, -1)
    output.close()
    
def pload(name):

    copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
    
    if not name[-4:] == '.pkl':
        name = name + '.pkl'
        
    
    try:
        pkl_file = open(name, 'rb')
        data = pickle.load(pkl_file)
    except:
        pkl_file.close()


        #print('Standard loading failed, resorting to python2 compatibility...')
        pkl_file = open(name, 'rb')
        data = pickle.load(pkl_file,encoding='latin1')

    pkl_file.close()
    return data


def server_transfer(*args,**kwargs):


    try:
        from server_transfer import server_transfer as st
        
        st(*args,**kwargs)
        
    except:
        print('Error: Sever transfer function not available')

        
    



### Plotting ############################################################################

def imshow(x,stack=1,fig=0,title=0,colorbar=1,cmap='gray',vrange=[]):


    try:

        if x.ndim>2 and stack:
            x = imshowstack(x)

        if not fig:
            fig = plt.figure()
            
        plt.figure(fig.number)
        if not vrange:
            plt.imshow(x,cmap=cmap,interpolation='none')
        else:
            plt.imshow(x,cmap=cmap,vmin=vrange[0],vmax=vrange[1],interpolation='none')
        if colorbar:
            plt.colorbar()
        if title:
            plt.title(title)
        fig.show()
        
    except:
        print('Display error. I assume that no display is available and continue...')
        fig = 0
    
    return fig



def plot(x,y=0,fig=0,title=0,label=0,linestyle='-'):

        

    try:
        if not fig:
            fig = plt.figure()
        plt.figure(fig.number)
        
        if not np.any(y):
            plt.plot(x,label=label,linestyle=linestyle)
        else:
            plt.plot(x,y,label=label,linestyle=linestyle)
            
        if title:
            plt.title(title)
            
        if label:
            plt.legend()


        fig.show()
        

        
    except:
        print('Display error. I assume that no display is available and continue...')
        fig = 0
    
    return fig
    
    
def surf(x,y=0,z=0,fig=0,title=0,label=0):

    try:
        if not fig:
            fig = plt.figure()
        plt.figure(fig.number)
        
        ax = fig.gca(projection='3d')
        
        if not np.any(y):
            ax.plot_surface(x,label=label)
        else:
            ax.plot_surface(x,y,z,label=label)
            
        if title:
            plt.title(title)
            
        if label:
            plt.legend()


        fig.show()
   
    except:
        print('Display error. I assume that no display is available and continue...')
        fig = 0
    
    return fig

#Stack a 3D array of images to produce a 2D image
#Optinal input: nimg = (n,m). Take n*m images and arrange them as n x m
def imshowstack(k,nimg = ()):

    N,M = k.shape[0:2]
    nk = k.shape[-1]

    if nimg:
        nx = nimg[1]
        ny = nimg[0]
    else:

        nx = np.ceil(np.sqrt(np.copy(nk).astype('float')))
        ny = np.ceil(nk/nx)

        nx = int(nx)
        ny = int(ny)

    if k.ndim == 3:
        kimg = np.zeros([N*ny,M*nx])
        for jj in range(ny):
            for ii in range(nx):
                    if ii + nx*jj < nk:
                        kimg[jj*N:(jj+1)*N,M*ii:M*(ii+1)] = k[...,ii + nx*jj]
    else:
        kimg = np.zeros([N*ny,M*nx,k.shape[2]])
        for ll in range(k.shape[2]):
            for jj in range(ny):
                for ii in range(nx):
                        if ii + nx*jj < nk:
                            kimg[jj*N:(jj+1)*N,M*ii:M*(ii+1),ll] = k[...,ll,ii + nx*jj]
    
    
    return kimg


def vecshow(z,step=1):

    #Optional argument: Take only every step'th entry

    fig = plt.figure()
    plt.quiver(z[::step,::step,0],z[::step,::step,1])
    fig.show()
    return fig

def veccolor(z,fig=0,title=0):

    if z.ndim>3:
        z = imshowstack(z)
    
    n = z.shape[0]
    m = z.shape[1]

    
    p = np.zeros([z.shape[0],z.shape[1],3])
    p[...,0] = (np.arctan2(z[...,1],z[...,0])/(2.0*np.pi)) + 0.5
    nz = np.sqrt(np.square(z).sum(axis=2))
    p[...,1] = nz/np.maximum(nz.max(),0.00001)
    p[...,2] = 1.0


    psz = 4
    l1 = np.linspace(-1,1,n+2*psz)
    l2 = np.linspace(-1,1,m+2*psz)
    a1,a2 = np.meshgrid(l2,l1)
    
    c = np.zeros( (n+2*psz,m+2*psz,3))
    
    c[...,0] = (np.arctan2(a1,a2)/(2.0*np.pi)) + 0.5
    c[...,1] = 1.0
    c[...,2] = 1.0
    
    c[psz:-psz,psz:-psz,:] = p
    
   
    fig = imshow(clr.hsv_to_rgb(c),stack=0,fig=fig,title=title,colorbar=0)
    return fig



def closefig():
    plt.close('all')

def rgb2gray(rgb):

    return 0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]


def imread(imname):
    
    return imageio.imread(imname).astype('float')/255.0  


#Function to scale image to [0,1]. Range defines current image range (default: [img.min(),img.max()], values above and below will be cliped
def imnormalize(img,rg=[]):
    
    if not rg:
        rg = [img.min(),img.max()]
        

    #Clip range boundaries
    img = np.clip(np.copy(img.astype('float')),rg[0],rg[1])
    
    #Convert rage to [0,1]
    img = img - rg[0]
    if (rg[1]-rg[0])>0:
        img = img/(rg[1]-rg[0])
    elif np.any(img):
        #raise ValueError('Function requires rg[0]<rg[1]')        
        print('image is constant')
    else:
        print('Warning: empty image, ignoring range argument, no normalization carried out')



    return img
    
def imsave(fname,img,format=None,rg=[0,1]): #rg defines grayscale boundary values. Choosing rg=[] uses [img.min(),img.max()]

    img = imnormalize(img,rg=rg)
    
    imageio.imwrite(fname,(255.0*img).astype('uint8'),format=format)



### Numerical ###########################################################################
def dct2 (block):
  return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')    

def idct2 (block):
  return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')


def mse(u,u0,rescaled=False):

    c = 1.0
    if rescaled:
        c = (u*u0).sum()/np.square(u).sum()

    return np.square(c*u-u0).sum() / np.square(u0).sum()

def psnr(u,u0,smax=1.0,rescaled=False):

    c = 1.0
    if rescaled:
        c = (u*u0).sum()/np.square(u).sum()


    N = np.prod(u.shape).astype('float')
    err = np.square(c*u-u0).sum()/N
    
    return 20.0*np.log10( smax/ np.sqrt(err) )


#Takes a matrix of norms as input and computes the norm of the resulting block-operator
#Example: nr = get_product_norm([[np.sqrt(8),1],[0,np.sqrt(8)]]) returns the norm of operator [[grad,-1],[0,symgrad]]
def get_product_norm(A):
    
    A = np.array(A)
    s = scipy.linalg.svd(A,compute_uv=0)
    return s[0]
    
def fgauss(sz,mu,sig):

    l1 = np.linspace(-1,1,sz)
    l2 = np.linspace(-1,1,sz)
    a1,a2 = np.meshgrid(l1,l2)

    return ( 1.0/np.sqrt(2.0*np.pi*sig*sig) )*np.exp( -(np.square(a1-mu) + np.square(a2-mu))/(2.0*sig*sig))

#All imput array must be odd
def f_sinc(x):
    sz = x.shape
    l1 = np.linspace(-2,2,sz[0])
    l2 = np.linspace(-2,2,sz[1])
    a1,a2 = np.meshgrid(l1,l2)
    
    z = np.sinc(a1)*np.sinc(a2)
    return z  - (z.sum()/(sz[0]*sz[1]))



def get_circle(sz=128,r=0.8,sharp=0):
    
    if not sharp:
        sharp = sz*0.5

    l1 = np.linspace(-1,1,sz)
    l2 = np.linspace(-1,1,sz)
    a1,a2 = np.meshgrid(l1,l2)

    rad = np.sqrt( np.square(a1) + np.square(a2))

    z = np.maximum(0.0,np.minimum(1.0,sharp*(r-rad)))
#    z = np.zeros([sz,sz])
#    z[rad<=r] = 1.0
    
    return z
    


### Algorithmic functions ###############################################################
#########################################################################################

def dct2 (block):
  return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')    

def idct2 (block):
  return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

#Straightforwad implementation of nuclear norm prox for single matrix
def nucprox(A,sig):

    U,s,V = scipy.linalg.svd(A,full_matrices=0)
    
    s = shrink(s,sig)
    S = np.diag(s)
    
    return np.dot(U, np.dot(S, V))

def nucnorm(A):

    return scipy.linalg.svd(A,compute_uv=0).sum()


#Get rank of matrix
def rank(A,tol=1e-10,show_sval=False):

    sig = scipy.linalg.svd(A,compute_uv=0)
    
    rank = sum(sig>tol)
    
    if show_sval:
        print(sig)
    
    return rank,sig

#Compute prox of dual function based on primal prox
#Input: x 
#       prox proximal mapping, takes prox parameter as second argument
#       tau prox parameter
def prox_dual(x,prox,ppar=1.0):

    return x - ppar*prox(x/ppar,ppar=1.0/ppar)

#Projection to the l1 ball of radius tau
#nx gives the number of dimensions along which to project, counted from the last
def project_l1(y,tau=1.0,nx=1):

    #Get dimension of flat vector
    K = np.prod(y.shape[-nx:])
    
    #Store original shape
    s0 = y.shape

    #Reshape to N x K array
    y = np.reshape(y,(-1,K))
    
    #Get pointwise value function    
    t = np.maximum(np.amax( (np.cumsum(np.sort(np.abs(y))[:,::-1],axis=1) - tau)/(1+np.arange(y.shape[-1])), axis=1),0)

    #Solve for prox and return
    return np.reshape(np.sign(y)*np.maximum(np.abs(y) - np.expand_dims(t,1),0),s0)


#Proximal mapping of the convex conjugate of mu|.|_1
#The prox parameter tau is not needed
def proxl1s(p,mu=1.0,dims=None,vdims=(),copy=True,tau=1.0):

    if copy:
        z = np.copy(p)
    else:
        z = p

    if np.sum(mu) not in [0,np.inf]:
        if vdims:
            z/= np.maximum( np.sqrt(np.square(p).sum(axis=vdims,keepdims=True) )/mu , 1.0)
        else:
            z /= np.maximum( np.abs(p)/mu , 1.0)
    elif mu == 0:
        z = np.zeros(z.shape)
    return z

#Proximal mapping of tau|.|_1, i.e., computes (I + tau|.|_1)^(-1)
def shrink(p,tau=1.0,mshift=0.0,vdims=(),copy=True,vecweights=False):

    #Note that we skip -mshift + mshift
    #Using prox_f(x) = prox_0(x-f)+f
    return p - proxl1s((p-mshift),tau,vdims=vdims,copy=copy) #Moreau's Identity


#Computes (I + tau*DF)^(1-) with F(u) = 1/2|u-f|_2^2
#Does not modify the input
def proxl2f(u,f=0,tau=1.0):

    return (u+tau*f) / (1.0+tau)

#Computes (I + tau*DF)^(1-) with F(u) = 1/2|K*u-f|_2^2 where K*u is pointwise multiplikation
#Does not modify the input
def proxl2f_with_forward_operator(u,K=1,f=0,tau=1.0):

    return (u+tau*K*f) / (1.0+tau*K**2)

#Same as above, but dual
def proxl2fs(u,mshift=0,npar=1.0,ppar=1.0):

    return (u-ppar*mshift) / (1.0+(ppar/npar))


def l2nsq(x,mshift=0.0,dims = None):
    return np.square(np.abs(x-mshift)).sum(axis = dims,keepdims=True)

def huber_loss(x,mshift=0.0,alpha = 1.0, dims = None,verbose = False):
    
    nrm_sq = l2nsq(x,mshift=mshift, dims = dims)
    nrm_leq_alpha = 1.0*(nrm_sq<=alpha**2)
    res = (nrm_sq/(2*alpha))*nrm_leq_alpha + (np.sqrt(nrm_sq)-alpha/2.0)*(1-nrm_leq_alpha)
    if verbose:
        print('both included?')
        print(nrm_leq_alpha.mean())
    return res

def huber_grad(x,mshift=0.0,alpha = 1.0, dims = None,verbose = False):
    
    nrm_sq = l2nsq(x,mshift=mshift, dims = dims)
    nrm_leq_alpha = 1.0*(nrm_sq<=alpha**2)
    nrm_sq_at_small_vals = nrm_sq + 1.0*nrm_leq_alpha
    grad_res = ((x-mshift)/alpha)*nrm_leq_alpha + ( (x-mshift)/(np.sqrt(nrm_sq_at_small_vals)) )*(1-nrm_leq_alpha)

    if verbose:
        print('both included?')
        print(nrm_leq_alpha.mean())

    return grad_res

def huber_prox(x,mshift=0.0,tau=1.0,alpha = 1.0, dims = None,verbose = False):
    
    nrm_sq = l2nsq(x,mshift=mshift, dims = dims)
    nrm_leq_alpha = 1.0*(nrm_sq<=(tau+alpha)**2)
    nrm_sq_at_small_vals = nrm_sq + 1.0*nrm_leq_alpha
    prox_res = mshift + ((x-mshift)/(1+tau/alpha))*nrm_leq_alpha + (x-mshift)*(1-tau/(np.sqrt(nrm_sq_at_small_vals)))*(1-nrm_leq_alpha)
    if verbose:
        print('both included?')
        print(nrm_leq_alpha.mean())
    return prox_res

def l1nrm(x,mshift=0.0,dims=None,vdims=(),eps=0):

    if vdims or eps>0:
        return np.sqrt(np.square(x-mshift).sum(axis=vdims) + eps ).sum(axis=dims,keepdims=True)
    else:
        return np.abs(x-mshift).sum(axis=dims,keepdims=True)

#Rotate vector valued signal
def rot90(c):

    x = np.zeros(c.shape)
    x[...,0] = c[...,1]
    x[...,1] = -c[...,0]
    
    return x


#Generate a Matrix for the linear interpolation of a vector of size n at points given by r
def lininterM(rr,n):
    A = np.zeros([rr.shape[0],n])
    
    for ii in range(rr.shape[0]):
        r = rr[ii]
        
        pl = np.floor(r*(n-1)).astype('int')
        ld = 1 - (r*(n-1) - pl)
        if pl<n:
            A[ii,pl] = ld
            if pl < n-1:
                A[ii,pl+1] = (1-ld)

    return A



#Reshape array to matrix using the last rowdims dimensions for the rows
def mshape(A,rowdims=2):

    s = np.array(A.shape)
    return np.reshape(A,[ np.prod(s[:-rowdims]) , np.prod(s[-rowdims:]) ])




sz = 5
sig = 1
mu = 0
l1 = np.linspace(-1,1,sz)
l2 = np.linspace(-1,1,sz)
a1,a2 = np.meshgrid(l1,l2)
gauss_kernel = ( 1.0/np.sqrt(2.0*np.pi*sig*sig) )*np.exp( -(np.square(a1-mu) + np.square(a2-mu))/(2.0*sig*sig))
# different semynorm-type functionals for which we implement evaluation, proximal mapping, and subgradient explicitly
class nfun(object):

    def __init__(self,ntype,npar=1.0,mshift=0.0,dims=None,vdims=(), mask=False,eps=0.1,delta=1.0,l1eps_par=0.1, 
                        blur_kernel = gauss_kernel, huber_alpha = 1.0):
        
        self.npar = npar #Scalar mulitplier
        self.ntype = ntype #Type of norm
        self.mshift = np.copy(mshift) #Minus-shift: We consider N(x-mshift)
        self.vdims = vdims #Variable that fixes some dimensions for particular norms
        self.mask = mask #Mask for inpainting-norm
        self.eps = eps #Parameter for semi-convex function
        self.delta = delta #Second parameter for semi-convex function
        self.l1eps_par = l1eps_par # Smoothing parameter for l1+eps norm
        self.dims=dims # axis over which the norm is computed.
        self.blur_kernel=blur_kernel
        self.huber_alpha = huber_alpha

        check = True
        for size in self.blur_kernel.shape[:-1]:
            check = check and (size%2==1)

        assert check, 'Blur kernel has to have odd dimensions'
        
        
        #List of implemented types
        ntypes = ['l1','l2sq','2d_l2blur','huber']
        
        if ntype not in ntypes:
            raise NameError('Unknown ntype: ' + ntype)
        
        #List of types that implement vdims
        vdims_types = ['l1']
        if vdims and ntype not in vdims_types:
            print('Warning: vdims not implemented for ' + ntype)


        #Set evaluation
        if ntype == 'l1':
            def val(x): return self.npar*l1nrm(x,mshift=self.mshift, dims = self.dims, vdims=self.vdims)
            def prox(x,ppar): return shrink(x,tau=self.npar*ppar,mshift=self.mshift, vdims=self.vdims)
            def subgrad(x):
                if vdims:
                    pointwise_2norm = np.sqrt(np.sum(np.power(x-self.mshift,2), axis=self.vdims, keepdims=True))
                    divider = pointwise_2norm + 1.0*(pointwise_2norm<1e-10)
                    subgrad = self.npar*(x-self.mshift)/divider
                else:
                    subgrad = self.npar*np.sign(x-self.mshift)

                return subgrad

        elif ntype == 'l2sq':
            def val(x): return 0.5*self.npar*l2nsq(x,mshift=self.mshift, dims = self.dims) 
            def prox(x,ppar): return proxl2f(x,f=self.mshift,tau=self.npar*ppar)
            def subgrad(x): return self.npar*(x-self.mshift)

        elif ntype == '2d_l2blur':
            
            kernel = np.zeros(self.mshift.shape)
            if len(kernel.shape)==2:
                kernel = kernel[...,None]

            kernel[0:self.blur_kernel.shape[0],0:self.blur_kernel.shape[1],:] = np.copy(self.blur_kernel[...,None])

            kernel = np.roll(kernel, (-(self.blur_kernel.shape[0]//2),-(self.blur_kernel.shape[1]//2)), axis=(0,1))
            Fkernel = scipy.fft.fft2(kernel,norm='ortho',axes=(0,1))

            self.FK = np.sqrt(kernel.shape[0]*kernel.shape[1])*Fkernel
            self.Fshift = scipy.fft.fft2(self.mshift,norm='ortho',axes=(0,1))

            def val(x):
                Fx = scipy.fft.fft2(x,norm='ortho',axes=(0,1))
                return 0.5*self.npar*l2nsq( self.FK*Fx ,mshift=self.Fshift, dims = self.dims)


            def prox(x,ppar):

                Fx = scipy.fft.fft2(x,norm='ortho',axes=(0,1))
                Fprox = proxl2f_with_forward_operator(Fx,K=self.FK,f=self.Fshift,tau=ppar*self.npar)
                res = np.real(scipy.fft.ifft2(Fprox,norm='ortho',axes=(0,1)))
                return res

            def subgrad(x):
                Fx = scipy.fft.fft2(x,norm='ortho',axes=(0,1))
                outer_grad = self.FK*Fx-self.Fshift
                grad = np.conjugate(self.FK)*outer_grad
                grad = np.real(scipy.fft.ifft2(grad,norm='ortho',axes=(0,1)))

                return self.npar*grad

            def test_prox():
                x = 10*np.random.normal(size = self.mshift.shape)+np.pi
                p = prox(x,np.pi)
                print(np.sum(np.square(np.abs(np.pi*subgrad(p)+p-x))))
                return
            self.test_prox = test_prox

        elif ntype == 'huber':
            def val(x): return self.npar*huber_loss(x,mshift=self.mshift,alpha = self.huber_alpha, dims = self.dims)
            def prox(x,ppar): return huber_prox(x,mshift=self.mshift,tau=ppar*self.npar,alpha = self.huber_alpha, dims = self.dims)
            def subgrad(x): return self.npar*huber_grad(x,mshift=self.mshift,alpha = self.huber_alpha, dims = self.dims)

        else:
            raise Exception("Not a valid nfun type")

        #Set value, prox and dprox                
        self.val = val
        self.prox = prox
        self.subgrad = subgrad


def test_prox(fun,dim1):
    for i in range(10):
        z = 10*np.random.normal(size = dim1)
        p = fun.prox(z,np.pi)
        grad = fun.subgrad(p)
        print(np.sum(np.square(np.abs(np.pi*grad+p-z))))
        input()
    return


def test_grad(fun,dim1):
    t=1e-3
    for k in range(10):
        x = 10*np.random.normal(size = dim1)
        print(x.shape)
        for i in range(x.shape[0]):
            x_ij = np.copy(x)
            x_ij[i,:] = x_ij[i,:] + t

            print( np.sum(np.square((fun.val(x_ij)-fun.val(x))/t - fun.subgrad(x)[i])))

        input()
    return

# test fourier theorem for convolution
def test_fourier_theorem():

    sz = 5
    sig = 1
    mu = 0
    l1 = np.linspace(-1,1,sz)
    l2 = np.linspace(-1,1,sz)
    a1,a2 = np.meshgrid(l1,l2)
    gauss_kernel = ( 1.0/np.sqrt(2.0*np.pi*sig*sig) )*np.exp( -(np.square(a1-mu) + np.square(a2-mu))/(2.0*sig*sig))
    x = np.random.rand(128,128)
    #x = np.load('images/barbara.npy')

    conv = scipy.signal.convolve2d(x,gauss_kernel,mode='same',boundary='wrap')
    fourier_conv = scipy.fft.fft2(conv,norm='ortho',axes=(0,1))

    kernel = np.zeros(x.shape)
    kernel[0:sz,0:sz] = gauss_kernel
    gauss_kernel = kernel
    gauss_kernel = np.roll(gauss_kernel, (-(sz//2),-(sz//2)), axis=(0,1))

    fourier_kernel = scipy.fft.fft2(gauss_kernel,norm='ortho',axes=(0,1))
    fourier_im = scipy.fft.fft2(x,norm='ortho',axes=(0,1))

    conv_fourier = np.sqrt(np.prod(x.shape))*fourier_kernel*fourier_im
    inv_fourier_conv = scipy.fft.ifft2(conv_fourier,norm='ortho',axes=(0,1))

    print('Abs. error:' + str(np.sum(np.square(np.abs(conv_fourier-fourier_conv)))))
    print('Rel. error:' + str(np.sum(np.square(np.abs(conv_fourier-fourier_conv)))/np.sum(np.square(np.abs(fourier_conv)))))

    print('Abs. error:' + str(np.sum(np.square(np.abs(conv-inv_fourier_conv)))))
    print('Rel. error:' + str(np.sum(np.square(np.abs(conv-inv_fourier_conv)))/np.sum(np.square(conv))))

    return

def test_fourier_norm():

    sz = 5
    sig = 1
    mu = 0
    l1 = np.linspace(-1,1,sz)
    l2 = np.linspace(-1,1,sz)
    a1,a2 = np.meshgrid(l1,l2)
    gauss_kernel = np.random.rand(sz,sz,10)
    x = np.random.rand(128,128,10)
    mshift = np.random.rand(128,128,10)



    kernel = np.zeros(x.shape)
    kernel[0:sz,0:sz,...] = np.copy(gauss_kernel)
    kernel = np.roll(kernel, (-(sz//2),-(sz//2)), axis=(0,1))
    Fkernel = scipy.fft.fft2(kernel,norm='ortho',axes=(0,1))
    FK = np.sqrt(x.shape[0]*x.shape[1])*Fkernel
    Fshift = scipy.fft.fft2(mshift,norm='ortho',axes=(0,1))

    res1 = np.zeros(10)
    for i in range(10):
        blurred_image = scipy.signal.convolve2d(x[:,:,i],gauss_kernel[:,:,i],mode='same',boundary='wrap')
        res1[i] = 0.5*l2nsq( blurred_image ,mshift=mshift[:,:,i], dims = (0,1))

    res2 = np.zeros(10)
    Fx = scipy.fft.fft2(x,norm='ortho',axes=(0,1))
    res2 = 0.5*l2nsq( FK*Fx ,mshift=Fshift, dims = (0,1))

    print(np.max(np.abs(res1-res2)))

    return



### Linear Operators ####################################################################
#########################################################################################
def test_adj(fwd,adj,dim1,*dim2):

        if not dim2:
            dim2 = dim1
        else:
            dim2 = dim2[0]

        x = np.random.rand(*dim1)
        y = np.random.rand(*dim2)

        s1 = (fwd(x)*y).sum()
        s2 = (x*adj(y)).sum()
        
        print('Abs err: ' + str(np.abs(s1-s2)))
        print('Rel err: ' + str( np.abs(s1-s2)/np.abs(x).sum() ))

# identity operator
class identity(object):

    def __init__(self,shape):
        self.nrm = 1.0
        self.indim = list(shape)
        self.outdim = list(shape)
        
    def fwd(self,x):

        return x

    def adj(self,x):

        return x

#1-dimensional gradient along the first axis
class gradient_1d(object):

    def __init__(self,shape):
    
        self.indim = list(shape)
        self.outdim = self.indim
        
        self.nrm = 2.0
        
    def fwd(self,x):

        z = np.zeros(self.outdim)
        
        z[:-1,...] = x[1:,...] - x[:-1,...]

        return z


    def adj(self,p):


        x = np.zeros(self.indim)
        
        x[0,...]    = p[0,...]
        x[-1,...]   =             - p[-2,...]
        x[1:-1,...] = p[1:-1,...] - p[:-2,...]
        

        return -x


    def test_adj(self):

        test_adj(self.fwd,self.adj,self.indim,self.outdim)



class gradient(object):

    def __init__(self,shape):
    
        self.indim = list(shape)
        
        outdim = list(shape[0:2])
        outdim.append(2)
        
        for dim in shape[2:]:
            outdim.append(dim)
            
        self.outdim = outdim
        
        self.nrm = np.sqrt(8.0)
        
        self.oS = 2.0
        self.oT = 4.0
        
    def fwd(self,x):

        z = np.zeros(self.outdim)
        
        z[:-1,:,0,...] = x[1:,:,...] - x[:-1,:,...]
        z[:,:-1,1,...] = x[:,1:,...] - x[:,:-1,...]

        return z


    def adj(self,p):


        x = np.zeros(self.indim)
        
        x[0,:,...]    = p[0,:,0,...]
        x[-1,:,...]   =             - p[-2,:,0]
        x[1:-1,:,...] = p[1:-1,:,0,...] - p[:-2,:,0,...]
        

        x[:,0,...]    += p[:,0,1,...]
        x[:,-1,...]   +=             - p[:,-2,1,...]
        x[:,1:-1,...] += p[:,1:-1,1,...] - p[:,:-2,1,...]

        return -x


    def test_adj(self):

        test_adj(self.fwd,self.adj,self.indim,self.outdim)


