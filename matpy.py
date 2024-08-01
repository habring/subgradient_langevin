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
try:
    import pickle5 as pickle
except:
    import pickle
import imageio
from scipy import linalg

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
            else:
                raise NameError('No filename given')

        #Generate folder if necessary
        if folder:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # The following was commented by Andi, in order to be able to save with names containing '.'
        # Remove ending if necessary
        pos = fname.find('.')
        if pos>-1:
            fname = fname[:pos]

        # if fname[-4:] == '.png':
        #     fname = fname[:-4]
        
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


def plot(x,y=0,fig=0,title=0,label=0):

        

    try:
        if not fig:
            fig = plt.figure()
        plt.figure(fig.number)
        
        if not np.any(y):
            plt.plot(x,label=label)
        else:
            plt.plot(x,y,label=label)
            
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
    
    #Convert range to [0,1]
    img = img - rg[0]
    if (rg[1]-rg[0])>0:
        img = img/(rg[1]-rg[0])
    elif np.any(img):
        #raise ValueError('Function requires rg[0]<rg[1]')        
        print('image is constant')
    else:
        print('Warning: empty image, ignoring range argument, no normalization carried out')



    return img
    
def imsave(fname,img,format=None,rg=[], normalize=False): #rg defines grayscale boundary values. Choosing rg=[] uses [img.min(),img.max()]
    
    
    img = np.clip(np.copy(img.astype('float')),0,1)
    if normalize:
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
def proxl1s(p,mu=1.0,vdims=(),copy=True,tau=1.0,vecweights=False):


    if copy:
        z = np.copy(p)
    else:
        z = p

    if not vecweights:
        if np.sum(mu) not in [0,np.inf]:
            if vdims:
                z/= np.maximum( np.sqrt(np.square(p).sum(axis=vdims,keepdims=True) )/mu , 1.0)
            else:
                z /= np.maximum( np.abs(p)/mu , 1.0)
        elif mu == 0:
            z = np.zeros(z.shape)
        
        #If mu=np.inf nothing needs to be done
    else:
        
        wdim = [1,1,len(vecweights)] + [1 for i in range(len(p.shape[3:]))]
        weight = np.ones(wdim)
        for i in range(len(vecweights)):
            weight[0,0,i,...] = vecweights[i]
        
        z/= np.maximum( np.sqrt((weight*np.square(p)).sum(axis=vdims,keepdims=True) )/mu , 1.0)

    return z

#Proximal mapping of tau|.|_1, i.e., computes (I + tau|.|_1)^(-1)
def shrink(p,tau=1.0,mshift=0.0,vdims=(),copy=True,vecweights=False):

    #Note that we skip -mshift + mshift
    #Using prox_f(x) = prox_0(x-f)+f
    return p - proxl1s((p-mshift),tau,vdims=vdims,copy=copy,vecweights=vecweights) #Moreau's Identity


#Computes (I + tau*DF)^(1-) with F(u) = 1/2|u-f|_2^2
#Does not modify the input
def proxl2f(u,f=0,tau=1.0):

    return (u+tau*f) / (1.0+tau)

#Same as above, but dual
def proxl2fs(u,mshift=0,npar=1.0,ppar=1.0):

    return (u-ppar*mshift) / (1.0+(ppar/npar))


def l2nsq(x,mshift=0.0):
    return np.square(np.abs(x-mshift)).sum()

def l1nrm(x,mshift=0.0,vdims=(),eps=0,vecweights=False):

    if not vecweights:
        if vdims or eps>0:
            return np.sqrt(np.square(x-mshift).sum(axis=vdims) + eps ).sum()
        else:
            return np.abs(x-mshift).sum()
    else:
        wdim = [1,1,len(vecweights)] + [1 for i in range(len(x.shape[3:]))]
        weight = np.ones(wdim)
        for i in range(len(vecweights)):
            weight[0,0,i,...] = vecweights[i]
        
        return np.sqrt((weight*np.square(x-mshift)).sum(axis=vdims) + eps ).sum()


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

#Semi-norm type function
class nfun(object):

    def __init__(self,ntype,npar=1.0,mshift=0.0,vdims=(),vecweights = False , mask=False,eps=0.1,delta=1.0,l1eps_par=0.1):
        
        self.npar = npar #Scalar mulitplier
        self.ntype = ntype #Type of norm
        self.mshift = mshift #Minus-shift: We consider N(x-mshift)
        self.vdims = vdims #Variable that fixes some dimensions for particular norms
        self.mask = mask #Mask for inpainting-norm
        self.eps = eps #Parameter for semi-convex function
        self.delta = delta #Second parameter for semi-convex function
        self.vecweights = vecweights #Allows to put weights on the vector-dimension for the l1 norm, used for the symgrad and symgrad2 norm
        self.l1eps_par = l1eps_par # Smoothing parameter for l1+eps norm

        
        #List of implemented types
        ntypes = ['l1','linfty','l2sq','l1infty','semiconv','semiconv-svd',
        'inpaint','l1_2','l1_2-svd','l0','l0-svd','l1-svd','I_linfty','I0','IN','IP','zero','usprox','l1eps']
        
        if ntype not in ntypes:
            raise NameError('Unknown ntype: ' + ntype)
        
        #List of types that implement mshift
        mshift_types = ['l1','inpaint','l2sq','l1eps','I0','zero']
        if np.any(mshift) and ntype not in mshift_types:
            print('Warning: mshift not implemented for ' + ntype)
        #List of types that implement vdims
        vdims_types = ['l1','l1infty','I_linfty','l1-svd','semiconv-svd','l1_2-svd','l0-svd','l1eps']
        if vdims and ntype not in vdims_types:
            print('Warning: vdims not implemented for ' + ntype)
        #List of types that implement vecweights
        vecweights_types = ['l1']
        if vecweights and ntype not in vecweights_types:
            print('Warning: vecweights not implemented for ' + ntype)



        #Set evaluation    
        if ntype == 'l1':
        
            def val(x): return self.npar*l1nrm(x,mshift=self.mshift,vdims=self.vdims,vecweights=self.vecweights)
            def dprox(x,ppar): return proxl1s(x,mu=self.npar,vdims=self.vdims,vecweights=self.vecweights)
            def prox(x,ppar): return shrink(x,tau=self.npar*ppar,mshift=self.mshift,vdims=self.vdims,vecweights=self.vecweights)
            
        if ntype == 'linfty':
        
            def val(x): return self.npar*np.max(np.abs(x))
            def dprox(x,ppar=1.0): return project_l1(x,tau=npar,nx=len(x.shape))
            def prox(x,ppar): return  prox_dual(x,self.dprox,ppar=ppar)

        #Indicator function of Linfty-ball (dual of L1)        
        if ntype == 'I_linfty':
        
            def val(x): 
                
                z = np.sqrt( (x*x).sum(axis=vdims,keepdims=True))
                
                return z[z>self.npar].sum()
                    
            def prox(x,ppar): return proxl1s(x,mu=self.npar,vdims=self.vdims)            
            def dprox(x,ppar): return prox_dual(x,prox,ppar=ppar)

        if ntype == 'l2sq':
            
            def val(x): return 0.5*self.npar*l2nsq(x,mshift=self.mshift) 
            def dprox(x,ppar): return proxl2fs(x,mshift=self.mshift,npar=self.npar,ppar=ppar)
            def prox(x,ppar): return proxl2f(x,f=self.mshift,tau=self.npar*ppar)
            
            def grad(x): return self.npar*(x-self.mshift)
        

        if ntype == 'l1eps':
            
            def val(x): return self.npar*((np.sqrt(np.square(x-self.mshift).sum(axis=vdims) + self.l1eps_par )).sum())
            def grad(x): return self.npar*(x-self.mshift)/np.sqrt(np.square(x-self.mshift).sum(axis=vdims,keepdims=True) + self.l1eps_par ) 
            
            def prox(x,ppar): raise NameError('Error: Prox not implemented for l1eps')
            def dprox(x,ppar): raise NameError('Error: Dual Prox not implemented for l1eps')
            


        if ntype == 'l1infty':

            if vdims:    
                print('l1infty-norm: Warning: taking the len(vdims) last dimensions as vdims!!')
            
            def val(x): return self.npar*np.max(np.abs(x),axis=vdims).sum()
            def dprox(x,ppar=1.0): return project_l1(x,tau=npar,nx=len(vdims))
            def prox(x,ppar): return  prox_dual(x,self.dprox,ppar=ppar)


        if ntype == 'semiconv':


            def val(x): 
                
                idx = np.abs(x)<1.0/(2.0*self.eps)
                
                return self.npar*( np.sum( np.abs(x[idx]) - self.eps*self.delta*x[idx]*x[idx]) + 
                np.sum( (1-self.delta)*np.abs(x[~idx])) + self.delta/(4.0*self.eps) )
                
            def prox(x,ppar):
            
                tau = self.npar*ppar

                if tau> 1.0/(2.0*self.eps*self.delta):
                    print('Warning, non-admissible choice of tau that violates semi-convexity!')


                
                z = np.zeros(x.shape)
                
                lb = (1.0/(2.0*self.eps))+tau*(1-self.delta)
                
                idx = (tau<x) & (x<lb)
                z[idx] = (x[idx] - tau)/(1-2.0*self.eps*self.delta*tau)
                idx =  (-lb<x) & (x<-tau)
                z[idx] = (x[idx] + tau)/(1-2.0*self.eps*self.delta*tau)     
                
                z[x>lb] = x[x>lb] - tau*(1.0-self.delta)
                z[x<-lb] = x[x<-lb] + tau*(1.0-self.delta)
                
                return z
                
            def dprox(x,ppar): return prox_dual(x,self.prox,ppar=ppar)
            
            
        if ntype == 'l1_2':
                    
            #Value  
            def val(x): return self.npar*np.sqrt(np.abs(x)).sum()
            
            #Prox
            c_thresh = np.power(54.0,1/3.0)*0.25*np.power(self.npar,2.0/3.0)
            
            def phi(x,ppar): return np.arccos(0.125*ppar*self.npar*( np.power(np.abs(x)/3.0,-1.5)))
            
            
            def f(x,ppar): return (2.0/3.0)*x*(1 + np.cos( (2*np.pi/3.0) - (2.0/3.0)*phi(x,ppar) ) )
            
            
            def prox(x,ppar):
                
                idx = np.abs(x) > c_thresh*np.power(ppar,2.0/3.0)
                
                x[idx] = f(x[idx],ppar)
                x[~idx] = 0.0
                
                return x
                
            def dprox(x,ppar): return  prox_dual(x,self.prox,ppar=ppar) 
            
        
        if ntype == 'l0':
            
            #Value  
            def val(x): return self.npar*(x>1e-14).sum()
            
            def prox(x,ppar):
                
                idx = np.abs(x) > np.power(self.npar*ppar,0.5)
                
                x[idx] = x[idx]
                x[~idx] = 0.0
                
                return x
                
            def dprox(x,ppar): return  prox_dual(x,self.prox,ppar=ppar)    

                
        #Any norm-type function on the singular values        
        if ntype.split('-')[-1] == 'svd': #If string indicates svd
            
            #Get functions on singular values
            self.vecfun = nfun(ntype.split('-')[0],npar=self.npar,eps=self.eps,delta=self.delta)
            
            self.rowdims = 2
            if vdims:    
                print('svd-norm: Warning: taking the len(vdims) last dimensions for rows!!')
                self.rowdims = len(vdims)
                
            def val(x): return self.vecfun.val( scipy.linalg.svd(mshape(x,rowdims=self.rowdims),compute_uv=0) )
            
            def prox(x,ppar):
            
                U,s,V = scipy.linalg.svd(mshape(x,rowdims=self.rowdims),full_matrices=0)
                s = self.vecfun.prox(s,ppar)
                S = np.diag(s)
            
                return np.reshape(np.dot(U, np.dot(S, V)),x.shape)
              
            def dprox(x,ppar): return  prox_dual(x,self.prox,ppar=ppar)  

        #Data fidelity for inpainting    
        if ntype == 'inpaint':
        
            if not np.any(mask):
                print('Error: mask required')
                
            def val(x): return np.abs(x[mask==1]-mshift[mask==1]).sum()
            def prox(x,ppar): x[mask==1] = mshift[mask==1] ; return x
            def dprox(x,ppar): return prox_dual(x,self.prox,ppar=ppar)
            def grad(x): return 0.0 #auxiliary function for the implementation


        #Indicator function of {0}
        if ntype == 'I0':
                                    
            def val(x): return np.abs(x-self.mshift).sum()
            def dprox(x,ppar): return x - ppar*self.mshift
            def prox(x,ppar): return np.zeros(x.shape) + self.mshift

        #Indicatur function of negative values
        if ntype == 'IN':
        
            def val(x): return np.abs(x[x>0.0]).sum()
            def dprox(x,ppar): z = np.copy(x); z[z<0] = 0; return z
            def prox(x,ppar): z = np.copy(x); z[z>0] = 0; return z

        #Indicator function of positive values
        if ntype == 'IP':

            if np.any(mshift):
                print('Warning: mshift not implemented for this prox')
                            
            def val(x): return np.abs(x[x<0.0]).sum()
            def dprox(x,ppar): z = np.copy(x); z[z>0] = 0; return z
            def prox(x,ppar): z = np.copy(x); z[z<0] = 0; return z
        
        #Zero function
        if ntype == 'zero':
                            
            def val(x): return 0.0
            def dprox(x,ppar): return np.zeros(x.shape)
            def prox(x,ppar): return np.copy(x)


        #Test of heuristic l1-norm on US where A = USV'
        if ntype == 'usprox':
            
            self.rowdims = 2
            if vdims:    
                print('svd-norm: Warning: taking the len(vdims) last dimensions for rows!!')
                self.rowdims = len(vdims)
                
            def val(x):
                
                U,s,V = scipy.linalg.svd(mshape(x,rowdims=self.rowdims),full_matrices=0)
                
                US = np.matmul(U,np.diag(s))
                
                return self.npar*np.abs(US).sum()
            
            def prox(x,ppar):
            
                U,s,V = scipy.linalg.svd(mshape(x,rowdims=self.rowdims),full_matrices=0)
                
                US = np.matmul(U,np.diag(s))
                
                US = shrink(US,tau=ppar*self.npar)
                
                return np.reshape(np.matmul(US,V),x.shape)
              
            def dprox(x,ppar):
            
                print('Warning, dual prox of usprox not implemented!')
                return  0  
        
        
        
        #Set value, prox and dprox                
        self.val = val
        self.dprox = dprox
        self.prox = prox
        if 'grad' in locals():
            self.grad = grad


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




# #Rotate 2D image 
# class rotate2d(object):

#     def fwd(self,x,ang):
    
#         return intp.rotate(x,ang*(360/(2*np.pi)),reshape=False,order=1,prefilter=False)


#     def adj(self,x,ang):
    
#         return intp.rotate(x,-ang*(360/(2*np.pi)),reshape=False,order=1,prefilter=False)


#     def test_adj(self,f1=0,f2=0,ang=0,ksz=15):
    
#         if not ang:
#             ang = 0.5*np.pi
    
#         if not f1:
        
#             l = np.linspace(-1.0,1.0,ksz)
#             a1,a2 = np.meshgrid(l,l)
#             r = np.sqrt( np.square(a1) + np.square(a2) )
#             idx = r>1.0
            
#             t1 = np.random.rand(ksz,ksz)
#             t1[idx] = 0
#             f1 = t1
            
#             t2 = np.random.rand(ksz,ksz)
#             t2[idx] = 0
#             f2 = t2
            
    
#         s1 = (self.fwd(f1,ang)*f2).sum()
#         s2 = (f1*self.adj(f2,ang)).sum()
        
        
#         print('Abs err: ' + str(s1-s2))
#         print('Rel err: ' + str( np.abs(s1-s2)/np.abs(f1).sum() ))
        
        

# #Generates a matrix that evaluates a given vector at points defined by rr, using interpolation
# class evalmtx(object):


#     def __init__(self,rr):
        
        
#         n = rr.shape[0]
#         m = (n+1)/2
        
#         A = np.zeros([n*n,m])
    
#         for ii in range(n):
#             for jj in range(n):
            
#                 r = rr[ii,jj]
#                 pl = np.floor(r*(m-1)).astype('int')
                
#                 ld = 1 - (r*(m-1) - pl)
#                 if pl<m:
#                     A[ii + n*jj,pl] = ld
#                     if pl < m-1:
#                         A[ii + n*jj,pl+1] = (1-ld)
    
#         self.A = A
#         self.n = n
#         self.m = m
    
        
#     def fwd(self,b):
        
#         n = self.n
#         nf = b.shape[-1]
#         z1 = np.zeros([n,n,nf])
#         z2 = np.zeros([n,n,nf])
        
#         for ii in range(nf):
#             z1[...,ii] = np.reshape(self.A.dot(b[:,0,ii]),[n,n])
#             z2[...,ii] = np.reshape(self.A.dot(b[:,1,ii]),[n,n])


#         return (z1,z2)
        
#     def adj(self,z1,z2):
    
#         nf = z1.shape[-1]
#         b = np.zeros([self.m,2,nf])
        
#         for ii in range(nf):
#             b[:,0,ii] = self.A.T.dot(np.reshape(z1[...,ii],self.n*self.n))
#             b[:,1,ii] = self.A.T.dot(np.reshape(z2[...,ii],self.n*self.n))
        
#         return b
        

#     def test_adj(self):
    
#         nf = 7
#         n = self.n
#         m = self.m
    
#         dim1 = [m,2,nf]
#         dim2 = [n,n,nf]
        
#         b = np.random.rand(*dim1)
#         z1 = np.random.rand(*dim2)
#         z2 = np.random.rand(*dim2)

#         r1,r2 = self.fwd(b)
        

#         s1 = (r1*z1 + r2*z2).sum()
#         s2 = (b*self.adj(z1,z2)).sum()
        
#         print('Abs err: ' + str(s1-s2))
#         print('Rel err: ' + str( np.abs(s1-s2)/np.abs(b).sum() ))



# #Basic 2D convolution with variable kernel
# #Important: Kernel dimensions must be odd!!
# #The functions fwd and adj do not modify the input
# class conv(object):

#     def fwd(self,c,k):

#         return scipy.signal.convolve2d(c,k,mode='same',boundary='wrap')

#     def adj(self,c,k):

#         return scipy.signal.convolve2d(c,np.flip(np.flip(k,0),1),mode='same',boundary='wrap')


#     def adj_ker(self,c,ksz,x):

#         psz = (ksz-1)/2
        
#         return scipy.signal.convolve2d(np.pad(x,[(psz,psz),(psz,psz)],'wrap'),np.flip(np.flip(c,0),1),mode='valid',boundary='wrap')

#     def test_adj(self):

#         k = np.random.rand(9,11)

#         fwd = lambda x: self.fwd(x,k)
#         adj = lambda x: self.adj(x,k)

#         test_adj(fwd,adj,[51,62])


#         c = np.random.rand(92,111)
#         ksz = 13

#         fwd = lambda x: self.fwd(c,x)
#         adj = lambda x: self.adj_ker(c,ksz,x)

#         test_adj(fwd,adj,[ksz,ksz],c.shape)


#Identity Operator
class id(object):

    def __init__(self,indim):
    
    
        if np.any(indim):
            self.indim = indim
            self.outdim = indim

        self.nrm = 1.0

    def fwd(self,x):
        
        return x

    def adj(self,x):

        return x

#Zero operator
class zero(object):

    def __init__(self,indim=False):
    
        if np.any(indim):
            self.indim = indim
            self.outdim = indim
            
        self.nrm = 1.0 #Is =0, but choosen 1.0 to avoid having to deal with 0 norms
        
    def fwd(self,x):
        
        return np.zeros(x.shape)

    def adj(self,x):

        return np.zeros(x.shape)

#Summation over given axis
class vecsum(object):

    def __init__(self,ndims,axis=2):
    
        self.axis=axis
        self.ndims = ndims

    def fwd(self,x):
        
        return x.sum(axis=self.axis)

    def adj(self,x):

        return np.stack([x for ii in range(self.ndims)],axis=self.axis)

    def test_adj(self):


        test_adj(self.fwd,self.adj,[51,62,self.ndims],[51,62])


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

# class symgrad(object):

#     def __init__(self,shape):
    
#         self.indim = list(shape)
        
#         outdim = list(shape[0:2])
#         outdim.append(3)
        
#         for dim in shape[3:]:
#             outdim.append(dim)
            
#         self.outdim = outdim
        
#         self.nrm = np.sqrt(8.0)
        
#     def fwd(self,x):

#         z = np.zeros(self.outdim)
        
        
        
#         z[1:,:,0,...] = x[1:,:,0,...] - x[:-1,:,0,...]
#         z[:,1:,1,...] = x[:,1:,1,...] - x[:,:-1,1,...]
        
#         z[1:,:,2,...] = x[1:,:,1,...] - x[:-1,:,1,...]
#         z[:,1:,2,...] += x[:,1:,0,...] - x[:,:-1,0,...]
#         z[:,:,2,...] *= 0.5

#         return z


#     def adj(self,p):


#         x = np.zeros(self.indim)
        
        
#         x[0,:,0,...]    = p[1,:,0,...]
#         x[-1,:,0,...]   =               - p[-1,:,0,...]
#         x[1:-1,:,0,...] = p[2:,:,0,...] - p[1:-1,:,0,...]
        
#         x[:,0,0,...]    += p[:,1,2,...]
#         x[:,-1,0,...]   +=               - p[:,-1,2,...]
#         x[:,1:-1,0,...] += p[:,2:,2,...] - p[:,1:-1,2,...]


#         x[0,:,1,...]    = p[1,:,2,...]
#         x[-1,:,1,...]   =               - p[-1,:,2,...]
#         x[1:-1,:,1,...] = p[2:,:,2,...] - p[1:-1,:,2,...]
        
#         x[:,0,1,...]    += p[:,1,1,...]
#         x[:,-1,1,...]   +=               - p[:,-1,1,...]
#         x[:,1:-1,1,...] += p[:,2:,1,...] - p[:,1:-1,1,...]


#         return -x


#     def test_adj(self):
    
    

#         x = np.random.rand(*self.indim)
#         y = np.random.rand(*self.outdim)

#         z = self.fwd(x)
#         z[:,:,2] *=2.0

#         s1 = (z*y).sum()
#         s2 = (x*self.adj(y)).sum()
        
#         print('Abs err: ' + str(s1-s2))
#         print('Rel err: ' + str( np.abs(s1-s2)/np.abs(x).sum() ))


#Computes the symmetrized gradient of a [n,m,3] symmetric matrix using forward differences
#Build to be used for TGV3
#Expected indim: [n,m,3,...]
#Resulting outdim: [n,m,4,...]

# class symgrad2(object):

#     def __init__(self,shape):
    
#         self.indim = list(shape)
        
#         outdim = list(shape[0:2])
#         outdim.append(4)
        
#         for dim in shape[3:]:
#             outdim.append(dim)
            
#         self.outdim = outdim
        
#         self.nrm = np.sqrt(8.0)
    
#     #Output dx(x0) dy(x1) 1/3[ dy(x0) + 2*dx(x2) ] 1/3[ 2*dy(x2) + dx(x1) ]        
#     def fwd(self,x):

#         z = np.zeros(self.outdim)
        
        
#         #dx of x[0]
#         z[1:,:,0,...] = x[1:,:,0,...] - x[:-1,:,0,...]
#         #dy of x[1]        
#         z[:,1:,1,...] = x[:,1:,1,...] - x[:,:-1,1,...]
        
#         #2*dx of x[2]
#         z[1:,:,2,...] = 2*( x[1:,:,2,...] - x[:-1,:,2,...] )
#         # + dy of x[0]
#         z[:,1:,2,...] += x[:,1:,0,...] - x[:,:-1,0,...]
        
#         z[:,:,2,...] /=3.0
        
#         #dx of x[1]
#         z[1:,:,3,...] = x[1:,:,1,...] - x[:-1,:,1,...]
#         # + 2*dy of x[2]
#         z[:,1:,3,...] += 2*( x[:,1:,2,...] - x[:,:-1,2,...] )
        
#         z[:,:,3,...] /=3.0

#         return z


#     def adj(self,p):


#         x = np.zeros(self.indim)
        
#         #dx of p[0] + dy of p[2]
#         x[0,:,0,...]    = p[1,:,0,...]
#         x[-1,:,0,...]   =               - p[-1,:,0,...]
#         x[1:-1,:,0,...] = p[2:,:,0,...] - p[1:-1,:,0,...]
        
#         x[:,0,0,...]    += p[:,1,2,...]
#         x[:,-1,0,...]   +=               - p[:,-1,2,...]
#         x[:,1:-1,0,...] += p[:,2:,2,...] - p[:,1:-1,2,...]

#         #dy of p[1] + dx of p[3]
#         x[:,0,1,...]    = p[:,1,1,...]
#         x[:,-1,1,...]   =               - p[:,-1,1,...]
#         x[:,1:-1,1,...] = p[:,2:,1,...] - p[:,1:-1,1,...]

#         x[0,:,1,...]    += p[1,:,3,...]
#         x[-1,:,1,...]   +=               - p[-1,:,3,...]
#         x[1:-1,:,1,...] += p[2:,:,3,...] - p[1:-1,:,3,...]
        
#         #dx of p[2] + dy of p[3]
#         x[0,:,2,...]    = p[1,:,2,...]
#         x[-1,:,2,...]   =               - p[-1,:,2,...]
#         x[1:-1,:,2,...] = p[2:,:,2,...] - p[1:-1,:,2,...]
        
#         x[:,0,2,...]    += p[:,1,3,...]
#         x[:,-1,2,...]   +=               - p[:,-1,3,...]
#         x[:,1:-1,2,...] += p[:,2:,3,...] - p[:,1:-1,3,...]


#         return -x


#     def test_adj(self):
    
    

#         x = np.random.rand(*self.indim)
#         y = np.random.rand(*self.outdim)

#         z1 = self.fwd(x)
#         z1[:,:,2,...] *=3.0
#         z1[:,:,3,...] *=3.0

#         s1 = (z1*y).sum()
        
#         z2 = self.adj(y)
#         z2[:,:,2,...] *=2.0
        
#         s2 = (x*z2).sum()
        
#         print('Abs err: ' + str(s1-s2))
#         print('Rel err: ' + str( np.abs(s1-s2)/np.abs(x).sum() ))



# class hessian(object):

#     def __init__(self,shape):
    
#         self.indim = list(shape)

#         self.grad = gradient(self.indim)
#         self.symgrad = symgrad(self.grad.outdim)
        
#         self.outdim = self.symgrad.outdim
            
        
#     def fwd(self,x):

#         return self.symgrad.fwd(self.grad.fwd(x))


#     def adj(self,p):

#         return self.grad.adj(self.symgrad.adj(p))


#     def test_adj(self):

#         x = np.random.rand(*self.indim)
#         y = np.random.rand(*self.outdim)

#         z = self.fwd(x)
#         z[:,:,2] *=2.0

#         s1 = (z*y).sum()
#         s2 = (x*self.adj(y)).sum()
        
#         print('Abs err: ' + str(s1-s2))
#         print('Rel err: ' + str( np.abs(s1-s2)/np.abs(x).sum() ))




# class divergence(object):

#     def __init__(self,shape):
    
    
#         self.indim = shape
        
#         if not shape[2]==2:
#             print('Warning: Unexpected size of dimension 3')
            
#         self.outdim = shape[0:2] + shape[3:]
        
#         self.grad = gradient(self.outdim)

#     def fwd(self,x):
    
#         return self.grad.adj(x)


#     def adj(self,p):

#         return self.grad.fwd(p)

#     def test_adj(self):
    
#         test_adj(self.fwd,self.adj,self.indim,self.outdim)


# #Decomposition of gradient and adjoint
# #Warning: Boundary conditions are inappropriate
# class laplace(object):

#     def __init__(self,shape):
    
    
#         self.indim = shape
                    
#         self.outdim = shape
        
#         self.grad = gradient(self.indim)


#     def fwd(self,x):
    
#         return -self.grad.adj(self.grad.fwd(x))


#     def adj(self,x):

#         return -self.grad.adj(self.grad.fwd(x))

#     def test_adj(self):
    
#         test_adj(self.fwd,self.adj,self.indim,self.outdim)



### Algorithms ##########################################################################
#########################################################################################

#TV-regularized denoising or inpainting (or any other prox-explicit data fidelity)
def tv_denoise(**par_in):


    #Initialize parameters and data input
    par = parameter({})
    data_in = data_input({})
    data_in.mask = 0 #Inpaint requires a mask

    ##Set data
    data_in.u0 = 0 #Direct image input



    #Version information:
    par.version='Version 0'


    par.imname = 'barbara_crop.png'

    par.niter = 10


    par.ld = 10.0 #regpar

    par.noise = 0

    #Data type: {'l1','l2sq','inpaint','I0'}
    par.dtype='l2sq' 

    par.s_t_ratio = 1.0
    
    par.check=-1


    ##Data and parmaeter parsing
    
    #Set parameters according to par_in
    par_parse(par_in,[par,data_in])

    #Read image
    if not np.any(data_in.u0):
        data_in.u0 = imread(par.imname)
    elif not par.imname:
        par.imname='direct_input'



    Ln = np.sqrt(8.0)
    par.sig = par.s_t_ratio/Ln
    par.tau = 1.0/(par.s_t_ratio*Ln)
    #print('Stepsizes: ' + str(par.sig)+',' + str(par.tau))


    ## Data initilaization
    u0 = np.copy(data_in.u0)
    
    #Add noise
    if par.noise:
        np.random.seed(1)
        
        rg = np.abs(u0.max() - u0.min()) #Image range
        
        u0 += np.random.normal(scale=par.noise*rg,size=u0.shape) #Add noise



    #Set variables
    size = u0.shape[0:2]

    #Operator
    if len(u0.shape)==2:
        gradient_1d 
        grad = gradient_1d(u0.shape)
    elif len(u0.shape)==3:
        grad = gradient(u0.shape)
    else:
        print('Problem')
    
    #Data fidelity
    dnrm = nfun(par.dtype,mshift=u0,npar=par.ld,mask=data_in.mask)

    #Primal
    u = np.zeros(u0.shape)
    ux = np.zeros(u0.shape)

    #Dual
    p = np.zeros(grad.outdim)

    ob_val = []#np.zeros([par.niter+1])
    
    ob_val = ob_val + [dnrm.val(u) + l1nrm(grad.fwd(u),vdims=())]

    if par.niter<0:
        maxit=np.inf
    else:
        maxit = par.niter

    stopping_criterion = False

    k=0
    while k<maxit and not stopping_criterion:
        
        
        #Dual
        p = p + par.sig*( grad.fwd(ux) )
        p = proxl1s(p,1.0,vdims=())

        #Primal
        ux = dnrm.prox( u - par.tau*(grad.adj(p))   ,ppar=par.tau)    


        stopping_criterion = (np.max(np.abs(2.0*(ux-u)))<1e-4)

        u = 2.0*ux - u

        [u,ux] = [ux,u]

  
        ob_val = ob_val + [dnrm.val(u) + l1nrm(grad.fwd(u),vdims=())]
        
        #stopping_criterion = ((np.abs(ob_val[k]-ob_val[k+1])/ob_val[k])<1e-4)

        
        if np.remainder(k,par.check) == 0 and par.check>0:
            print('Iteration: ' + str(k) + ' / Ob-val: ' + str(ob_val[k+1]))

        k += 1


    ob_val = np.array(ob_val)

    #Initialize output class
    res = output(par)

    #Set original input
    res.orig = data_in.u0
    res.u0 = u0

    res.u = u
    res.p = p
    res.ob_val = ob_val    
    
    #Save parmeters and input data
    res.par = par
    res.par_in = par_in
    res.data_in = data_in


    res.grad = grad
    
    return res



# #TGV-regularized denoising or inpainting (or any other prox-explicit data fidelity)
# def tgv_denoise(**par_in):

#     #Load image from name
#     imname = ''

#     #Direct image input
#     u0 = 0
    
#     #Data type
#     dtype='l2sq' #Data type: {'l1','l2sq','inpaint'}
#     mask = 0 #Inpaint requires a mask

#     niter = 1

#     #Regularization parameter
#     ld = 20.0
    
#     #TGV parameter
#     alpha0 = np.sqrt(2.0)
    
#     noise=0.1
    
#     s_t_ratio = 100.0

#     check = niter #Show results ever "check" iteration
    
#     #Set parameter according to par_in ----------
#     par = par_parse(locals().copy(),par_in)
#     #Update parameter
#     for key,val in par.items():
#         exec(key + '= val')
#     res = output(par) #Initialize output class
#     #--------------------------------------------

#     if not np.any(u0):
#         u0 = imread(imname)
#     elif not imname:
#         imname='direct_input'


#     #Set original input
#     res.orig = np.copy(u0)
    
#     #Add noise
#     if noise:
#         np.random.seed(1)
#         u0 = np.copy(u0)
#         rg = np.abs(u0.max() - u0.min()) #Image range
        
#         u0 += np.random.normal(scale=noise*rg,size=u0.shape) #Add noise
        
#     if np.any(mask):
    
#         u0 = np.copy(u0)
#         u0[~mask] = 0.0
        
#     res.u0 = u0
    
    
#     #Stepsize
#     nrm = get_product_norm( [ [8,8],[0,1]] )
    
#     sig = s_t_ratio/nrm
#     tau = 1/(nrm*s_t_ratio)

#     print('Stepsizes: sig: '+str(sig)+' / tau: '+str(tau))

#     #Image size
#     N,M = u0.shape

#     #Operators and norms
#     grad = gradient(u0.shape)
#     sgrad = symgrad(grad.outdim)
    

#     dnrm = nfun(dtype,mshift=u0,npar=ld,mask=mask)
#     l1vec = nfun('l1',vdims=(2),npar=1.0)
#     l1mat = nfun('l1',vdims=(2),npar=alpha0*1.0,vecweights=[1,1,2])

#     ##Variables

#     #Primal
#     u = np.zeros(u0.shape)
#     ux = np.zeros(u.shape)
    
#     v = np.zeros(grad.outdim)
#     vx = np.zeros(v.shape)
            
#     #Dual
#     p = np.zeros(grad.outdim)
#     q = np.zeros(sgrad.outdim)
    
    
#     ob_val = np.zeros([niter+1])

#     ob_val[0] = dnrm.val(u) + l1vec.val(grad.fwd(u) - v) + l1mat.val(sgrad.fwd(v))
    
#     for k in range(niter):

        
#         #Dual        
#         p = p + sig*( grad.fwd(ux) - vx )
#         p = l1vec.dprox(p,ppar=sig)
                
#         q = q + sig*( sgrad.fwd(vx) )
#         q = l1mat.dprox(q,ppar=sig)
        
#         #Primal
#         ux = dnrm.prox( u - tau*(grad.adj(p))   ,ppar=tau)
#         vx =            v - tau*( -p + sgrad.adj(q))

#         u = 2.0*ux - u
#         v = 2.0*vx - v

#         [u,ux] = [ux,u]
#         [v,vx] = [vx,v]

#         ob_val[k+1] = dnrm.val(u) + l1vec.val(grad.fwd(u) - v) + l1mat.val(sgrad.fwd(v))

#         if np.remainder(k,check) == 0:
#             print('Iteration: ' + str(k) + ' / Ob-val: ' + str(ob_val[k]) + ' / Image: '+imname)

#     #Save variables
#     res.u = u
#     res.v = v
#     res.ob_val = ob_val

    
#     #Save operators
#     res.grad = grad
#     res.sgrad = sgrad
    
#     imshow(np.concatenate((res.orig,res.u0,res.u),axis=1),title='Orig vs noisy vs rec')


#     return res


# #TGV-regularized reconstruction via dualizing the data term
# def tgv_recon(**par_in):


#     #Initialize parameters and data input
#     par = parameter({})
#     data_in = data_input({})
    
#     ##Set data
#     data_in.u0 = 0 #Direct image input
#     data_in.mask = 0 #Inpaint requires a mask
#     data_in.corrupted = 0 #Direct input of corrupted data

#     #Possible forward operator. Standard (False) sets identity. Needs to have F.outdim, F.nrm, F.fwd, F.adj
#     data_in.F = False

#     #Version information:
#     par.version='Version 0'


#     #Load image from name
#     par.imname = 'barbara_crop.png'
   
#     par.dtype='l2sq' #Data type: {'l1','l2sq','inpaint','I0'}
    
#     par.niter = 1

#     par.ld = 20.0 #Data missfit

#     #TGV parameter
#     par.alpha1 = np.sqrt(2.0)
#     par.alpha2 = np.sqrt(2.0)
    
#     par.order = 2 #Order of TGV regularizatoin, can be 2,3
    
#     par.noise=0.1 #Standard deviaion of gaussian noise in percent of image range
    
#     par.s_t_ratio = 100.0

#     par.check = 10 #Show results ever "check" iteration
    
#     par.nrm_red = 1.0 #Factor for norm reduction, no convergence guarantee with <1

#     par.show = False


#     ##Data and parmaeter parsing
    
#     #Set parameters according to par_in
#     par_parse(par_in,[par,data_in])

#     #Load data or image
#     if not np.any(data_in.u0):
#         data_in.u0 = imread(par.imname)
#         if len(data_in.u0.shape)==3:
#             data_in.u0 = 0.3*data_in.u0[:,:,0] + 0.59*data_in.u0[:,:,1] + 0.11*data_in.u0[:,:,2]
#     elif not par.imname:
#         par.imname='direct_input'

#     #Standard initializaton of forward operator
#     par.datadual = True
#     if not data_in.F:
#         data_in.F = id(data_in.u0.shape)
#         par.datadual = False
              
#     ## Data initilaization
#     F = data_in.F
#     u0 = np.copy(data_in.u0)
#     u0 = F.fwd(u0)
    
#     #np.random.seed(100)
#     np.random.seed(1)
    
#     #Add noise
#     if par.noise:
#         rg = np.abs(u0.max() - u0.min()) #Image range
#         u0 += np.random.normal(scale=par.noise*rg,size=u0.shape) #Add noise

#     if np.any(data_in.mask):
#         data_in.mask = (data_in.mask).astype(np.int32)
#         u0[data_in.mask==0] = 0.0


#     if np.any(data_in.corrupted):
#         u0 = np.copy(data_in.corrupted)
    

#     #Image size
#     N,M = u0.shape

#     #Operators and norms
#     grad = gradient(u0.shape)
#     sgrad = symgrad(grad.outdim)
#     sgrad2 = symgrad2(sgrad.outdim)

#     dnrm = nfun(par.dtype,mshift=u0,npar=par.ld,mask=data_in.mask)
#     l1vec = nfun('l1',vdims=(2),npar=1.0)
#     l1mat = nfun('l1',vdims=(2),npar=par.alpha1*1.0,vecweights=[1,1,2])
#     l1ten = nfun('l1',vdims=(2),npar=par.alpha2*1.0,vecweights=[1,1,3,3])


#     ##Stepsize
#     if par.order == 1:
#         opnorms = [[grad.nrm]]
#         fwd_nrm = [0]
#     if par.order == 2:
#         opnorms = [ 
#         [grad.nrm,1],
#         [0,sgrad.nrm]
#         ]
#         fwd_nrm = [0,0]
#     if par.order == 3:
#         opnorms = [ 
#         [grad.nrm,1,0],
#         [0,sgrad.nrm,1],
#         [0,0,sgrad2.nrm]
#         ]
#         fwd_nrm = [0,0,0]
#     if par.datadual:
#         fwd_nrm[0] = F.nrm
#         opnorms.append(fwd_nrm)
        
#     nrm = par.nrm_red*get_product_norm(opnorms) 

#     print('Estimated norm: ' + str(nrm))
    
#     par.sig = par.s_t_ratio/nrm
#     par.tau = 1/(nrm*par.s_t_ratio)

#     print('Stepsizes: sig: '+str(par.sig)+' / tau: '+str(par.tau))

#     #Image size
#     N,M = u0.shape


#     ##Variables


#     u = np.zeros(u0.shape)
#     ux = np.zeros(u.shape)
    
#     p = np.zeros(grad.outdim)
    
#     if par.order >1:
#         v = np.zeros(grad.outdim)
#         vx = np.zeros(v.shape)
#         q = np.zeros(sgrad.outdim)
#     if par.order >2:
#         w = np.zeros(sgrad.outdim)
#         wx = np.zeros(w.shape)
#         r = np.zeros(sgrad2.outdim)
        

#     if par.datadual: #If we have a forward operator, the data term needs to be dualized
#         d = np.zeros(F.outdim)

#     ob_val = np.zeros([par.niter+1])

#     ob_val[0] = dnrm.val(F.fwd(u))
#     if par.order == 1:
#         ob_val[0] += l1vec.val(grad.fwd(u))
#     if par.order == 2:
#         ob_val[0] += l1vec.val(grad.fwd(u) - v) + l1mat.val(sgrad.fwd(v))
#     if par.order == 3:
#         ob_val[0] += l1vec.val(grad.fwd(u) - v) + l1mat.val(sgrad.fwd(v) - w) + l1ten.val(sgrad2.fwd(w))

    
#     for k in range(par.niter):

        
#         #Dual
#         if par.datadual:
#             d = d + par.sig*( F.fwd(ux) )
#             d = dnrm.dprox(d,ppar=par.sig)

#         if par.order == 1:

#             p = p + par.sig*( grad.fwd(ux) )
#             p = l1vec.dprox(p,ppar=par.sig)

#         if par.order == 2:
                                
#             p = p + par.sig*( grad.fwd(ux) - vx )
#             p = l1vec.dprox(p,ppar=par.sig)
                    
#             q = q + par.sig*( sgrad.fwd(vx) )
#             q = l1mat.dprox(q,ppar=par.sig)

#         if par.order == 3:

#             p = p + par.sig*( grad.fwd(ux) - vx )
#             p = l1vec.dprox(p,ppar=par.sig)
                    
#             q = q + par.sig*( sgrad.fwd(vx) - wx)
#             q = l1mat.dprox(q,ppar=par.sig)

#             r = r + par.sig*( sgrad2.fwd(wx))
#             r = l1ten.dprox(r,ppar=par.sig)

        
#         #Primal
#         if par.datadual: 
#             ux = u - par.tau*(grad.adj(p) + F.adj(d) )
#         else:
#             ux = dnrm.prox( u - par.tau*(grad.adj(p))   ,ppar=par.tau)

#         u = 2.0*ux - u
#         [u,ux] = [ux,u]
                
#         if par.order >1:
   
#             vx = v - par.tau*( -p + sgrad.adj(q))
#             v = 2.0*vx - v
#             [v,vx] = [vx,v]    
#         if par.order >2:
   
#             wx = w - par.tau*( -q + sgrad2.adj(r))
#             w = 2.0*wx - w
#             [w,wx] = [wx,w]



#         ob_val[k+1] = dnrm.val(F.fwd(u))
#         if par.order == 1:
#             ob_val[k+1] += l1vec.val(grad.fwd(u))
#         if par.order == 2:
#             ob_val[k+1] += l1vec.val(grad.fwd(u) - v) + l1mat.val(sgrad.fwd(v))
#         if par.order == 3:
#             ob_val[k+1] += l1vec.val(grad.fwd(u) - v) + l1mat.val(sgrad.fwd(v) - w) + l1ten.val(sgrad2.fwd(w))


#         if np.remainder(k,par.check) == 0:
#             print('Iteration: ' + str(k) + ' / Ob-val: ' + str(ob_val[k]) + ' / Image: '+par.imname)

#     #Initialize output class
#     res = output(par)

#     #Set original input
#     res.orig = data_in.u0
#     res.u0 = u0

#     #Save variables
#     res.u = u
#     if par.order >1:
#         res.v = v
#         res.sgrad = sgrad
#     if par.order >2:
#         res.w = w
#         res.sgrad2 = sgrad2
        
#     res.ob_val = ob_val

#     #Save operators
#     res.grad = grad

#     res.F = F
    
#     #Save parmeters and input data
#     res.par = par
#     res.par_in = par_in
#     res.data_in = data_in
    
#     if par.show:
#         imshow(np.concatenate((res.orig,res.u0,res.u),axis=1),title='Orig vs noisy vs rec')


#     return res




# #Function to save the output and compute PSNR values
# def save_output(res,with_psnr=True,folder=''):

#     #Create folder if necessary
#     if not os.path.exists(folder):
#         os.makedirs(folder)
        
#     #Generate output name
#     outname = res.output_name(folder=folder)
#     #Save result file
#     res.save(folder=folder)
    
#     if with_psnr:
#         psnr_val = np.round(psnr(res.u,res.orig,smax = np.abs(res.orig.max()-res.orig.min()),rescaled=True),decimals=2)
#         print( outname + '\nPSNR: ' + str(psnr_val) )
#     else:
#         psnr_val = np.nan

#     res_txt = open(folder+'psnr_results.txt','a') 
#     res_txt.write('TGV '+res.par.imname + ' PSNR = '+str(psnr_val)+'\n')
#     res_txt.close()

#     #Save original image
#     #rg = [res.orig.min(),res.orig.max()]
#     rg = []
#     imsave(outname+'_orig.png',res.orig,rg=rg)
#     #Save data images
#     imsave(outname+'_data.png',res.u0,rg=rg)
#     #Save reconstructed image
#     imsave(outname+'_recon.png',res.u,rg=rg)




