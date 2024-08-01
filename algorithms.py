import numpy as np
from util import *
import seaborn as sns
import time
from matpy import tv_denoise

# proximal-subgradient method for sampling from potential F(x)+G(Kx). A proximal step wrt. F and a subgradient step wrt. GoK is used.


class subgradient_Langevin(object):
    def __init__(self,**par_in):

        par = parameter({})
        data_in = data_input({})

        ##Set data
        data_in.u0 = 0 #Direct image input

        par.niter = 100000
        par.n_parallel_chains = int(500)
        fac = .01*.5
        par.reg_par = fac*200
        par.data_par = fac
        par.noise = 0
        par.tau = 5e-3
        par.ld = 1e-3
        par.check = 500
        par.save_every = 0
        par.metropolis_check = False
        par.folder = './'

        #Data type: {'l1','l2sq','inpaint','I0'}
        #par.F_name='l2sq'
        par.F_name='l2sq'
        par.G_name='l1'
        par.blur_kernel = np.ones([5,5])/25
        par.K = []
        par_parse(par_in,[par,data_in])


        x_0 = np.concatenate(par.n_parallel_chains*[data_in.u0[...,np.newaxis]],axis = -1)

        if len(data_in.u0.shape)==1:
            if par.K==[]:
                par.K = gradient_1d(x_0.shape)

            par.F = nfun(par.F_name, npar=par.data_par, mshift=x_0, dims = tuple(range(len(data_in.u0.shape))))
            par.G = nfun(par.G_name, npar=par.reg_par, dims = tuple(range(len(data_in.u0.shape))))

        elif len(data_in.u0.shape)==2:
            if par.K==[]:
                par.K = gradient(x_0.shape)

            par.F = nfun(par.F_name, npar=par.data_par, blur_kernel=par.blur_kernel, mshift=x_0, dims = tuple(range(len(data_in.u0.shape))))
            vdims = ()#2
            dims = (0,1,2)
            par.G = nfun(par.G_name, npar=par.reg_par, dims = dims,vdims=vdims)

        self.par = par
        self.data_in = data_in


    def metropolis_check(self, old, proposal, p_old_proposal, p_proposal_old):
        acceptance_crit = ( - (self.par.F.val(proposal) + self.par.G.val(self.par.K.fwd(proposal))) + (self.par.F.val(old) + self.par.G.val(self.par.K.fwd(old)))).flatten()
        acceptance_crit = acceptance_crit + p_proposal_old - p_old_proposal
        acceptance_crit = np.exp(acceptance_crit)
        acceptance_rate = np.random.uniform(low = 0., high = 1.,size = self.par.n_parallel_chains)
        acceptance = (acceptance_crit>acceptance_rate)*1.0
        acc = np.copy(acceptance)
        acceptance = acceptance[np.newaxis,...]
        if len(self.data_in.u0.shape)==2:
            acceptance = acceptance[np.newaxis,...]
        return proposal*acceptance + old*(1 - acceptance),acc

    def grad_subgrad(self,average_distribution=False,burnin = 0):

        measure_times_per_this_many_iterates = 1000
        times = np.zeros(self.par.niter//measure_times_per_this_many_iterates)

        x_0 = np.concatenate(self.par.n_parallel_chains*[self.data_in.u0[...,np.newaxis]],axis = -1)
        x_k = np.copy(x_0)
        
        if self.par.save_every:
            np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_x0.npy',x_0)

        if average_distribution:
            stopping_times = np.concatenate([np.random.randint(burnin, high = K+1,size = self.par.n_parallel_chains)[:,np.newaxis] for K in range(burnin,self.par.niter+1,min(self.par.niter,self.par.save_every))], axis=-1)
            sample = np.zeros([*x_0.shape,stopping_times.shape[-1]])

        running_mmse = np.zeros(x_k.shape)
        running_var_uncentered = np.zeros(x_k.shape)

        t=0
        acc_rate=0
        for k in range(self.par.niter):

            if k%measure_times_per_this_many_iterates==0 and k>0:
                times[k//measure_times_per_this_many_iterates-1] = t
                print('Average time for 1000 iterations')
                print(np.sum(times)/np.sum(times>0))
                t = 0
            
            if self.par.check:
                if k%self.par.check==0 and k>burnin:
                    if len(self.data_in.u0.shape)==2:
                        f, axarr = plt.subplots(3)
                        axarr[0].imshow(np.mean(running_mmse,axis=(-1)),cmap = 'gray')
                        axarr[0].set_title('MMSE after '+str(k)+' iterations')
                        axarr[1].imshow(np.mean(x_0,axis=(-1)),cmap = 'gray')
                        axarr[1].set_title('Initial')
                        axarr[2].imshow(np.mean(running_var_uncentered-running_mmse**2,axis=(-1)),cmap = 'hot')
                        axarr[2].set_title('Marginal posterior variances')
                        plt.show()
                    else:
                        f, axarr = plt.subplots(2,2)
                        axarr[0,0].plot(np.mean(x_k_intermediate,axis=-1))
                        axarr[0,0].set_title('MMSE after '+str(k)+' iterations')
                        axarr[0,1].plot(np.var(x_k_intermediate,axis=-1))
                        axarr[1,0].plot(np.mean(x_0,axis=-1))
                        plt.show()


            if average_distribution:
                picks = 1.0*(stopping_times==k)
                picks = picks[np.newaxis,...]
                if len(self.data_in.u0.shape)==2:
                    picks = picks[np.newaxis,...]
                sample += picks*x_k[...,np.newaxis]

            
            dt = time.time()

            Kx_k = self.par.K.fwd(x_k)
            # compute subgradient of G at point Kx via prox
            subgrad_G = self.par.G.subgrad(Kx_k)
            # subgrad step
            x_k_intermediate = x_k - self.par.tau * self.par.K.adj(subgrad_G)
            
            if self.par.metropolis_check and k>100:

                gauss = np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)
                x_k_prop = x_k_intermediate - self.par.tau*self.par.F.subgrad(x_k_intermediate) + gauss

                Kx_k_prop = self.par.K.fwd(x_k_prop)
                # compute subgradient of G at point Kx via prox
                subgrad_G = self.par.G.subgrad(Kx_k_prop)
                # subgrad step
                x_k_prop_rev = x_k_prop - self.par.tau * self.par.K.adj(subgrad_G)
                x_k_prop_rev = x_k_prop_rev - self.par.tau*self.par.F.subgrad(x_k_prop_rev)
                
                q_x_y = -(gauss*gauss).sum( axis=tuple(range(len(gauss.shape)-1)) )/(4*self.par.tau)
                q_y_x = -((x_k_prop_rev-x_k)**2).sum( axis=tuple(range(len(x_k.shape)-1)) ) /(4*self.par.tau)

                x_k,acc = self.metropolis_check(x_k,x_k_prop,q_x_y,q_y_x)

                x_k_intermediate = x_k

                acc_rate = (k*acc_rate+acc)/(k+1)
                #print(acc_rate)

            else:
                x_k = x_k_intermediate - self.par.tau*self.par.F.subgrad(x_k_intermediate) + np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)


            dt = time.time() - dt
            t += dt

            if  k>= burnin:
                running_mmse = running_mmse*(k-burnin)/(k-burnin+1) + x_k_intermediate/(k-burnin+1)
                running_var_uncentered = running_var_uncentered*(k-burnin)/(k-burnin+1) + x_k_intermediate**2/(k-burnin+1)


            if self.par.save_every:

                if k%self.par.save_every==0:
                    np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_computation_times.npy',times)

                if k%self.par.save_every==0 and k>=burnin:
                    np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'.npy',x_k_intermediate)

                    if average_distribution and k>=burnin: 
                        np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'average_distribution.npy',sample[...,0])
                        sample = sample[...,1:]
                        stopping_times = stopping_times[...,1:]

                    if k>= burnin:
                        np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'_mmse.npy',running_mmse)
                        np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'_variance.npy',running_var_uncentered - running_mmse**2)


        if average_distribution:
            res = {'xk':x_k_intermediate,'average_distribution_sample':sample,'x0':x_0}
        else:
            res = {'xk':x_k_intermediate,'x0':x_0}
        return res


    def prox_subgrad(self,average_distribution=False,burnin=0):
        
        x_0 = np.concatenate(self.par.n_parallel_chains*[self.data_in.u0[...,np.newaxis]],axis = -1)
        
        measure_times_per_this_many_iterates = 1000
        times = np.zeros(self.par.niter//measure_times_per_this_many_iterates)


        x_k = np.copy(x_0)
        if self.par.save_every:
            np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_x0.npy',x_0)

        if average_distribution:
            stopping_times = np.concatenate([np.random.randint(burnin, high = K+1,size = self.par.n_parallel_chains)[:,np.newaxis] for K in range(burnin,self.par.niter+1,min(self.par.niter,self.par.save_every))], axis=-1)
            sample = np.zeros([*x_0.shape,stopping_times.shape[-1]])

        running_mmse = np.zeros(x_k.shape)
        running_var_uncentered = np.zeros(x_k.shape)
        

        t=0

        for k in range(self.par.niter):

            if k%measure_times_per_this_many_iterates==0 and k>0:
                times[k//measure_times_per_this_many_iterates-1] = t
                print('Average time for 1000 iterations')
                print(np.sum(times)/np.sum(times>0))
                t = 0

        
            if self.par.check:
                if k%self.par.check==0 and k>burnin:
                    if len(self.data_in.u0.shape)==2:
                        f, axarr = plt.subplots(3)
                        axarr[0].imshow(np.mean(running_mmse,axis=-1),cmap = 'gray')
                        axarr[0].set_title('MMSE after '+str(k)+' iterations')
                        axarr[1].imshow(np.mean(x_0,axis=-1),cmap = 'gray')
                        axarr[1].set_title('Initial')
                        axarr[2].imshow(np.mean(running_var_uncentered-running_mmse**2,axis=-1),cmap = 'hot')
                        axarr[2].set_title('Marginal posterior variances')
                        plt.show()
                    else:
                        f, axarr = plt.subplots(2,2)
                        axarr[0,0].plot(np.mean(x_k_intermediate,axis=-1))
                        axarr[0,0].set_title('MMSE after '+str(k)+' iterations')
                        axarr[0,1].plot(np.var(x_k_intermediate,axis=-1))
                        axarr[1,0].plot(np.mean(x_0,axis=-1))
                        plt.show()
            

            if average_distribution:
                picks = 1.0*(stopping_times==k)
                picks = picks[np.newaxis,...]
                if len(self.data_in.u0.shape)==2:
                    picks = picks[np.newaxis,...]
                sample += picks*x_k[...,np.newaxis]
            
            dt = time.time()

            Kx_k = self.par.K.fwd(x_k)
            # compute subgradient of G at point Kx
            subgrad_G = self.par.G.subgrad(Kx_k)
            # subgrad step
            x_k_intermediate = x_k - self.par.tau * self.par.K.adj(subgrad_G)

            x_k_intermediate = x_k - self.par.tau * self.par.K.adj(subgrad_G)
            if self.par.metropolis_check and k>100:
                gauss = np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)
                x_k_prop = self.par.F.prox(x_k_intermediate, ppar = self.par.tau) + gauss

                q_x_y = np.exp( -(gauss*gauss).sum( axis=tuple(range(len(gauss.shape)-1)) )/(4*self.par.tau) )

                Kx_k_prop = self.par.K.fwd(x_k_prop)
                # compute subgradient of G at point Kx via prox
                subgrad_G = self.par.G.subgrad(Kx_k_prop)
                # subgrad step
                x_k_prop_rev = x_k_prop - self.par.tau * self.par.K.adj(subgrad_G)
                x_k_prop_rev = self.par.F.prox(x_k_prop_rev, ppar = self.par.tau)

                q_y_x = np.exp( -((x_k_prop_rev-x_k)**2).sum( axis=tuple(range(len(gauss.shape)-1)) ) /(4*self.par.tau))

                x_k = self.metropolis_check(x_k,x_k_prop,q_x_y,q_y_x)
                x_k_intermediate = x_k


            else:
                x_k = self.par.F.prox(x_k_intermediate, ppar = self.par.tau) + np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)

            dt = time.time() - dt
            t += dt

            if  k>= burnin:
                running_mmse = running_mmse*(k-burnin)/(k-burnin+1) + x_k_intermediate/(k-burnin+1)
                running_var_uncentered = running_var_uncentered*(k-burnin)/(k-burnin+1) + x_k_intermediate**2/(k-burnin+1)


            if self.par.save_every:
                if k%self.par.save_every==0:
                    np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_computation_times.npy',times)

                if k%self.par.save_every==0 and k>=burnin:
                    np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'.npy',x_k_intermediate)

                    if average_distribution and k>=burnin: 
                        np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'average_distribution.npy',sample[...,0])
                        sample = sample[...,1:]
                        stopping_times = stopping_times[...,1:]

                    if k>= burnin:
                        np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'_mmse.npy',running_mmse)
                        np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'_variance.npy',running_var_uncentered - running_mmse**2)


        if average_distribution:
            res = {'xk':x_k_intermediate,'average_distribution_sample':sample,'x0':x_0}
        else:
            res = {'xk':x_k_intermediate,'x0':x_0}
        return res

    def subgrad(self,burnin=0):


        assert (not self.par.metropolis_check), "Metropolis correction not implemented for this algorithm"

        x_0 = np.concatenate(self.par.n_parallel_chains*[self.data_in.u0[...,np.newaxis]],axis = -1)
        x_k = np.copy(x_0)

        if self.par.save_every:
            np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_x0.npy',x_0)


        stopping_times = np.concatenate([np.random.randint(burnin, high = K+1,size = self.par.n_parallel_chains)[:,np.newaxis] for K in range(burnin,self.par.niter+1,self.par.save_every)], axis=-1)
        stopping_times = np.sort(stopping_times, axis = 0)
        sample = np.zeros([*x_0.shape,stopping_times.shape[-1]])

        for k in range(self.par.niter):

            if self.par.check:
                if k%self.par.check==0 and k>0:
                    if len(self.data_in.u0.shape)==2:
                        f, axarr = plt.subplots(2)
                        axarr[0].imshow(np.mean(x_k,axis=-1),cmap = 'gray')
                        axarr[0].set_title('MMSE after '+str(k)+' iterations. '+str(x_k.shape[-1]))
                        sns.heatmap(np.log(np.var(x_k,axis=-1)), ax = axarr[1])
                        axarr[1].set_title('Marginal posterior variances')
                        plt.show()
                    else:
                        f, axarr = plt.subplots(2,2)
                        axarr[0,0].plot(np.mean(sample[...,:stopping_time_index-1],axis=-1))
                        axarr[0,0].set_title('MMSE after '+str(k)+' iterations. '+str(stopping_time_index))
                        axarr[0,1].plot(np.var(sample[...,:stopping_time_index-1],axis=-1))
                        axarr[1,0].plot(np.mean(x_0,axis=-1))
                        plt.show()
                    

            picks = 1.0*(stopping_times==k)
            picks = picks[np.newaxis,...]
            if len(self.data_in.u0.shape)==2:
                picks = picks[np.newaxis,...]
            sample += picks*x_k[...,np.newaxis]



            Kx_k = self.par.K.fwd(x_k)
            # compute subgradient of G at point Kx via prox
            subgrad_G = self.par.G.subgrad(Kx_k)
            # subgrad step
            x_k_intermediate = x_k - self.par.tau*(self.par.F.subgrad(x_k) + self.par.K.adj(subgrad_G))
            x_k = x_k_intermediate + np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)


            if self.par.save_every:
                if k%self.par.save_every==0 and k>=burnin:
                    np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'average_distribution.npy',sample[...,0])
                    
                    sample = sample[...,1:]
                    stopping_times = stopping_times[...,1:]


            if k==0:
                np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_0.npy',x_k_intermediate)


        return {'average_distribution_sample':sample,'x0':x_0}


class MYULA(object):
    def __init__(self,**par_in):

        par = parameter({})
        data_in = data_input({})

        ##Set data
        data_in.u0 = 0 #Direct image input

        par.niter = 100000
        fac = .01*.5
        par.reg_par = fac*200
        par.data_par = fac
        par.noise = 0
        par.ld = 1/par.data_par
        par.tau = par.ld/4
        par.n_parallel_chains=1
        par.check = 500
        par.save_every = 0
        par.folder = './'

        #Data type: {'l1','l2sq','inpaint','I0'}
        #par.F_name='l2sq'
        par.F_name='l2sq'
        par.G_name='l1'
        par.blur_kernel = np.ones([5,5])/25
        par.K = []
        par_parse(par_in,[par,data_in])


        x_0 = np.concatenate(par.n_parallel_chains*[data_in.u0[...,np.newaxis]],axis = -1)

        if len(data_in.u0.shape)==1:
            if par.K==[]:
                par.K = gradient_1d(x_0.shape)

            par.F = nfun(par.F_name, npar=par.data_par, mshift=x_0, dims = tuple(range(len(data_in.u0.shape))))
            par.G = nfun(par.G_name, npar=par.reg_par, dims = tuple(range(len(data_in.u0.shape))))

        elif len(data_in.u0.shape)==2:
            if par.K==[]:
                par.K = gradient(x_0.shape)

            par.n_parallel_chains=1

            par.F = nfun(par.F_name, npar=par.data_par, blur_kernel=par.blur_kernel, mshift=x_0, dims = tuple(range(len(data_in.u0.shape))))
            vdims = ()#2
            dims = (0,1,2)
            par.G = nfun(par.G_name, npar=par.reg_par, dims = dims,vdims=vdims)
        
        self.par = par
        self.data_in = data_in

    def sample(self,burnin = 0, prox_iter=-1):

        measure_times_per_this_many_iterates = 1000
        times = np.zeros(self.par.niter//measure_times_per_this_many_iterates)

        x_0 = np.concatenate(self.par.n_parallel_chains*[self.data_in.u0[...,np.newaxis]],axis = -1)
        x_k = np.copy(x_0)
        
        if self.par.save_every:
            np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_x0.npy',x_0)


        running_mmse = np.zeros(x_k.shape)
        running_var_uncentered = np.zeros(x_k.shape)

        t = 0
        for k in range(self.par.niter):

            if k%measure_times_per_this_many_iterates==0 and k>0:
                times[k//measure_times_per_this_many_iterates-1] = t
                print('Average time for 1000 iterations')
                print(np.sum(times)/np.sum(times>0))
                t = 0

            if self.par.check:
                if k%self.par.check==0 and k>burnin:
                    if len(self.data_in.u0.shape)==2:
                        f, axarr = plt.subplots(3)
                        #axarr[0,0].imshow(np.mean(x_k_list[:,:,burnin:k],axis=2),cmap = 'gray')
                        axarr[0].imshow(np.squeeze(running_mmse),cmap = 'gray')
                        axarr[0].set_title('MMSE after '+str(k)+' iterations')
                        axarr[1].imshow(np.squeeze(x_0),cmap = 'gray')
                        axarr[1].set_title('Initial')
                        axarr[2].imshow(np.squeeze(running_var_uncentered-running_mmse**2),cmap = 'hot')
                        #axarr[2,0].imshow(np.var(x_k_list[:,:,burnin:k],axis=2),cmap = 'hot')
                        axarr[2].set_title('Marginal posterior variances')

                        plt.show()

            dt = time.time()

            grad_F = self.par.F.subgrad(x_k)
            prox_G_K = tv_denoise(u0=x_k,ld = 1/(self.par.reg_par*self.par.ld), niter = prox_iter).u
            x_k_intermediate = x_k*(1-self.par.tau/self.par.ld) - self.par.tau*grad_F + prox_G_K*(self.par.tau/self.par.ld) 
            x_k = x_k_intermediate + np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)


            dt = time.time()-dt
            t += dt

            if  k>= burnin:
                running_mmse = running_mmse*(k-burnin)/(k-burnin+1) + x_k/(k-burnin+1)
                running_var_uncentered = running_var_uncentered*(k-burnin)/(k-burnin+1) + x_k**2/(k-burnin+1)


            if self.par.save_every:

                if k%self.par.save_every==0:
                    np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'ld_'+str(self.par.ld)+'_'
                                        'tau_'+str(self.par.tau)+'_computation_times.npy',times)

                if k%self.par.save_every==0 and k>=burnin:
                    np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'ld_'+str(self.par.ld)+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'.npy',np.squeeze(x_k))


                    if k>= burnin:
                        np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'ld_'+str(self.par.ld)+'_'
                                        'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'_mmse.npy',np.mean(running_mmse,axis=-1))
                        np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'ld_'+str(self.par.ld)+'_'
                                        'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'_variance.npy',np.mean(running_var_uncentered,axis=-1) - np.mean(running_mmse,axis=-1)**2)


        res = {'xk':x_k,'x0':x_0}
        return res





class MALA(object):
    def __init__(self,**par_in):

        par = parameter({})
        data_in = data_input({})

        ##Set data
        data_in.u0 = 0 #Direct image input

        par.niter = 100000
        fac = .01*.5
        par.reg_par = fac*200
        par.data_par = fac
        par.noise = 0
        par.tau = 1e-5
        par.check = 500
        par.n_parallel_chains = 1
        par.save_every = 0
        par.folder = './'

        #Data type: {'l1','l2sq','inpaint','I0'}
        #par.F_name='l2sq'
        par.F_name='l2sq'
        par.G_name='l1'
        par.blur_kernel = np.ones([5,5])/25
        par.K = []
        par_parse(par_in,[par,data_in])


        x_0 = np.concatenate(par.n_parallel_chains*[data_in.u0[...,np.newaxis]],axis = -1)

        if len(data_in.u0.shape)==1:
            if par.K==[]:
                par.K = gradient_1d(x_0.shape)

            par.F = nfun(par.F_name, npar=par.data_par, mshift=x_0, dims = tuple(range(len(data_in.u0.shape))))
            par.G = nfun(par.G_name, npar=par.reg_par, dims = tuple(range(len(data_in.u0.shape))))

        elif len(data_in.u0.shape)==2:
            if par.K==[]:
                par.K = gradient(x_0.shape)

            par.n_parallel_chains=1

            par.F = nfun(par.F_name, npar=par.data_par, blur_kernel=par.blur_kernel, mshift=x_0, dims = tuple(range(len(data_in.u0.shape))))
            vdims = ()#2
            dims = (0,1,2)
            par.G = nfun(par.G_name, npar=par.reg_par, dims = dims,vdims=vdims)

        self.par = par
        self.data_in = data_in

    def metropolis_check(self, old, proposal, p_old_proposal, p_proposal_old):
        acceptance_crit = ( - (self.par.F.val(proposal) + self.par.G.val(self.par.K.fwd(proposal))) + (self.par.F.val(old) + self.par.G.val(self.par.K.fwd(old)))).flatten()
        acceptance_crit = acceptance_crit + p_proposal_old - p_old_proposal
        acceptance_crit = np.exp(acceptance_crit)
        acceptance_rate = np.random.uniform(low = 0., high = 1.,size = 1)
        acceptance = (acceptance_crit>acceptance_rate)*1.0
        acc = np.copy(acceptance)
        acceptance = acceptance[np.newaxis,...]
        if len(self.data_in.u0.shape)==2:
            acceptance = acceptance[np.newaxis,...]
        return proposal*acceptance + old*(1 - acceptance),acc

    def sample(self,burnin = 0, prox_iter=-1):

        measure_times_per_this_many_iterates = 1000
        times = np.zeros(self.par.niter//measure_times_per_this_many_iterates)

        x_0 = np.concatenate(self.par.n_parallel_chains*[self.data_in.u0[...,np.newaxis]],axis = -1)
        x_k = np.copy(x_0)
        
        if self.par.save_every:
            np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_x0.npy',x_0)


        running_mmse = np.zeros(x_k.shape)
        running_var_uncentered = np.zeros(x_k.shape)

        acceptance_rate = 0

        
        t = 0#time.time()
        for k in range(self.par.niter):

            if k%measure_times_per_this_many_iterates==0 and k>0:
                times[k//measure_times_per_this_many_iterates-1] = t
                print('Average time for 1000 iterations')
                print(np.sum(times)/np.sum(times>0))
                t = 0

            if self.par.check:
                if k%self.par.check==0 and k>burnin:
                    if len(self.data_in.u0.shape)==2:
                        f, axarr = plt.subplots(3)
                        #axarr[0,0].imshow(np.mean(x_k_list[:,:,burnin:k],axis=2),cmap = 'gray')
                        axarr[0].imshow(np.squeeze(running_mmse),cmap = 'gray')
                        axarr[0].set_title('MMSE after '+str(k)+' iterations')
                        axarr[1].imshow(np.squeeze(x_0),cmap = 'gray')
                        axarr[1].set_title('Initial')
                        axarr[2].imshow(np.squeeze(running_var_uncentered-running_mmse**2),cmap = 'hot')
                        #axarr[2,0].imshow(np.var(x_k_list[:,:,burnin:k],axis=2),cmap = 'hot')
                        axarr[2].set_title('Marginal posterior variances')

                        plt.show()



            dt = time.time()

            # we approximate prox of the sum. in case of denoising this could be improved
            grad_F = self.par.F.subgrad(x_k)
            x_k_intermediate = tv_denoise(u0=x_k-self.par.tau*grad_F,ld = 1/(self.par.reg_par*self.par.tau), niter = prox_iter).u

            # metropolis:
            gauss = np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)
            x_k_prop = x_k_intermediate + gauss

            # MALA step starting at x_k_prop
            grad_F_rev = self.par.F.subgrad(x_k_prop)
            x_k_prop_rev = tv_denoise(u0=x_k_prop-self.par.tau*grad_F_rev,ld = 1/(self.par.reg_par*self.par.tau), niter = prox_iter).u

            q_x_y = -(gauss*gauss).sum( axis=tuple(range(len(gauss.shape)-1)) )/(4*self.par.tau)
            q_y_x = -((x_k_prop_rev-x_k)**2).sum( axis=tuple(range(len(x_k.shape)-1)) ) /(4*self.par.tau)

            x_k,acceptance = self.metropolis_check(x_k,x_k_prop,q_x_y,q_y_x)

            dt = time.time() - dt
            t += dt


            #x_k = x_k_prop
            acceptance_rate = (acceptance_rate*k+acceptance)/(k+1)


            if  k>= burnin:
                running_mmse = running_mmse*(k-burnin)/(k-burnin+1) + x_k/(k-burnin+1)
                running_var_uncentered = running_var_uncentered*(k-burnin)/(k-burnin+1) + x_k**2/(k-burnin+1)


            if self.par.save_every:
                if k%self.par.save_every==0:
                    np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_computation_times.npy',times)

                if k%self.par.save_every==0 and k>=burnin:
                    np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'.npy',np.squeeze(x_k))


                    if k>= burnin:
                        np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'_mmse.npy',np.squeeze(running_mmse))
                        np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'_variance.npy',np.squeeze(running_var_uncentered - running_mmse**2))


        res = {'xk':x_k,'x0':x_0}
        return res