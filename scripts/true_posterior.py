'''

script to generate the true posterior

'''
import os 
import emcee
import numpy as np 
import scipy.optimize as op
# --- mnulfi --- 
import mnulfi.util as UT
import mnulfi.data as Data
import mnulfi.geepee as GeeP
# --- plotting --- 
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt
import corner as DFM
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


def truePosterior(nwalkers=100, burn_in_chain=200, main_chain=1000): 
    ''' sample the true posterior with MCMC and the true likelihood used to 
    sample the data 
    '''
    Cov     = Data.covariance(scaled=True) # scaled covariance matrix
    Cinv    = np.linalg.inv(Cov) # precision matrix 
    # prior range 
    thetas  = Data.theta_LHC() # LHC thetas
    prior_range = np.array([[thetas[:,i].min(), thetas[:,i].max()] for i in range(thetas.shape[1])]) 
    # read in fiducial peak counts -- i.e. our data  
    _, peaks_fid = Data.PeakCnts_fiducial() 
    mupeak_fid = np.mean(peaks_fid, axis=0) 
    
    # read in GP emulator (our model) 
    geep = GeeP.peakEmu()

    def lnprior(tt): 
        ''' log uniform prior 
        '''
        t0, t1, t2 = tt 
        if (prior_range[0][0] <= t0 <= prior_range[0][1]) and (prior_range[1][0] <= t1 <= prior_range[1][1]) and (prior_range[2][0] <= t2 <= prior_range[2][1]):
            return 0.0 
        return -np.inf 

    def lnlike(tt): 
        ''' log likelihood. Becuase all the data is sampled from the multivariate 
        gaussian N(GP emulator, covariance) this is the *true* likelihood 
        '''
        delta = (geep.predict(tt) - mupeak_fid) 
        return -0.5*np.dot(delta, np.dot(Cinv, delta.T))

    def lnprob(tt):
        lp = lnprior(tt)
        if not np.isfinite(lp): 
            return -np.inf 
        return lp + lnlike(tt)

    # initialize walkers randomly drawn from prior 
    ndim = thetas.shape[1]  
    pos = [np.random.uniform(prior_range[:,0], prior_range[:,1]) for i in range(nwalkers)]
    
    # run mcmc 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    print('running burn-in chain') 
    pos, prob, state = sampler.run_mcmc(pos, burn_in_chain)
    #sampler.reset()
    #print('running main chain') 
    #sampler.run_mcmc(pos, main_chain)
     
    # save chain 
    post = sampler.flatchain.copy() 
    post.dump(os.path.join(UT.dat_dir(), 'true_posterior.chain.npy'))

    fig = DFM.corner(post, labels=[r'$M_\nu$', '$\Omega_m$', '$A_s$'], color='C0', 
            quantiles=[0.16, 0.5, 0.84], bins=20, range=prior_range, 
            smooth=True, show_titles=True, label_kwargs={'fontsize': 20}) 
    
    # overplot the LHC cosmologies 
    axes = np.array(fig.axes).reshape(3,3)
    for i in range(3): 
        for j in range(i): 
            ax = axes[i,j] 
            ax.scatter(thetas[:,j], thetas[:,i], c='k', s=10, marker='x') 

    ffig = os.path.join(os.environ['MNULFI_FIGDIR'], 'true_posterior.png')
    fig.savefig(ffig, bbox_inch='tight')
    return None 


if __name__=="__main__":
    truePosterior(nwalkers=100, burn_in_chain=200, main_chain=1000)
