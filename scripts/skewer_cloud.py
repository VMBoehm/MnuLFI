'''

skewer versus cloud parameter space sampling test

skewer: 
    |       x
    |   x   x
  X |   x   x
    |   x
    |-----------
        theta 
cloud: 
    |        x
    |   x x x
  X |  xx  x
    |   
    |-----------
        theta 
'''
import os 
import h5py 
import pickle 
import numpy as np 
import scipy as sp 
import tensorflow as tf 
# --- pydelfi --- 
import pydelfi.ndes as NDEs
import pydelfi.delfi as DELFI 
import pydelfi.score as Score
import pydelfi.priors as Priors 
# --- sklearn ---
from sklearn.gaussian_process import GaussianProcessRegressor as GPR 
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
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


def gpPeakcounts(): 
    ''' read in peak counts and train GP 
    '''
    # read in peak counts data
    datdir = os.path.join(os.environ['MNULFI_DIR'], 'Peaks_MassiveNuS') # directory with the data 
    thetas = np.load(os.path.join(datdir, 'params_conc.full.npy')) # thetas 
    peakct = np.load(os.path.join(datdir, 'data_scaled.full.npy')) # average peak counts 

    kern = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1, (1e-4, 1e4)) # kernel
    gp = GPR(kernel=kern, n_restarts_optimizer=10) # instanciate a GP model
    gp.fit(thetas, peakct)
    return gp 


def makeSkewerCloud(seed=1): 
    ''' Generate skewer and cloud data using Gaussian Process
    '''
    np.random.seed(seed)
    # read in covariance 
    datdir = os.path.join(os.environ['MNULFI_DIR'], 'Peaks_MassiveNuS')

    thetas = np.load(os.path.join(datdir,'params_conc.full.npy')) # thetas               
    mudata = np.load(os.path.join(datdir,'data_scaled.full.npy')) # average peak counts 
    cov = np.load(os.path.join(datdir, 'covariance.npy')) # covariance  

    GP = gpPeakcounts() # gaussian process 
    
    # generate skewer data 
    theta_skewers = np.tile(thetas[None,:,:], 9999).reshape(thetas.shape[0] * 9999, thetas.shape[1])
    peaks_skewers = np.random.multivariate_normal(np.zeros(50), cov, size=theta_skewers.shape[0])
    for iskew in range(thetas.shape[0]): 
        mu_peak = GP.predict(np.atleast_2d(thetas[iskew,:]))
        peaks_skewers[iskew*9999:(iskew+1)*9999,:] += mu_peak 
    peaks_skewers = np.clip(peaks_skewers, 0., None) 

    #skewer_hull = Delaunay(sp.spatial.ConvexHull(thetas))
    skewer_hull = sp.spatial.Delaunay(thetas)
    theta_lims = [(theta_skewers[:,i].min(), theta_skewers[:,i].max()) for i in range(theta_skewers.shape[1])]

    # generate cloud data 
    _theta_cloud = np.zeros((4*theta_skewers.shape[0], 3))
    _theta_cloud[:,0] = np.random.uniform(theta_lims[0][0], theta_lims[0][1], 4*theta_skewers.shape[0])
    _theta_cloud[:,1] = np.random.uniform(theta_lims[1][0], theta_lims[1][1], 4*theta_skewers.shape[0])
    _theta_cloud[:,2] = np.random.uniform(theta_lims[2][0], theta_lims[2][1], 4*theta_skewers.shape[0])

    inhull = (skewer_hull.find_simplex(_theta_cloud) >= 0) 
    theta_cloud = _theta_cloud[inhull,:][:theta_skewers.shape[0]]

    peaks_cloud = np.random.multivariate_normal(np.zeros(50), cov, size=theta_cloud.shape[0])
    mu_peaks = GP.predict(theta_cloud)
    peaks_cloud += mu_peaks
    peaks_cloud = np.clip(peaks_cloud, 0., None) 
    
    # -- plot thetas --
    fig = plt.figure(figsize=(8,4))  
    sub = fig.add_subplot(121) 
    sub.scatter(theta_cloud[::100,0], theta_cloud[::100,1], c='C0', s=2, label='Cloud') 
    sub.scatter(theta_skewers[::100,0], theta_skewers[::100,1], c='C1', s=2, label='Skewer') 
    sub.set_xlabel(r'$\theta_1$', fontsize=20) 
    sub.set_xlim(theta_lims[0])
    sub.set_ylabel(r'$\theta_2$', fontsize=20) 
    sub.set_ylim(theta_lims[1])

    sub = fig.add_subplot(122) 
    sub.scatter(theta_cloud[::100,2], theta_cloud[::100,1], c='C0', s=2, label='Cloud') 
    sub.scatter(theta_skewers[::100,2], theta_skewers[::100,1], c='C1', s=2, label='Skewer') 
    sub.legend(loc='lower right', frameon=True, handletextpad=0.1, markerscale=3, fontsize=15) 
    sub.set_xlabel(r'$\theta_3$', fontsize=20) 
    sub.set_xlim(theta_lims[2])
    sub.set_ylim(theta_lims[1])
    fig.savefig(os.path.join(datdir, 'theta.skewer_cloud.png'), bbox_inches='tight') 

    # -- plot peak counts -- 
    fig = plt.figure(figsize=(8,4))  
    sub = fig.add_subplot(111) 
    for i in range(100): 
        sub.plot(range(50), peaks_cloud[10000*i,:], c='k', lw=0.2, label='Cloud') 
        sub.plot(range(50), peaks_skewers[10000*i,:], c='C0', lw=0.2, label='Skewer') 
        if i == 0: sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlim(0, 49)
    sub.set_ylabel('peak counts', labelpad=10, fontsize=20) 
    fig.savefig(os.path.join(datdir, 'peaks.skewer_cloud.png'), bbox_inches='tight') 

    # save skewer and cloud to files
    theta_cloud.dump(os.path.join(datdir, 'theta.cloud.npy')) 
    peaks_cloud.dump(os.path.join(datdir, 'peaks.cloud.npy')) 

    theta_skewers.dump(os.path.join(datdir, 'theta.skewers.npy')) 
    peaks_skewers.dump(os.path.join(datdir, 'peaks.skewers.npy')) 

    # score compress the peak data 
    theta_fid = thetas[51,:]
    _mu_fid = GP.predict(np.atleast_2d(theta_fid))
    mu_fid = _mu_fid.flatten() 
    h = 0.01

    _mu_theta1p = GP.predict(np.atleast_2d(np.array([theta_fid[0]*(1+h), theta_fid[1], theta_fid[2]])))
    _mu_theta1m = GP.predict(np.atleast_2d(np.array([theta_fid[0]*(1-h), theta_fid[1], theta_fid[2]])))
    _mu_theta2p = GP.predict(np.atleast_2d(np.array([theta_fid[0], theta_fid[1]*(1+h), theta_fid[2]])))
    _mu_theta2m = GP.predict(np.atleast_2d(np.array([theta_fid[0], theta_fid[1]*(1-h), theta_fid[2]])))
    _mu_theta3p = GP.predict(np.atleast_2d(np.array([theta_fid[0], theta_fid[1], theta_fid[2]*(1+h)])))
    _mu_theta3m = GP.predict(np.atleast_2d(np.array([theta_fid[0], theta_fid[1], theta_fid[2]*(1-h)])))

    mu_theta1p = _mu_theta1p.flatten() 
    mu_theta1m = _mu_theta1m.flatten() 
    mu_theta2p = _mu_theta2p.flatten() 
    mu_theta2m = _mu_theta2m.flatten() 
    mu_theta3p = _mu_theta3p.flatten() 
    mu_theta3m = _mu_theta3m.flatten() 

    dmudt1 = (mu_theta1p - mu_theta1m)/(2. * h * theta_fid[0])
    dmudt2 = (mu_theta2p - mu_theta2m)/(2. * h * theta_fid[1])
    dmudt3 = (mu_theta3p - mu_theta3m)/(2. * h * theta_fid[2])
    dmudt = np.vstack((dmudt1,dmudt2,dmudt3))
    Cinv = np.linalg.inv(cov) 

    Comp = Score.Gaussian(len(mu_fid), theta_fid, mu=mu_fid, Cinv=Cinv, dmudt=dmudt)
    Comp.compute_fisher()
    Finv = Comp.Finv
    Finv.dump(os.path.join(datdir, 'peak.Finv.npy'))

    scores_cloud = np.zeros((peaks_cloud.shape[0], 3)) 
    scores_skewers = np.zeros((peaks_cloud.shape[0], 3)) 
    for i in range(peaks_cloud.shape[0]): 
        scores_cloud[i,:] = Comp.scoreMLE(peaks_cloud[i,:]) 
        scores_skewers[i,:] = Comp.scoreMLE(peaks_skewers[i,:]) 

    scores_fid = Comp.scoreMLE(mudata[51,:]) 
    scores_fid.dump(os.path.join(datdir, 'scores.fid.npy')) 
    scores_cloud.dump(os.path.join(datdir, 'scores.cloud.npy')) 
    scores_skewers.dump(os.path.join(datdir, 'scores.skewers.npy')) 

    # -- plot scores -- 
    fig = plt.figure(figsize=(8,4))  
    sub = fig.add_subplot(111) 
    for i in range(100): 
        sub.plot(range(3), scores_cloud[10000*i,:], c='k', lw=0.2, label='Cloud') 
        sub.plot(range(3), scores_skewers[10000*i,:], c='C0', lw=0.2, label='Skewer') 
        if i == 0: sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlim(0, 2)
    sub.set_ylabel('peak count score', labelpad=10, fontsize=20) 
    fig.savefig(os.path.join(datdir, 'scores.skewer_cloud.png'), bbox_inches='tight') 
    return None


def makeConvolvedSkewers(seed=1, percent=10.): 
    ''' Take skewers and convolve their parameter values with a 
    multivariate Gaussian in order to reduce the skeweriness. 
    '''
    np.random.seed(seed)
    # read in skewer data (i.e. parameter values and score) 
    datdir = os.path.join(os.environ['MNULFI_DIR'], 'Peaks_MassiveNuS')
    theta_skewers = np.load(os.path.join(datdir, 'theta.skewers.npy')) 
    score_skewers = np.load(os.path.join(datdir, 'scores.skewers.npy')) 

    # now we want to scatter the parameter values 
    theta_lims = [(theta_skewers[:,i].min(), theta_skewers[:,i].max()) for i in range(theta_skewers.shape[1])]
    lower = np.array([theta_lim[0] for theta_lim in theta_lims])
    upper = np.array([theta_lim[1] for theta_lim in theta_lims])
    dtheta = upper - lower
    sig_theta = percent * 0.01 * dtheta # lets start off with gaussian w/ sigma = 1% of the prior limits 

    theta_conv = np.random.multivariate_normal(np.zeros(3), sig_theta**2 * np.eye(3), size=theta_skewers.shape[0])
    theta_conv += theta_skewers 

    theta_conv.dump(os.path.join(datdir, 'theta.convskewers.%i.npy' % percent)) 
    score_skewers.dump(os.path.join(datdir, 'scores.convskewers.%i.npy' % percent)) 
    
    # -- plot thetas --
    theta_cloud = np.load(os.path.join(datdir, 'theta.cloud.npy')) 
    fig = plt.figure(figsize=(8,4))  
    sub = fig.add_subplot(121) 
    sub.scatter(theta_cloud[::100,0], theta_cloud[::100,1], c='C0', s=2, label='Cloud') 
    sub.scatter(theta_conv[::100,0], theta_conv[::100,1], c='C1', s=1) 
    sub.scatter(theta_skewers[::100,0], theta_skewers[::100,1], c='k', s=2) 
    sub.set_xlabel(r'$\theta_1$', fontsize=20) 
    sub.set_xlim(theta_lims[0])
    sub.set_ylabel(r'$\theta_2$', fontsize=20) 
    sub.set_ylim(theta_lims[1])

    sub = fig.add_subplot(122) 
    sub.scatter(theta_cloud[::100,2], theta_cloud[::100,1], c='C0', s=2, label='Cloud') 
    sub.scatter(theta_conv[::100,2], theta_conv[::100,1], c='C1', s=1, label='Conv. Skewer') 
    sub.scatter(theta_skewers[::100,2], theta_skewers[::100,1], c='k', s=2, label='Skewer') 
    sub.legend(loc='lower right', frameon=True, handletextpad=0.1, markerscale=3, fontsize=15) 
    sub.set_xlabel(r'$\theta_3$', fontsize=20) 
    sub.set_xlim(theta_lims[2])
    sub.set_ylim(theta_lims[1])
    fig.savefig(os.path.join(datdir, 'theta.convskewer.%i.png' % percent), bbox_inches='tight') 
    return None 


def skewerscloud_NDE(sampling='skewers'): 
    '''
    '''
    datdir = os.path.join(os.environ['MNULFI_DIR'], 'peaks_massivenus')
    # fiducial theta and scores 
    thetas = np.load(os.path.join(datdir, 'params_conc_means.npy')) # thetas 
    theta_fid = thetas[51,:]
    scores_fid = np.load(os.path.join(datdir, 'scores.fid.npy')) 

    Finv = np.load(os.path.join(datdir, 'peak.Finv.npy')) # inverse fisher
    
    theta_samp  = np.load(os.path.join(datdir, 'theta.%s.npy' % sampling)) 
    scores_samp = np.load(os.path.join(datdir, 'scores.%s.npy' % sampling)) 

    # uniform prior
    theta_lims = [(theta_samp[:,i].min(), theta_samp[:,i].max()) for i in range(theta_samp.shape[1])]
    lower = np.array([theta_lim[0] for theta_lim in theta_lims])
    upper = np.array([theta_lim[1] for theta_lim in theta_lims])
    prior = Priors.Uniform(lower, upper)

    # create an ensemble of ndes
    ndata = scores_samp.shape[1]
    ndes = [NDEs.ConditionalMaskedAutoregressiveFlow(n_parameters=3, n_data=ndata, 
        n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=0)]
    #        NDEs.MixtureDensityNetwork(n_parameters=3, n_data=ndata, n_components=1, 
    #            n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=1),
    #        NDEs.MixtureDensityNetwork(n_parameters=3, n_data=ndata, n_components=2, 
    #            n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=2),
    #        NDEs.MixtureDensityNetwork(n_parameters=3, n_data=ndata, n_components=3, 
    #            n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=3),
    #        NDEs.MixtureDensityNetwork(n_parameters=3, n_data=ndata, n_components=4, 
    #            n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=4),
    #        NDEs.MixtureDensityNetwork(n_parameters=3, n_data=ndata, n_components=5, 
    #            n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=5)]

    # create the delfi object
    DelfiEnsemble = DELFI.Delfi(scores_fid, prior, ndes, 
            Finv=Finv, 
            theta_fiducial=theta_fid, 
            param_limits = [lower, upper],
            param_names = [r'M_\nu', '\Omega_m', 'A_s'],
            results_dir = './',
            input_normalization='fisher') 
    print('loading simulations') 
    DelfiEnsemble.load_simulations(scores_samp, theta_samp) 
    DelfiEnsemble.fisher_pretraining()
    print('training ndes') 
    DelfiEnsemble.train_ndes() 
    print('sampling') 
    posterior_samples = DelfiEnsemble.emcee_sample()
    pickle.dump(posterior_samples, open(os.path.join(datdir, 'posterior.%s.p' % sampling), 'wb'))
    return None 


def plot_skewerscloud_posterior(sampling='skewers'): 
    ''' plot the posterior distribution dumped by skewerscloud_NDE
    '''
    datdir = os.path.join(os.environ['MNULFI_DIR'], 'peaks_massivenus')
    # read in posterior dump 
    post = pickle.load(open(os.path.join(datdir, 'posterior.%s.p' % sampling), 'rb'))

    # fiducial theta and scores 
    thetas = np.load(os.path.join(datdir, 'params_conc_means.npy')) # thetas 
    theta_fid = thetas[51,:]

    theta_samp  = np.load(os.path.join(datdir, 'theta.%s.npy' % sampling)) 
    theta_lims = [(theta_samp[:,i].min(), theta_samp[:,i].max()) for i in range(theta_samp.shape[1])]

    fig = DFM.corner(post, labels=[r'$M_\nu$', '$\Omega_m$', '$A_s$'], 
            quantiles=[0.16, 0.5, 0.84], bins=25, 
            range=theta_lims, truths=theta_fid, truth_color='C1', 
            smooth=True, show_titles=True, label_kwargs={'fontsize': 20}) 
    fig.savefig(os.path.join(datdir, 'posterior.%s.png' % sampling), bbox_inch='tight')
    return None 


if __name__=="__main__":
    #makeSkewerCloud()
    #makeConvolvedSkewers(seed=1, percent=1.)
    #skewerscloud_NDE(sampling='skewers') 
    #skewerscloud_NDE(sampling='cloud') 
    #skewerscloud_NDE(sampling='convskewers.1') 
    #skewerscloud_NDE(sampling='convskewers.5') 
    plot_skewerscloud_posterior(sampling='skewers')
    plot_skewerscloud_posterior(sampling='cloud')
    #plot_skewerscloud_posterior(sampling='convskewers')
    #plot_skewerscloud_posterior(sampling='convskewers.5')
