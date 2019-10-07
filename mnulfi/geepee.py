'''

GP emulator of the peak counts


'''
import os
import pickle
import numpy as np
import scipy.stats as stats
# --- mnulfi ---
from . import data as Data
from . import dat_dir
# --- sklearn ---
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class peakEmu(object):
    ''' emulator class object. make it convenient to load in GP
    without having to pass the GP hyper parameters

    To run the emulator simply run the following:
    > from mnulfi import geepee as GeeP
    >
    > theta = np.array([0.1, 0.3, 2.1])
    > gp = GeeP.peakEmu()
    > gp.predict(theta)

    You can simply call `gp.predict` whenever you want to run
    the GP
    '''
    def __init__(self):
        ''' reads in the hyper parameters for the GPs at
        each peak count bins
        '''
        # read in GP emulator hyper parameters
        fgp = os.path.join(dat_dir(), 'GP.peakcnt.p')
        self._hyper = pickle.load(open(fgp, 'rb'))

    def predict(self, _theta):
        ''' emulate peak count at input theta

        parameters
        ----------
        _theta: array
            array specifying (Mnu, Om, As) parameters to predict
            the peak count.

        return
        ------
        preds: array
            peak count at specified (Mnu, Om, As). 50 dim array

        notes
        -----
        * loops through the list of 50 GPs and runs GP.predict
        for each one. Surely there's a better way to this...
        '''
        _scaling, _gp_list = self._hyper # unpack meta data

        preds = np.zeros(50)
        for i, gp in enumerate(_gp_list):
            pred = gp.predict(np.atleast_2d(_theta))
            preds[i] = pred[0]
        preds *= np.array(_scaling)

        return preds.flatten()


def _fit_GP():
    ''' script for fitting the GP. This is copied from Virginia's script.
    read in peak counts data, fit a Gaussian Process emulator for each
    peak count bins, and pickle the scaling factors and emulators.

    notes
    -----
    * updated implementation fits a separate GP for each of the 50 data bins
    '''
    # read in peak counts data
    thetas = Data.theta_LHC() # thetas
    # average peak count at each theta
    peakct = Data.PeakCnts_LHC(average=True, scaled=False)

    # train GPs for each bin
    scaling, gp_list = [], []
    for i in range(peakct.shape[1]):
        peak_bin = peakct[:,i]

        scale = np.mean(peak_bin) # scaling to avoid over-regularization
        scaling.append(scale)

        alphas = np.repeat(stats.sem(peak_bin) / scale , peakct.shape[0]) #scale the standard errors of the mean
        peak_bin /= scale #scale the data

        #define the kernel
        kernel = ConstantKernel(5.0, (1e-4, 1e4)) * RBF([3, 0.3, 5], (1e-4, 1e4))

        #instantiate the gaussian process
        gp = GPR(kernel=kernel, alpha=alphas**2, n_restarts_optimizer=50, normalize_y=True)
        gp.fit(thetas, peak_bin)
        gp_list.append(gp)

    # pickle dump GP to file
    fgp = os.path.join(dat_dir(), 'GP.peakcnt.p')
    pickle.dump([scaling, gp_list], open(fgp, 'wb'))
    return None
