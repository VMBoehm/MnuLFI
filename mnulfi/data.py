''' 


functions with informative names to easily read 
MassiveNus data witout dealing with messy i/o 


'''
import os 
import glob 
import numpy as np 
import scipy.stats as stats
# -- mnufli --
from . import util as UT 


def theta_LHC(): 
    ''' parameter values of the Latin Hypercube 
    
    returns
    -------
    thetas: array 
        101 set of (Mnu, Om, As) parameter values of the Latin HyperCube
    '''
    thetas = np.load(os.path.join(UT.dat_dir(), 'theta_lhc.npy'), allow_pickle=True) # thetas
    return thetas


def PeakCnts_LHC(average=True, scaled=False): 
    ''' average peak count for all the latin hypercube cosmologies 

    parameters
    ----------
    average: bool
        If True, return average peak counts (101 x 50 array). 
        If False, return full set of peak counts (101 x 9999 x 50 array)
    scaled: bool
        If True, LSST scaled peak counts 

    returns
    -------
    peakct: array
        101 peak counts at each of the LHC cosmologies. 
        If average is True 101x50 array; If average is False 101x9999x50 array
    '''
    if average: # average peak counts at each LHC (scaling it doesn't change anything)  
        peakct = np.load(os.path.join(UT.dat_dir(), 'avg_peakcnts_lhc.npy'), allow_pickle=True)
    else: # peak counts of all realizations at each LHC 
        if not scaled: 
            peakct = np.load(os.path.join(UT.dat_dir(), 'peakcnts_lhc.npy'), allow_pickle=True) 
        else: 
            peakct = np.load(os.path.join(UT.dat_dir(), 'peakcnts_lhc.scaled.npy'), allow_pickle=True) 
    return peakct


def PeakCnts_fiducial(): 
    ''' fiducial cosmology peak counts. These peak counts are UNSCALED!
    
    returns 
    -------
    bin_mid: array
        bin_mid 
    peakcnt: array 
        peakcounts of 9999 realizations at the fiducial cosmology. 
        (9999 x 50) array
    '''
    # read in peak counts at fiducial cosmology 
    peaks = np.load(os.path.join(UT.dat_dir(), 'Peaks_KN_s2.00_z1.00_ng40.00_b050.npy'), allow_pickle=True)
    bin_mid = peaks[0,:]
    peakcnt = peaks[2:,:]
    return bin_mid, peakcnt


def covariance(scaled=False): 
    ''' covariance matrix of the peak counts calculated using the 9999 
    realizations at fiducial theta 

    parameters
    ----------
    scaled: bool
        If True, returns LSST scaled covariance matrix. If False, returns
        unscaled covariance matrix.
    
    returns
    -------
    cov: array 
        covariance matrix fo the peak counts. 
        50x50 array 
    
    notes
    -----
    * The LSST scaling factor is 12.25/20000. 
    '''
    cov = np.load(os.path.join(UT.dat_dir(), 'covariance.npy'), allow_pickle=True) # covariance  

    f_sky = 12.25/20000.  
    if scaled: cov *= f_sky  
    return cov 


def _make_data(): 
    ''' script used generate the data files (for posterity) 
    '''
    thetas, peaks = [], [] 
    for fpeak in glob.glob(os.path.join(UT.dat_dir(), 'Om*')): 
        # extract parameter values from file name
        # e.g. Om0.31128_As1.99129_mva0.01524_mvb0.01749_mvc0.05262_h0.70000_Ode0.68679_Peaks_KN_s2.00_z1.00_ng40.00_b050.npy
        fname = os.path.basename(fpeak) 
        Om      = float(fname.split('_')[0][2:])
        As      = float(fname.split('_')[1][2:])
        Mnua    = float(fname.split('_')[2][3:])
        Mnub    = float(fname.split('_')[3][3:])
        Mnuc    = float(fname.split('_')[4][3:])

        theta_i = np.array([Mnua + Mnub + Mnuc, Om, As]) 
        thetas.append(theta_i) 
        # extract peak counts 
        _peak_i = np.load(fpeak)
        peak_i  = _peak_i[2:,:] # first two are bins (9999 x 50 array) 
        peaks.append(peak_i) 
    
    thetas  = np.array(thetas)  # 101 x 3 
    peaks   = np.array(peaks)   # 101 x 9999 x 50  
    # write out the LHC parameters
    thetas.dump(os.path.join(UT.dat_dir(), 'theta_lhc.npy')) 
    # write out the LHC peak counts (full) 
    peaks.dump(os.path.join(UT.dat_dir(), 'peakcnts_lhc.npy')) 
    
    # average peak counts at each LHC cosmology (101 x 50) 
    avg_peak = np.average(peaks, axis=1) 
    avg_peak.dump(os.path.join(UT.dat_dir(), 'avg_peakcnts_lhc.npy'))

    # scaled peak counts 
    LSST_scale=np.sqrt(12.25/20000)
    peaks_scaled = [] 
    for i in range(len(thetas)): 
        peaks_scaled_i = LSST_scale * (peaks[i,:,:] - avg_peak[i]) + avg_peak[i] 
        peaks_scaled.append(peaks_scaled_i) 
    peaks_scaled = np.array(peaks_scaled)
    peaks_scaled.dump(os.path.join(UT.dat_dir(), 'peakcnts_lhc.scaled.npy')) 
    return None
