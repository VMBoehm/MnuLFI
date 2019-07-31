'''

validating the GP 


'''
import os 
import numpy as np 
# --- mnulfi --- 
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


def peak_comparison(): 
    thetas  = Data.theta_LHC()
    peaks   = Data.PeakCnts_LHC(average=False, scaled=False) 
    mupeaks = Data.PeakCnts_LHC(average=True, scaled=False) 

    geep    = GeeP.peakEmu()
    
    fig = plt.figure(figsize=(10, 10)) 
    for i in range(3): 
        sub = fig.add_subplot(3,1,i+1) 
        for ii in range(100): 
            sub.plot(range(50), peaks[i,ii,:], c='k', alpha=0.1) 
        sub.plot(range(50), mupeaks[i], c='C0')
        sub.plot(range(50), geep.predict(thetas[i]), c='C1', ls='--') 
        sub.set_xlim(0,50) 
    fig.savefig(os.path.join(os.environ['MNULFI_FIGDIR'], 'valid_gp.peak_comparison.png'), bbox_inches='tight') 
    return None

# feel free to validate the GP in other ways! 


if __name__=="__main__": 
    peak_comparison()
