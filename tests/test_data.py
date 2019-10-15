__all__ = ['test_data'] 

import pytest
# --- mnulfi --- 
import mnulfi.data as Data

#@pytest.mark.parametrize("noise", ('none', 'bgs1', 'bgs2'))
def test_data(): 
    # make data 
    Data._make_data() 

    # theta values of the LHC 
    thetas = Data.theta_LHC() 
    assert thetas.shape[1] == 3
    
    # average peak counts of the LHC
    peak_cnts = Data.PeakCnts_LHC(average=True) 
    assert peak_cnts.shape[1] == 50

    # full peak counts of the LHC
    peak_cnts_full = Data.PeakCnts_LHC(average=False) 
    assert len(peak_cnts_full.shape) == 3
