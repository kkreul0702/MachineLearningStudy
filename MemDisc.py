
import numpy as np
import pandas as pd

ttbar_muPlus = 0.16874 
ttbar_eMinus = 0.17406 
ttbar_ePlus = 0.17752 
ttbar_muMinus = 0.16659 
Wjj_muPlus = 0.00361 
Wjj_eMinus = 0.00331 
Wjj_ePlus = 0.00051 
Wjj_muMinus = 0.00157 
sChannel_muPlus = 0.01198 
sChannel_eMinus = 0.00747 
sChannel_ePlus = 0.01151 
sChannel_muMinus = 0.00827 
tChannel4FS_muPlus = 0.03649 
tChannel4FS_eMinus = 0.02488 
tChannel4FS_ePlus = 0.03529 
tChannel4FS_muMinus = 0.02156 
wHF_muPlus = 0.03879 
wHF_eMinus = 0.03416 
wHF_ePlus = 0.04196 
wHF_muMinus = 0.03172 


sChannel2jVS3jFrac = 0.65  

fP_sChannel2j_ePlus   = sChannel2jVS3jFrac * sChannel_ePlus 
fP_sChannel2j_muPlus  = sChannel2jVS3jFrac * sChannel_muPlus 
fP_sChannel2j_eMinus  = sChannel2jVS3jFrac * sChannel_eMinus 
fP_sChannel2j_muMinus = sChannel2jVS3jFrac * sChannel_muMinus 
      
fP_sChannel3j_ePlus   = (1.0 - sChannel2jVS3jFrac) * sChannel_ePlus 
fP_sChannel3j_muPlus  = (1.0 - sChannel2jVS3jFrac) * sChannel_muPlus 
fP_sChannel3j_eMinus  = (1.0 - sChannel2jVS3jFrac) * sChannel_eMinus 
fP_sChannel3j_muMinus = (1.0 - sChannel2jVS3jFrac) * sChannel_muMinus 

fP_tChannel4FS_ePlus   = 0.0199503 
fP_tChannel4FS_muPlus  = 0.0250097 
fP_tChannel4FS_eMinus  = 0.0128461 
fP_tChannel4FS_muMinus = 0.0151461 

fP_Wjj_ePlus   = 0.00233655 
fP_Wjj_muPlus  = 0.0129941 
fP_Wjj_eMinus  = 0.000406719 
fP_Wjj_muMinus = 0.00516182 

wbbVSwcjFrac = 0.8 

fP_Wbb_ePlus   = wbbVSwcjFrac * 0.0431757 
fP_Wbb_muPlus  = wbbVSwcjFrac * 0.0558853 
fP_Wbb_eMinus  = wbbVSwcjFrac * 0.0335755 
fP_Wbb_muMinus = wbbVSwcjFrac * 0.0409918 

fP_Wcj_ePlus   = (1.0 - wbbVSwcjFrac) * 0.0431757 
fP_Wcj_muPlus  = (1.0 - wbbVSwcjFrac) * 0.0558853 
fP_Wcj_eMinus  = (1.0 - wbbVSwcjFrac) * 0.0335755 
fP_Wcj_muMinus = (1.0 - wbbVSwcjFrac) * 0.0409918 

# v15 fraction before veto
ttbarSL_vs_ttbarDL_frac_el = 0.2361 
ttbarSL_vs_ttbarDL_frac_mu = 0.2398 

fP_ttbarSL_ePlus   = ttbarSL_vs_ttbarDL_frac_el * 0.158366 
fP_ttbarSL_muPlus  = ttbarSL_vs_ttbarDL_frac_mu * 0.190364 
fP_ttbarSL_eMinus  = ttbarSL_vs_ttbarDL_frac_el * 0.159029 
fP_ttbarSL_muMinus = ttbarSL_vs_ttbarDL_frac_mu * 0.189734 

fP_ttbarDL_ePlus   = (1.0 - ttbarSL_vs_ttbarDL_frac_el) * 0.15837 
fP_ttbarDL_muPlus  = (1.0 - ttbarSL_vs_ttbarDL_frac_mu) * 0.19036 
fP_ttbarDL_eMinus  = (1.0 - ttbarSL_vs_ttbarDL_frac_el) * 0.15903 
fP_ttbarDL_muMinus = (1.0 - ttbarSL_vs_ttbarDL_frac_mu) * 0.18973 

class MemDisc():
    
    df = 0
    
    
    def __init__(self, X_test):
        print("MEMDISC")
        self.df = X_test
        return
    
    def calc(self):
        
    
        # temporary solution without splitting mu/el/+/-: use only e+
        p_sChannel2j  = fP_sChannel2j_ePlus 
        p_sChannel3j  = fP_sChannel3j_ePlus 
        p_tChannel4FS = fP_tChannel4FS_ePlus 
        p_Wjj         = fP_Wjj_ePlus 
        p_Wbb         = fP_Wbb_ePlus 
        p_Wcj         = fP_Wcj_ePlus 
        p_ttbarSL     = fP_ttbarSL_ePlus 
        p_ttbarDL     = fP_ttbarDL_ePlus 
        
        
        llh_sChannel2j  = self.df["llh_sChannel2j"]
        llh_sChannel3j  = self.df["llh_sChannel3j"]
        llh_tChannel4FS = self.df["llh_tChannel4FS"]
        llh_ttbarSL     = self.df["llh_ttbarSL"]
        llh_ttbarDL     = self.df["llh_ttbarDL"]
        llh_Wjj         = self.df["llh_Wjj"]
        llh_Wcj         = self.df["llh_Wcj"]
        llh_Wbb         = self.df["llh_Wbb"]
    
        sig = p_sChannel2j * llh_sChannel2j + p_sChannel3j * llh_sChannel3j
        bkg = (p_tChannel4FS * llh_tChannel4FS +
            p_Wjj * llh_Wjj +
            p_Wbb * llh_Wbb +
            p_Wcj * llh_Wcj +
            p_ttbarSL * llh_ttbarSL +
            p_ttbarDL * llh_ttbarDL ) 
       
        
        sChannelRatio = sig / (sig + bkg);
        
        return sChannelRatio
    

