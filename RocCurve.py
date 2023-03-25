import sys
import uproot
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from hist import Hist
import hist
import mplhep

def main():
    with uproot.open("/rdsk/dats1/atlas/kkreul/combination/A++/v02_v32lj_GridRun_NominalOnly_MC16a/sgtop_schan_signal/AtlSgTop_tChannelAnalysis/MemDisc/2Jets/lnu/nominal/SgTop_s_QCDsplit_wjetsMerged/adapted_for_trexfitter_allhist/sgtop_sChannel.root") as File:
        print(File.keys())
        Histo = File["MemDiscriminant/ManyBins/sChannelProb_raw"]
        print(Histo)
        edges = Histo.axis().edges()
        Values = Histo.values()
        print(Values, edges)
        #plt.plot(edges[:-1], Values)
        #plt.show()
        Histogramm = Histo.to_hist()
        Histogramm.plot()
        print(np.spacing(edges))
        plt.show()
        
        
    return


if __name__ == "__main__": 
    sys.exit(main())
