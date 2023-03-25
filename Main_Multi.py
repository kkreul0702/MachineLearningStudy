#!/mnt/eeh/kkreul/MachineLearning/Env2/bin/python

# usage:first: python Main.py nominal
#       then: python Main.py SystName

import sys
#from FileName import ClassName
import uproot
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('agg') # fixes Qt error in condor submission

import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow import keras

from DataWriter import DataWriter
from DataReader import DataReader
from Preprocesser import Preprocesser
from ModelBuilder import ModelBuilder
from Plotter import Plotter
from MemDisc import MemDisc


ListOfFeatures = ["llh_sChannel2j", "llh_sChannel3j", "llh_ttbarSL",
    "llh_ttbarDL", "llh_tChannel4FS", "llh_Wbb", "llh_Wcj", "llh_Wjj", "jet_pt", "jet_m","el_pt", "mu_pt", 
    "jet_eta", "el_eta", "mu_eta", "met_met", "eventNumber", "evt_weight"]
    
"""
ListOfFeatures = ["llh_sChannel2j", "llh_sChannel3j", "llh_ttbarSL",
    "llh_ttbarDL", "llh_tChannel4FS", "llh_Wbb", "llh_Wcj", "llh_Wjj", "eventNumber", "evt_weight"]
    


ListOfFeatures = ["jet_pt", "jet_m","el_pt", "mu_pt", 
    "jet_eta", "el_eta", "mu_eta", "met_met", "eventNumber", "evt_weight"]
"""

def SystVariation(SystName):
    ReadNtuples = False
    
    path = "/rdsk/dat23/atlas/kkreul/combination/A++/v02_v34lj_77Btag_NewFullMEM_MC16a/sgtop_tchan_signal/AtlSgTop_tChannelAnalysis/MemDisc/2Jets/lnu/" + SystName + "/"
    if SystName == "sample_syst":
        path = "/rdsk/dat23/atlas/kkreul/combination/A++/v02_v34lj_77Btag_sampleSysts_MC16a/sgtop_tchan_signal/AtlSgTop_tChannelAnalysis/MemDisc/2Jets/lnu/nominal/"
    elif SystName == "QCD":
        path = "/rdsk/dat23/atlas/kkreul/combination/A++/v02_v34lj_77Btag_QCD_NewFullMEM_MC16a/sgtop_tchan_signal/AtlSgTop_tChannelAnalysis/MemDisc/2Jets/lnu/nominal/"
    
    if (ReadNtuples):
        Reader = DataReader(path)
    

    
        Reader.SetListOfFeatures(ListOfFeatures)
        Reader.SetDebugLevel(2)
        Reader.process()
        Df = Reader.GetFrame()
    
        Df.to_pickle("my_data.pkl")
    else:
        Df = pd.read_pickle("my_data.pkl")

    
    Preproc = Preprocesser(Df)
    
    Df = Preproc.AddLeptons()
    Df = Preproc.MultiClass()
    Df = Preproc.ScaleToXSection()
    
    
    #Add MemDisc to Input
    Mem = MemDisc(Df)
    MemDiscResult = Mem.calc()
    Df["MemDisc"] = MemDiscResult
    
    X_train, X_test, y_train, y_test = Preproc.TrainTestSplitOddEvenMulti()
    
    X_train_scaled, X_test_scaled, Weight_train, Weight_test = Preproc.Scale(X_train, X_test)
    

    
    
    if SystName == "nominal":
        epochs = 2
        Builder = ModelBuilder(len(ListOfFeatures)+1)  #We add 2 and remove 2 Features
        Builder.AddLayers(1)
        Builder.CompileModel()
        Model = Builder.GetModel()
        #class_weights = class_weight.compute_class_weight('balanced',
        #                                         np.unique(y_train),
        #                                         y_train)
        #class_weights = dict(enumerate(class_weights))
        from IPython import embed

        embed()
        history = Model.fit(x=X_train_scaled, y=y_train.to_numpy(), epochs=epochs, 
            #class_weight=class_weights, 
            #sample_weight= Weight_train,
            shuffle=True)
    
        Model.save("/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/NominalModel")
    
    else:
    
        Model = keras.models.load_model('/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/NominalModel')
    
    Result = 0
    #Mode = "MemDisc"
    Mode = ""
    if Mode == "MemDisc":
        Mem = MemDisc(X_test)
        Result = Mem.calc()
    else:
        Result = Model.predict(X_test_scaled)
    
    from IPython import embed

    embed()
    X_test[["SmallBkg", "Wjets", "WtChannel", "Zjets_Diboson", "sChannel",
        "tChannel", "ttbar"]] = Result

    Plot = Plotter()
    Plot.MakePlot("sChannel", X_test, SystName = SystName)
    Plot.MakePlot("ttbar", X_test, isDens = False,  SystName = SystName)
    
    """
    X_test["NeuralNetOutput"] = Result
    predic = [ 1 if x>0.5 else 0 for x in Result]   
    
    confusion_matrix = np.zeros((2,2))
    
    for i in np.arange(len(predic)):
        if predic[i] == y_test.iloc[i] and predic[i] == 1:
            confusion_matrix[0,0] += 1 #True Positiv
        if predic[i] == y_test.iloc[i] and predic[i] == 0: 
            confusion_matrix[1,1] += 1 #True negative
        if predic[i] != y_test.iloc[i] and predic[i] == 1: 
            confusion_matrix[1,0] += 1 #False positive
        if predic[i] != y_test.iloc[i] and predic[i] == 0: 
            confusion_matrix[0,1] += 1 #False negative
    
    print(confusion_matrix)
    acc = (confusion_matrix[0,0] + confusion_matrix[1,1]) / confusion_matrix.sum()
    prec = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[1,0]) 
    recall = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1])
    print("Accuracy: " , acc)
    print("Precision: " , prec)
    print("Recall: " , recall)
    
    
    Plot = Plotter()
    Plot.MakePlot("NeuralNetOutput", X_test, SystName = SystName)
    Plot.MakePlot("NeuralNetOutput", X_test, isDens = False,  SystName = SystName)
    #Plot.MakePlot("llh_sChannel2j", X_test, SystName = SystName)
       
    Plot.PlotRocCurve(y_test, Result, Overlay = "/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/Graph.root")
    #plt.show()
    
    if SystName == "nominal":
        Plot.PlotLoss(history)
    
    Writer = DataWriter(X_test, SystName)
    Writer.WriteData("Test_Architecture")

    from IPython import embed

    embed()
    
    #Why is the sChannel Peak at 0.8 and what are the events above that ?:
    X_test["Scaled_llh_sChannel"]=X_test_scaled[:,0]
    dfScale = X_test.loc[(X_test["NeuralNetOutput"] > 0.9) & (X_test["Group"] == "sChannel"),["Scaled"]]
    
    DF2 = pd.DataFrame(X_test_scaled)
    DFNEU = X_test.reset_index()
    DFNEU.drop("Number", inplace = True, axis=1)

    pd.concat([DFNEU,DF2],axis=1)
    
    DFCONC.loc[(DFCONC["NeuralNetOutput"] > 0.9) & (DFCONC["Group"] == "sChannel"),[14]]
    """
    return


if __name__ == "__main__": 
    
    print(sys.argv)
    Syst = sys.argv[1]
    print("Running for Syst: " + Syst)
    sys.exit(SystVariation(Syst))
    #if Syst == "nominal":
    #    sys.exit(nominal())
    #else: 
    #    sys.exit(SystVariation(Syst))
    


