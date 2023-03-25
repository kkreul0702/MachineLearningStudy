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


def SystVariation(SystName):
    Campaign = "MC16a"
    UseOddForTrain = True      # True means test on Even    
    DirectoryName = "Log_MainOutput"
    #DirectoryName = "TESTOutput"
    
    
    
    
    path = "/rdsk/dat23/atlas/kkreul/combination/A++/v02_v34lj_77Btag_NewFullMEM_"+Campaign+"/sgtop_tchan_signal/AtlSgTop_tChannelAnalysis/MemDisc/2Jets/lnu/" + SystName + "/"
    if SystName == "sample_syst":
        path = "/rdsk/dat23/atlas/kkreul/combination/A++/v02_v34lj_77Btag_sampleSysts_"+Campaign+"/sgtop_tchan_signal/AtlSgTop_tChannelAnalysis/MemDisc/2Jets/lnu/nominal/"
    elif SystName == "QCD":
        path = "/rdsk/dat23/atlas/kkreul/combination/A++/v02_v34lj_77Btag_QCD_NewFullMEM_"+Campaign+"/sgtop_tchan_signal/AtlSgTop_tChannelAnalysis/MemDisc/2Jets/lnu/nominal/"
    Reader = DataReader(path)
    
    Campaign = "Even/" + Campaign if UseOddForTrain else "Odd/" + Campaign
    
   
    
    Reader.SetListOfFeatures(ListOfFeatures)
    #Reader.SetDebugLevel(2)
    Reader.process()
    Df = Reader.GetFrame()
    
     
    Preproc = Preprocesser(Df, SystName, DirectoryName, Campaign)
    
    Df = Preproc.AddLeptons()
    Df = Preproc.MakeSignalBool()
    Df = Preproc.ScaleToXSection()
    Df = Preproc.DropNaNAndInf()
    Df = Preproc.LogLikelihood()
    Df = Preproc.DropNaNAndInf()  # this drops all with llh = 0
    
  
    
    X_train, X_test, y_train, y_test = Preproc.TrainTestSplitOddEven(UseOddForTrain)
    
    X_train_scaled, X_test_scaled, Weight_train, Weight_test = Preproc.Scale(X_train, X_test)
    
    if SystName == "nominal":
        epochs = 100
        Builder = ModelBuilder(len(ListOfFeatures))  #We add 2 and remove 2 Features
        Builder.AddLayers(4)
        Builder.CompileModel()
        Model = Builder.GetModel()
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
        class_weights = dict(enumerate(class_weights))
    
        history = Model.fit(x=X_train_scaled, y=y_train, epochs=epochs, class_weight=class_weights,
         shuffle=True, validation_split = 0.1, verbose = 2)
    
        Model.save("/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/"+DirectoryName+"/"+Campaign+"/NominalModel")
    
    else:
    
        Model = keras.models.load_model('/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/'+DirectoryName+'/' +Campaign+'/NominalModel')
    
    Result = 0
    #Mode = "MemDisc"
    Mode = ""
    if Mode == "MemDisc":
        Mem = MemDisc(X_test)
        Result = Mem.calc()
    else:
        Result = Model.predict(X_test_scaled)
    
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
    
    
    Plot = Plotter(Directory  = DirectoryName+"/"+Campaign)
    Plot.MakePlot("NeuralNetOutput", X_test, SystName = SystName)
    #Plot.MakePlot("NeuralNetOutput", X_test, isDens = False,  SystName = SystName)
    #Plot.MakePlot("llh_sChannel2j", X_test, SystName = SystName)
       
    Plot.PlotRocCurve(y_test, Result, Overlay = "/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/Graph.root")
    #plt.show()
    
    if SystName == "nominal":
        Plot.PlotLoss(history)
    
    Writer = DataWriter(X_test, SystName)
    Writer.WriteData(DirectoryName+"/"+Campaign)

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
    

