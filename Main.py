import sys
#from FileName import ClassName
import uproot
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from DataWriter import DataWriter
from DataReader import DataReader
from Preprocesser import Preprocesser
from ModelBuilder import ModelBuilder
from Plotter import Plotter
from MemDisc import MemDisc

def main():
    
    ListOfFeatures = ["llh_sChannel2j", "llh_sChannel3j", "llh_ttbarSL",
    "llh_ttbarDL", "llh_tChannel4FS", "llh_Wbb", "llh_Wcj", "llh_Wjj", "jet_pt", "jet_m","el_pt", "mu_pt", 
    "jet_eta", "el_eta", "mu_eta", "met_met", "eventNumber", "evt_weight"]
    #ListOfFeatures = [  "llh_sChannel2j","jet_pt", "mu_pt"]
    #ListOfFeatures = [ "llh_sChannel2j", "jet_pt", "el_pt"]
    #ListOfFeatures = ["llh_sChannel2j", "llh_sChannel3j", "llh_ttbarSL"]
    Reader = DataReader("/rdsk/dat23/atlas/kkreul/combination/A++/v02_v34lj_77Btag_NewFullMEM_MC16a/sgtop_tchan_signal/AtlSgTop_tChannelAnalysis/MemDisc/2Jets/lnu/nominal/")
    Reader.SetDebugLevel(2)  
    Reader.SetListOfFeatures(ListOfFeatures)
    Reader.process()
    Df = Reader.GetFrame()
    print(Df)
    
    
    
    
    Preproc = Preprocesser(Df)
    #Df = Preproc.ReorderJets() #Not needed
    Df = Preproc.AddLeptons()
    Df = Preproc.MakeSignalBool()
    Df = Preproc.ScaleToXSection()

    print("SPLIT")
    X_train, X_test, y_train, y_test = Preproc.TrainTestSplitOddEven()
    print(X_train)
    
    
    
    
    
    print("SCALE")
    X_train_scaled, X_test_scaled = Preproc.Scale(X_train, X_test)
    print(X_train_scaled)
    
    Builder = ModelBuilder(len(ListOfFeatures)) #We add 3 and remove 2 Features
    Builder.AddLayers(2)
    Builder.CompileModel()
    Model = Builder.GetModel()
    print(Df["IsSignal"].value_counts())
    #Weights = Df["Weight"]
    #print("HELLO")
    #print(Weights)
    
    from sklearn.utils import class_weight


    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
    print(y_train.value_counts())
    print(class_weights)
    class_weights = dict(enumerate(class_weights))
    Result = 0
    Mode = "MemDisc"
    if Mode == "MemDisc":
        Mem = MemDisc(X_test)
        Result = Mem.calc()
    else:
        history = Model.fit(x=X_train_scaled, y=y_train, epochs=1, class_weight=class_weights, shuffle=True)
    
        Model.save("TEST_NominalModel")
        Mode = "MemDisc"
    
        Result = Model.predict(X_test_scaled)
    
    print(Result)
    
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
    
    #print(X_test.value_counts())
    
    #X_test.plot(kind='hist', y="NeuralNetOutput")
    print(X_test)
    print(X_test["Group"].value_counts())
   # Plot = Plotter()
   # Plot.MakePlot("NeuralNetOutput", X_test)
   # Plot.MakePlot("NeuralNetOutput", X_test, isDens = False)
   # Plot.MakePlot("llh_sChannel2j", X_test)
   # #sChanPlot = Plot.MakePlot("sChannel", X_test)
   # 
   # # summarize history for loss
   # ax = plt.axes([0.1, 0.1, 0.8, 0.8], label = "Loss") 
   # plt.plot(history.history['loss'])
   # #plt.plot(history.history['val_loss'])
   # plt.title('model loss')
   # plt.ylabel('loss')
   # plt.xlabel('epoch')
   # plt.legend(['train', 'test'], loc='upper left')
   # plt.savefig("Plots/Loss.png")
    
   # Plot.PlotRocCurve(y_test, Result, Overlay = "Graph.root")
    #plt.show()
    
    print("WRITER")
    Writer = DataWriter(X_test, "nominal")
    Writer.WriteData("TEST_Output")
    
    
    return

"""
Accuracy:  0.8847364901829676
Precision:  0.47186261558784676
Recall:  0.9820509799300892
"""

if __name__ == "__main__": 
    sys.exit(main())
