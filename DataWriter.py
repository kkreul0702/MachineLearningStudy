
import numpy as np
import pandas as pd
import uproot

# This is called for each syst and then has to write it, to free up memory
# for the next syst

class DataWriter:
    
    df = 0
    Groups = ["sChannel", "tChannel", "Wjets", "ttbar", "Zjets_Diboson", "WtChannel", "Data"]
    SystName = 0
    Directory = "MainOutput"
    
    def __init__(self, X_test, SystName):
        self.SystName = SystName
        self.df = X_test #if not X_test.empty() else print("ERROR")
        self.Groups = self.df["Group"].drop_duplicates()
        print(self.Groups)
        print(self.SystName)
        
        if self.SystName == "nominal":
            self.SystName = ''
        elif SystName.__contains__("MCWEIGHT") or SystName.__contains__("sample_syst") or SystName.__contains__("QCD"):
            self.SystName = "_" + SystName
        else:
            SystNameList = self.SystName.rsplit("1",maxsplit = 1)
            self.SystName = SystNameList[0] + "1_" + SystNameList[1] # To reuse the TRexFitter ConfigFile
            self.SystName = "_" + self.SystName
        print(self.SystName)
        return
    
    def WriteGroups(self, Output):
        print("writeGroups")
        total = 0
        for Group in self.Groups :
                GroupData = self.df.loc[(self.df["Group"] == Group), ["NeuralNetOutput"]].to_numpy()
                weights = self.df.loc[(self.df["Group"] == Group), ["evt_weight"]].to_numpy()
                print("HERE")
                print(GroupData)
                print(np.shape(GroupData) [0])
                total += np.shape(GroupData) [0]
                bins = np.linspace(0,1,20000)   #6000
                hist = np.histogram(GroupData, bins = bins, weights = weights)
                print(hist)
                
                #from IPython import embed
                #embed()
                #err = np-histogram(GroupData, bins=bins, weights= weights*weights)
                #uproot.writing.identify.to_TH1x thats the solution
                #https://uproot.readthedocs.io/en/latest/uproot.writing.identify.to_TH1x.html
                print("Saving to Hist " + "NeuralNetOutput_" + Group + self.SystName)
                #Output["NeuralNetOutput_" + Group + self.SystName] = np.histogram(GroupData, bins = bins, weights = weights)
                
                #Sum of Squared Weights
                bins_withOverUnder = np.append(bins,100)
                bins_withOverUnder = np.insert(bins_withOverUnder,0,-100)
                hist2, Binning = np.histogram(GroupData, bins = bins_withOverUnder, weights = weights)
                
                axis = uproot.writing.identify.to_TAxis("xaxis","",len(bins),0,1)
                weightsSquare, Binning = np.histogram(GroupData, bins=bins_withOverUnder, weights=weights*weights)
                
                #Scale to XSection
                #hist2 *= 0.000187878
                #weightsSquare *= 0.000187878*0.000187878
                
                Histo2 = uproot.writing.identify.to_TH1x("Name","Title",hist2,
                    len(weights),np.sum(weights),np.sum(weights*weights),np.sum(weights)
                    ,np.sum(weights*weights),weightsSquare.astype(float),axis )
                Output["NeuralNetOutput_" + Group + self.SystName] = Histo2
                
                #from IPython import embed
                #embed()
                
                
                
                
                
                
            #ResultHist = np.histogram(Result, bins = 18)
            #Output["NeuralNetOutput"] = ResultHist
        
        print("TOtal", total)  
        return
        
    def WriteData(self, FileName):
        Name =( "/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/"+FileName+"/NeuralNetOutput/"
            +"NeuralNetOutput_"+self.SystName + ".root" ) 
        print("Writing to File " +  Name) 
        with uproot.recreate(Name) as Output:
            self.WriteGroups(Output)
        
        """
        # When submitting: Multiple Jobs write to Output.root --> errror
        if self.SystName == "nominal":
            with uproot.recreate(FileName) as Output:
                self.WriteGroups(Output)
        else:
            with uproot.update(FileName) as Output:
                self.WriteGroups(Output)
        """   
        return
    
    """
    uproot.writing.identify.to_TH1x("Name","Title",hist[0],len(weights),np.s
   ...: um(weights),np.sum(weights*weights),np.sum(weights),np.sum(weights*weigh
   ...: ts),np.sum(weights*weights),bins  )
   
   #axis: axis = uproot.writing.identify.to_TAxis("xaxis","",len(bins),0,1)
   
   #working: but errors must be sum(weights squared)
   uproot.writing.identify.to_TH1x("Name","Title",hist[0],len(weights),np.
    ...: sum(weights),np.sum(weights*weights),np.sum(weights),np.sum(weights*wei
    ...: ghts),(weights.astype(float)*weights).flatten(),axis  )
    #Sum suqraed weights
    Histo = uproot.writing.identify.to_TH1x("Name","Title",hist[0],len(weig
    ...: hts),np.sum(weights),np.sum(weights*weights),np.sum(weights),np.sum(wei
    ...: ghts*weights),weightsSquare[0].astype(float),axis  )
    
    Histo2 = uproot.writing.identify.to_TH1x("Name","Title",hist2[0],len(w
     ...: eights),np.sum(weights),np.sum(weights*weights),np.sum(weights),np.sum
     ...: (weights*weights),weightsSquare[0].astype(float),axis ) #Correct sum of values
     hist2 = np.histogram(GroupData, bins = bins2, weights = weights)
     bins2 = np.linspace(-0.11111,1.111,12)
     bins = np.linspace(0,1,10)
     weightsSquare = np.histogram(GroupData, bins=bins2, weights=weights*weights) 
    """
