import pandas as pd
import uproot
import os
import re
import numpy as np
import awkward as ak

class DataReader:
    Frame = pd.DataFrame()
    path = ""
    DebugLevel = 0
    DoData = False
    ListOfFeature = []
    XSectionFile = 0
    isSampleSyst = False
    Campaign = ""
    
    def GetTotalEvts(self,File):
        
        job_info = File["job_info"]
        
        HistEvents = job_info["h_nevts"]
        
        NEvents = HistEvents.values()[2]
        
        return NEvents
    
    
    def FindGroup(self, SampleName):
        if self.isSampleSyst :
            return SampleName
            
        if(SampleName.__contains__("410644") or  SampleName.__contains__("410645")):
            return "sChannel"
            
        elif(SampleName.__contains__("410470")):
            return "ttbar"
            
        elif(SampleName.__contains__("410658") or  SampleName.__contains__("410659")):
            return "tChannel"
        
        elif(SampleName.__contains__("410646") or  SampleName.__contains__("410647")):
            return "WtChannel"
            
        elif(SampleName.__contains__("3633") or  SampleName.__contains__("3642")):
            return "Zjets_Diboson"
               
        elif(SampleName.__contains__("Z")):
            return "Zjets_Diboson"
        
        elif(SampleName.__contains__("34634") or SampleName.__contains__("410219")):
            return "SmallBkg"      
        
        elif(SampleName.__contains__("W")):
            return "Wjets"
        
        elif SampleName.__contains__("QCDANTIMUON"):
            if SampleName.__contains__("ShapeSyst1"):
                return "QCDANTIMUONShapeSyst1"
            elif SampleName.__contains__("ShapeSyst2"):
                return "QCDANTIMUONShapeSyst2"
            else:
                return "QCDANTIMUON"
        elif(SampleName.__contains__("data")):
            return "Data"
        
        else : 
            return SampleName   #Needed for Sample Syst
            #return "SmallBkg"
    
    
    def __init__(self,path, DoData, Campaign):
        self.path = path   
        self.DoData = DoData  
        self.Campaign = Campaign
        self.XSectionFile = pd.read_csv("/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/dev/AnalysisTop/TopDataPreparation/XSection-MC15-13TeV.data", sep= '\s+', skip_blank_lines = True, comment='#', index_col=0, header=None)  
        print(self.XSectionFile)
        if path.__contains__("sampleSysts"):
            self.isSampleSyst = True
        return
        
    def GetFrame(self):
        #if self.DebugLevel > 0 : print(self.Frame.head())
        return self.Frame
    
    def SetDebugLevel(self, Level):
        self.DebugLevel = Level
        return
        
    def SetListOfFeatures(self, List):
        self.ListOfFeature = List
        self.Frame = pd.DataFrame(columns=List)
        return
    
    def process(self):
        
        Storage = []
       
        ListOfSamples = os.listdir(self.path)
        
        for Sample in ListOfSamples:   # ~36 is schanl
            if not Sample.startswith("ntup"): continue
            
            
            print(Sample)
            if  Sample.__contains__("data") and not self.DoData : # This causes Problem with JER Pseudodata
                print("data")
                continue
            else:
                if not Sample.__contains__("data") and self.DoData:
                    continue
                print("ELSE")
                #continue    
            
            if self.DebugLevel > 0 : print (Sample)
            with uproot.open(self.path+Sample) as File:
                
                if  not 'physics;1' in File.keys():
                    print("Bad Sample: " + Sample)
                    continue
                Physics = File["physics"]
                
                if Sample.__contains__("mll"): # These have a "_" in their name
                    SampleName = re.split(r'[_.]', Sample[::-1], maxsplit = 6)[2]  #These 2 lines = rsplit()
                    SampleName = SampleName[::-1]
                else: 
                    SampleName = re.split(r'[_.]', Sample[::-1], maxsplit = 6)[1]  #These 2 lines = rsplit()
                    SampleName = SampleName[::-1]                               
                
                
                
                print(SampleName)
                """
                #Jet_pt is stored with subentry here, Time consuming to reorder that
                Df = Physics.arrays(self.ListOfFeature, library="pd")
               
                if type(Df) is tuple : Df = pd.concat(Df)
                Df["SampleName"] = SampleName
                Df.reset_index(level = [0,1], inplace=True)
                
                Storage.append(Df) 
                 """     
                
                #This version not so elegant here, but no need to reorder stuff
                #that means faster
                Dict = {}
                length =0
                for Feature in self.ListOfFeature:
                    Data = Physics[Feature].array()
                   
                    length = len(Data)
                    if length == 0:
                        print("WARNING this sample has no entries")
                        break
                                   
                    if Feature.__contains__("jet") :
                        #Working for jet_
                        Numpy = ak.to_numpy(Data)
                        
                        JetLead = Numpy.T[0]
                        SubLead = Numpy.T[1] 
                        Dict.update({"Lead_"+Feature:JetLead, "SubLead_"+Feature: SubLead})
                        
                    elif Feature.__contains__("el_") or Feature.__contains__("mu_"):
                        
                        Numpy = np.array(ak.to_list(Data), dtype="O")
                        
                        good = [ False if np.size(x) == 0 else True for x in Numpy]
                        
                        Masked = ak.mask(Data,good)
                        MaskedNumpy = ak.to_numpy(Masked)
                                             
                        if np.size(MaskedNumpy) == 0: #If there are ONLY mu or e then the to_numpy() is not working as expected -.-
                            MaskedNumpy = np.zeros(( len(Masked),1 ))*np.nan
                            
                        Dict.update({Feature:MaskedNumpy[:,0]})
                        

                    #if not np.isscalar(Data[0]) : # eg. jet_pt is array of size 2  # Old version of Jet_pt reordering with numpy array
                        
                        #Reshaped = np.hstack(Data).reshape(int(len(np.hstack(Data))/2),2)
                        #Dict.update({"Lead_"+Feature: Reshaped[:,0], "SubLead_"+Feature: Reshaped[:,1]})
                        
                    else:
                        Dict.update({Feature : Data})
                    
                     
                Df = pd.DataFrame(Dict, index = np.arange(0,length))
                Df["SampleName"] = SampleName
                # in the end: nEvents / xSection
                evt_weight = Df.get("evt_weight")
                
                
                NEvents = self.GetTotalEvts(File) # Total events for job_info Tree
                               
                print(SampleName)
                
                if SampleName.__contains__("QCDANTI") :
                    if self.Campaign == "MC16a":
                         Df["LumiMC"] = 7607.96 #Mc16a
                    elif self.Campaign == "MC16d":
                         Df["LumiMC"] = 9309.87
                    elif self.Campaign == "MC16e":
                         Df["LumiMC"] = 12281.5
                    else :
                        print("NO LUMIMC")

                elif SampleName.__contains__("QCDJet") :
                    if self.Campaign == "MC16a":
                        Df["LumiMC"] = 5627.91 #Mc16a
                    elif self.Campaign == "MC16d":
                        Df["LumiMC"] = 6886.88 #MC16d
                    elif self.Campaign == "MC16e":
                        Df["LumiMC"] = 9085.14
                    else :
                        print("NO LUMIMC")
                
                elif SampleName.__contains__("data") :
                    Df["LumiMC"] = 1 #Mc16a
                    print("DATA")
                
                else: 
                    Df["LumiMC"] = ( NEvents / 
                    (self.XSectionFile.loc[int(SampleName[0:6]),1] * 
                    self.XSectionFile.loc[int(SampleName[0:6]),2]) )  # NEvent / (xSection * k Factor)
                
                Df["Group"] = self.FindGroup(SampleName) 
                Storage.append(Df)
                #print(Storage)
                 
                
                       
            
        self.Frame = pd.concat(Storage, ignore_index=True)
        self.Frame.index.name = "Number"
        #self.Frame.reset_index(level = [0], inplace=True)
        
        #self.Frame.set_index(["Number","subentry"], inplace=True) 
        #self.Frame.rename(columns = {'entry':'EventInSample'}, inplace = True)
        
        return
