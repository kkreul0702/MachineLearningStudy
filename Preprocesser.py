import pandas as pd
import numpy as np
import time
import copy
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

class Preprocesser:
    Df = pd.DataFrame()
    SystName = 0
    Campaign = ""
    DirectoryName = ""
    
    def __init__(self, Df, SystName, DirectoryName, Campaign):
        self.Df = Df
        self.SystName = SystName
        self.Campaign = Campaign
        self.DirectoryName = DirectoryName
        return
    
    def ReorderJets(self):
        """
        Not needed anymore and slow
        """
        start = time.time()
        self.Df["SubLead_jet_pt"] = np.zeros(len(self.Df))
       
        Copy = self.Df.copy() #Pandas Doc:"Indexing"--> "Returning a view versus a copy"
        for index in self.Df.index:
            if index[1] == 1:
                Copy.loc[prevIndex,"SubLead_jet_pt"] = Copy.loc[index]["jet_pt"]
                Copy.loc[prevIndex,"SubLead_jet_eta"] = Copy.loc[index]["jet_eta"]
                Copy.loc[prevIndex,"SubLead_jet_m"] = Copy.loc[index]["jet_m"]
                Copy.drop(index, inplace=True)
                
            prevIndex = index
        self.Df = Copy
       
        end = time.time()
        Timer = end -start
        print("Time: " ,Timer)
        return self.Df
    
    
        
    
    def AddLeptons(self):
        self.Df= self.Df.replace({np.nan:0}) 
        
        Result_pt = self.Df["mu_pt"] + self.Df["el_pt"]      
        self.Df["lep_pt"] = Result_pt   
        self.Df.drop(["el_pt","mu_pt"], axis = 1, inplace = True) 
        
        Result_eta = self.Df["mu_eta"] + self.Df["el_eta"]      
        self.Df["lep_eta"] = Result_eta  
        self.Df.drop(["el_eta","mu_eta"], axis = 1, inplace = True) 
        
        return self.Df

    def MakeSignalBool(self):
        IsSignal = [ 1 if x.__contains__("410644") or  x.__contains__("410645") else 0 for x in self.Df["SampleName"] ]
        
        self.Df["IsSignal"] = IsSignal
        #self.Df.drop("SampleName", axis = 1, inplace=True) #Need this for Grouping of samples
        
        return self.Df

    def TrainTestSplitRandom(self, Fraction = 0.5):
        
        X = self.Df.drop('IsSignal', axis=1)
        y = self.Df['IsSignal']
        
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Fraction, random_state=1
        )
             
        return X_train, X_test, y_train, y_test
    
    def TrainTestSplitOddEven(self, UseOddForTrain):
        print("TRAINTEST")
        
        pt = self.Df.loc[(self.Df["Group"] == "ttbar"), ["llh_sChannel2j"]].to_numpy()
        #from IPython import embed
        #embed()
        
        self.Df = self.Df.sample(frac=1)
        if UseOddForTrain :
            Test = self.Df.loc[(self.Df["eventNumber"] % 2 == 0)]        
            Train = self.Df.loc[(self.Df["eventNumber"] % 2 != 0)]
        else:
            Train = self.Df.loc[(self.Df["eventNumber"] % 2 == 0)]        
            Test = self.Df.loc[(self.Df["eventNumber"] % 2 != 0)]
            
        X_train = Train.drop('IsSignal', axis=1)
        y_train = Train['IsSignal']
               
        X_test = Test.drop('IsSignal', axis=1)
        y_test = Test['IsSignal']
         
        return X_train, X_test, y_train, y_test
        
    def TrainTestSplitOddEvenMulti(self):
        
        Train = self.Df.loc[(self.Df["eventNumber"] % 2 == 0)]        
        Test = self.Df.loc[(self.Df["eventNumber"] % 2 != 0)]
        
   
        
        X_train = Train.drop(["SmallBkg", "Wjets", "WtChannel", "Zjets_Diboson", "sChannel", "tChannel", "ttbar"], axis=1)
        y_train = Train[["SmallBkg", "Wjets", "WtChannel", "Zjets_Diboson", "sChannel", "tChannel", "ttbar"]]
               
        X_test = Test.drop(["SmallBkg", "Wjets", "WtChannel", "Zjets_Diboson", "sChannel", "tChannel", "ttbar"], axis=1)
        y_test = Test[["SmallBkg", "Wjets", "WtChannel", "Zjets_Diboson", "sChannel", "tChannel", "ttbar"]]
        
        return X_train, X_test, y_train, y_test
        
    def ScaleToXSection(self,  Campaign, isData):
        """
        Df["Norm_evt_weight"] = Df["evt_weight"]
        Df.drop("evt_weight", axis = 1, inplace = True)
        Df.drop("LumiMC", axis = 1, inplace = True)
        return self.Df
        """
        print("XSECTION")
        Weight = self.Df["evt_weight"]
        LumiMC = self.Df["LumiMC"]
        
        if Campaign == "MC16a":
            Lumi = 36207.66
        elif Campaign == "MC16d":  
            Lumi = 44307.4
        elif Campaign == "MC16e":  
            Lumi = 58450.1
        else:
            print("ERROR: NO LUMI")
            return

        if isData:
            Lumi = 1
            
        Normalized = Weight * Lumi / LumiMC
        print("NORMALIZE", Normalized, LumiMC, Lumi)
        #self.Df["Norm_evt_weight"] = Normalized
        self.Df["evt_weight"] = Normalized
        #self.Df.drop("evt_weight", axis = 1, inplace = True)
        self.Df.drop("LumiMC", axis = 1, inplace = True)
        print(Normalized)
        
        return self.Df
    
    def DropNaNAndInf(self):
        self.Df.replace([np.inf, - np.inf], np.nan, inplace = True) #replace inf with NaN
        self.Df.dropna(inplace=True)
        return self.Df
    
    def LogLikelihood(self):
        for column in self.Df.columns:
            if column.__contains__("llh"):
                self.Df[column] = np.log(self.Df[column])
        return self.Df
    
    def Scale(self, X_train_full, X_test_full):
        X_train = copy.deepcopy(X_train_full.drop("SampleName", axis = 1) )# Not needed in training/test data
        X_test = copy.deepcopy(X_test_full.drop("SampleName", axis = 1) )
        GroupNames = copy.deepcopy(X_train["Group"])
       
        
        
        if 'Group' in X_train.columns: 
            X_train.drop("Group", axis = 1, inplace = True)
            X_test.drop("Group", axis = 1, inplace = True)
            
        
        if 'evt_weight' in X_train.columns: 
            #Should be False
            Weight_train = X_train.drop("evt_weight", axis = 1, inplace = True)
            Weight_test = X_test.drop("evt_weight", axis = 1, inplace = True)
        
        if 'Norm_evt_weight' in X_train.columns: 
            print("HERE")
            Weight_train = X_train.drop("Norm_evt_weight", axis = 1, inplace = True)
            Weight_test = X_test.drop("Norm_evt_weight", axis = 1, inplace = True)
        
        if 'eventNumber' in X_train.columns: 
            print("HERE")
            X_train.drop("eventNumber", axis = 1, inplace = True)
            X_test.drop("eventNumber", axis = 1, inplace = True)
        
        if 'LumiMC' in X_train.columns: 
            print("HERE")
            X_train.drop("LumiMC", axis = 1, inplace = True)
            X_test.drop("LumiMC", axis = 1, inplace = True)
            
             
        
        
        scaler = StandardScaler() 
       
        #scaler = MinMaxScaler() #RobustScaler prob better: visible at ZJets, but prob overall
        #scaler = RobustScaler() #RobustScaler prob better: visible at ZJets, but prob overall
        if self.SystName == "nominal":
            print("SCALERNOMINAL")
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            Mean = scaler.mean_
            Var = scaler.var_
            Scale = scaler.scale_
            np.save("/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/"+self.DirectoryName+"/"+self.Campaign+"/Mean.npy", Mean)
            np.save("/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/"+self.DirectoryName+"/"+self.Campaign+"/Var.npy", Var)
            np.save("/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/"+self.DirectoryName+"/"+self.Campaign+"/Scale.npy", Scale)
            
        else: 
            Mean = np.load("/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/"+self.DirectoryName+"/"+self.Campaign+"/Mean.npy")
            Var = np.load("/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/"+self.DirectoryName+"/"+self.Campaign+"/Var.npy")
            Scale = np.load("/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/"+self.DirectoryName+"/"+self.Campaign+"/Scale.npy")
            
            scaler.mean_ = Mean
            scaler.var_ = Var
            scaler.scale_ = Scale
            
  
            
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
        
        #X_train_scaled = X_train.to_numpy()
        #X_test_scaled = X_test.to_numpy()
        
        #X_train_scaled = scaler.fit_transform(X_train)
        #X_test_scaled = scaler.transform(X_test)
        
        
        return X_train_scaled, X_test_scaled, Weight_train, Weight_test
    
    def NormalizeWeight(self):
        #Not used
        ListOfGroups = self.Df["Group"]
        ListOfGroups.drop_duplicates(inplace = True)
        
        print("NORMALIZE")
        Copy = self.Df
        Copy["Weight"] = np.zeros(len(Copy))
        
        for Group in ListOfGroups:
            
            df_slice = self.Df.loc[(self.Df["Group"] == Group)]
            df_slice["Weight"] = 1./len(df_slice)
            Copy.loc[(Copy["Group"] == Group), ["Weight"]] = df_slice["Weight"]
        
        print(self.Df)
        
        return
    
    def MultiClass(self):
        print("MULT")
        df = self.Df 
        from sklearn.preprocessing import OneHotEncoder

        #creating instance of one-hot-encoder
        encoder = OneHotEncoder(handle_unknown='ignore')

        #perform one-hot encoding on 'team' column 
        encoder_df = pd.DataFrame(encoder.fit_transform(df[['Group']]).toarray())
        encoder_df.columns = ["SmallBkg", "Wjets", "WtChannel", "Zjets_Diboson", "sChannel", "tChannel", "ttbar"]
        #   merge one-hot encoded columns back with original DataFrame
        final_df = df.join(encoder_df)
        
        
        self.Df = final_df
        print(final_df)
        """
        print("MULT")
        encoder = LabelEncoder()
        Y = self.Df["Group"]
        print(Y.to_numpy())
        encoder.fit(Y)
        encoded_Y = encoder.transform(self.Df["Group"])
        print(encoded_Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)
        print(self.Df["Group"])
        print(dummy_y)
        """
        return final_df
        
    
