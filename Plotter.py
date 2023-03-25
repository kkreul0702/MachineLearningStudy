import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
import os
from matplotlib.ticker import MultipleLocator
import uproot

class Plotter:
    Groups = ["sChannel", "tChannel", "ttbar", "Wjets", "WtChannel", "Zjets_Diboson"]
    GroupTitles = ["s-channel", "t-channel", r'$t\bar{t}$', "W+jets", "Wtchannel", "Z+jets"]
    Directory = ""
    
    def __init__(self, Directory = ""):
        self.Directory = Directory
        plt.rcParams['text.usetex'] = True
        plt.rc('legend',fontsize='large')
        #plt.rc('legendheader',fontsize='large')
        plt.style.use('classic')
        return
    
    def MakePlot(self, Name, df, isDens = True, SystName = ""):
        print("PLOT")
        plt.figure()
        ax = plt.axes([0.1, 0.1, 0.8, 0.8], label = Name) 
        Log=False
        
        if Name.__contains__("llh"):
            Bins = np.linspace(0,8,70)
            Log = True
        elif Name.__contains__("NeuralNet"):
            
            Bins = np.linspace(0,1,16)
        
        else:
            Bins = np.linspace(0,1,20)
        
        for Group in self.Groups:
            
            df_plot = df.loc[(df["Group"] == Group), [Name]]
            
            weights = df.loc[(df["Group"] == Group), ["evt_weight"]].to_numpy()
            
            histtype = "step" if isDens else "barstacked"
            df_plot.plot(kind= "hist", figsize=(10,6), ax = ax, 
                histtype = "step", bins = Bins, density = isDens, log=Log, weights=weights)
        
        if Log :
            plt.yscale('log')

        
        if isDens :
            Name += "_normalized"
        legend = ax.legend(self.GroupTitles,loc='upper center',  frameon=False)
        ax.xaxis.set_minor_locator(MultipleLocator(.05))
        ax.yaxis.set_minor_locator(MultipleLocator(.1))
        
        
        plt.ylabel('Fraction of Events', loc="top")
        plt.xlabel('Neural Net Output', loc="right")
        
        ax.text(0.1, 0.95, " $\sqrt{s} = \sqrt{13}$ TeV, 139 $fb^{-1}$ \n signal region, l + 2j ", transform=ax.transAxes, fontsize=14,
        verticalalignment='top')
       
        
        plt.savefig("/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/"+self.Directory+"/Plots/"+ Name +"_"+SystName+ ".png", format="png")
        return
        
    def PlotRocCurve(self, y_test, Result, Overlay = None):
            
        fpr, tpr, _ = roc_curve(y_test, Result)
        ax = plt.axes([0.1, 0.1, 0.8, 0.8], label = "ROC Curve")
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(label = "Neural Net")
        #plt.savefig("RocCurve.pdf")
        #plt.show()
        if (Overlay):

            with uproot.open(Overlay) as File:
                X = File["Graph"].member("fX")[1:-1]
                Y = File["Graph"].member("fY")[1:-1]
                print(X, "X")
                print(Y, "Y")
                
                plt.plot(X, Y, label="MEM Discriminant")
                plt.legend( frameon=False)
                ax.text(0.05, 0.95, " $\sqrt{s} = \sqrt{13}$ TeV, 139 $fb^{-1}$ \n signal region, l + 2j ", transform=ax.transAxes, fontsize=14,
                verticalalignment='top')
        
        #ax.xaxis.set_minor_locator(MultipleLocator(.05))
        #ax.yaxis.set_minor_locator(MultipleLocator(.05))
        
        plt.savefig("/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/"+self.Directory+"/Plots/"+ "RocCurve.png", format="png")  
        return 
    
    def PlotLoss(self, history):
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.8, 0.8], label = "Loss") 
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        
        x = history.history['loss']
        #start, end = ax.get_xlim()
        #ax.xaxis.set_ticks(np.arange(start, end, 1))
        
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(.005))
        ax.text(0.1, 0.95, " $\sqrt{s} = \sqrt{13}$ TeV, 139 $fb^{-1}$ \n signal region, l + 2j ", transform=ax.transAxes, fontsize=14,
        verticalalignment='top')
        plt.legend(['training sample', 'validation sample'], loc='upper right', frameon=False)
        plt.savefig("/users/eeh/kkreul/MachineLearning/Env2/schanmachinelearning/"+self.Directory+"/Plots/Loss.png")
        return
        
