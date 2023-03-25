import pandas as pd
import numpy as np
import tensorflow as tf
#from tf.keras import regularizers

class ModelBuilder : 
    Model = 0
    
    
    def __init__(self, InputLength):
        self.Model = tf.keras.Sequential([
        tf.keras.layers.Dense(InputLength-1, 
        activation='relu',          #tanh clearly worse
         input_shape=(InputLength-1,),
         #kernel_regularizer=tf.keras.regularizers.l1(0.01),
          #activity_regularizer=tf.keras.regularizers.l2(0.01)
        )
        ])
        
        return
    
    def AddLayers(self, nLayers):
        Rate = 1e-3
        for i in range(nLayers):
            N = int(50 * (1 - i / nLayers ) )
            self.Model.add(tf.keras.layers.Dense(N, 
            activation='relu', 
            kernel_regularizer=tf.keras.regularizers.l2(7e-2)
            #kernel_regularizer=tf.keras.regularizers.l2(Rate),
            #activity_regularizer=tf.keras.regularizers.l2(Rate),
            #bias_regularizer=tf.keras.regularizers.l2(Rate)
            ))
            #self.Model.add(tf.keras.layers.Dropout(0.5)) #Das erhöht overfitting
        
        self.Model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #Output Layer
        return
        
    def CompileModel(self):
        self.Model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.0003))
        return
        
    def GetModel(self):
        return self.Model
