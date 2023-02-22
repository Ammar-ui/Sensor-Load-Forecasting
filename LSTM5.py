# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 21:52:38 2022

@author: Ammar
"""

import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import matplotlib.pyplot as plt
import os
# dataset_abalone_train = pd.read_csv(
#     "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
#     names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
#            "Viscera weight", "Shell weight", "Age"])

data_path= "C:\\Users\\Omar Karam\\Data\\For Pred\\"
file_Name_1= data_path+""+"X_Train.csv" #devloper
file_Name_2= data_path+""+"Y_Train.csv"
file_Name_3= data_path+""+"X_Test.csv"
file_Name_4= data_path+""+"Y_Test.csv"
X_train= pd.read_csv(file_Name_1,index_col=False, header= None)
Y_train= pd.read_csv(file_Name_2,index_col=False, header=  None)
X_test= pd.read_csv(file_Name_3,index_col=False, header= None)
Y_test= np.array(pd.read_csv(file_Name_4,index_col=False, header=  None))





# X_train = np.array(X_train)
# Y_train = np.array(Y_train)


#model = tf.keras.Sequential([layers.Dense(336),layers.Dense(48)])
model = tf.keras.Sequential([layers.Dense(336),layers.Dense(500),layers.Dense(1150),layers.Dense(700),layers.Dense(350),layers.Dense(48)])
#MeanSquare error for Measure accuracy of NN
model.compile(loss = tf.losses.MeanSquaredError(),optimizer = tf.optimizers.Adam())
model.fit(X_train, Y_train, epochs=10,batch_size=500)
model.summary()
Y_Pridict=np.array(model.predict(X_test))
#model.save("Sensor.h5")‏
Y_Pridict11=model.evaluate(X_test,Y_test)
#################################################
# data_path="D:\\EE490\\Project\\"
# TestX_File= data_path+""+"Final_Test.csv"    


# if os.path.exists(TestX_File):
#     os.remove(TestX_File)

# df = pd.DataFrame({"Y_test" : Y_test, "Y_Pridict" : Y_Pridict})
# df.to_csv("Final_Test.csv", index=False)

# for d in range(126960):
#       for i in range(47):
#           df = pd.DataFrame({"Y_test" : Y_test[d][i], "Y_Pridict" : Y_Pridict[d][i]})
#           df.to_csv("Final_Test.csv", index=False)
          


# N=1
# t=np.arange(0,N)
# x=np.size(0.02*t)+2*np.random.rand(N)
# df=pd.DataFrame(x)
# index=df.index.values

# plt.plot(index,Y_Pridict)
# plt.plot(index,Y_test)

# plt.axvline(Y_test.insex[0.8],c='r')
# plt.show()    
AVP=0.0
AVT=0.0


for i in range(200) :# per day
      for j in range(48)  :# each day consist 48 reading
          #print("at is Y pridict",Y_Pridict[i][j],"j is ",j)
          AVP=AVP+Y_Pridict[i][j]
          AVT=AVT+Y_test[i][j]
      AVP=AVP/48
      AVT=AVT/48
      #print("the avrage of AVP is a",AVP)
      #print("the avrage of AVT is a",AVT)
      Accuracy=((AVT-AVP)/AVT)*100
      print("The percantage of error for  each sample per day is a ",abs(round(Accuracy))," % in ",i,"per week ")
      AVP=0
      AVT=0
      #Accuracy=0    
      


 
# Accuracy1=((Y_Pridict[1][1]-Y_test[1][1])/Y_Pridict[1][1])*100
# print("The percantage of error for  after  first week  at first Hour is a ",abs(round(Accuracy1)))        
         
         
         
# Accuracy2=((Y_Pridict[30][1]-Y_test[30][1])/Y_Pridict[29][1])*100
# print("The percantage of error for  after week 29   at first Hour is a ",abs(round(Accuracy2)))         
         
         
# Accuracy2=((Y_Pridict[480][1]-Y_test[480][1])/Y_Pridict[480][1])*100
# print("The percantage of error for  after week 500   at first Hour is a ",abs(round(Accuracy2)))   



# Accuracy3=((Y_Pridict[47][500]-Y_test[47][1])/Y_Pridict[47][1])*100
# print("The percantage of error for  after week 1000   at first Hour is a ",abs(round(Accuracy3)))     



   
         