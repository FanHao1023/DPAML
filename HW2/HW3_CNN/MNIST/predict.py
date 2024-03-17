# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 01:56:34 2021

@author: jacky
"""

"""
Revised on Mon Oct 03 20:36:58 2022

@reviser: peggy
"""

import numpy as np
from data import load_new_data
from keras import models
import time

# In[2]: load model
name='CNN_2024_03_14 22_06_20.h5'
model = models.load_model(name)
print("load ",name)

# In[3]: one hot decoder
def translate_Y(yi , show = False):
    temp=0    
    for i in range(10):          
          if yi[i] >= temp:
              temp=yi[i]
              yi_new=i
              
    return yi_new
# In[4]: load test data
import matplotlib.pyplot as plt

train_data,train_label,test_data,test_label=load_new_data(True)
train_data=train_data.reshape(train_data.shape[0],28,28,1)
test_data=test_data.reshape(test_data.shape[0],28,28,1)

train_data_len=train_data.shape[0]
test_data_len=test_data.shape[0]

# In[5]: predict
accumulate_time=0
index=0
error=0

for i in range(test_data_len):  # 依序於所有test data中

    print ('\n\n-------------------------------------------')
    print ('---------------- index',index,'----------------')
    #print ('train_data shape :',train_data[i].shape)
    input_data=test_data[i]
    input_data=input_data[np.newaxis, :]
    #print ('input_data shape :',input_data.shape)
    start = time.time()
    prediction=model.predict(input_data) 
    end = time.time()
    accumulate_time += (end - start)
    #print (prediction)
    result=translate_Y(prediction[0])
    golden=translate_Y(test_label[i])

    if result!=golden :
        
        error+=1
        print ('predict fail in',i,'image')
        print ('golden :',golden)
        print ('predict :',result)
        print (prediction[0])
        #break
    
    index+=1

# In[6]: result 
num_picture=index
acc= (num_picture-error)/num_picture
print (num_picture,"picture total wrong =",error)
print ("image accuracy = ",acc)

print ('average per execute time: %f ms' %(accumulate_time/test_data_len*1000))




