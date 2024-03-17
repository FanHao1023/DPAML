# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 01:56:34 2021

@author: jacky
"""

# In[1]:
import os
import scipy.io
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from PIL import Image 
from data import load_new_data, data_convert
from keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,concatenate,Cropping2D
from keras.models import Model
import time

from tensorflow import set_random_seed


# In[1]:
pixel=28
    
if os.path.isdir('model'):
    print("model dir exist")
else:
    print("create model dir")  
    os.mkdir('model')
    
if os.path.isfile('data.mat'):
    print ('data.mat already exist')
    #data_convert (new_astype='uint8', new_size=pixel)
else:
    #data_convert (new_astype='uint8', new_size=pixel)
    data_convert (new_size=pixel)

# In[1]:
    
train_data,train_label,test_data,test_label=load_new_data(True) # load data from 'data.mat

print ("train_data.shape:",train_data.shape)
print ("train_label.shape:",train_label.shape)
print ("test_data.shape:",test_data.shape)
print ("test_label.shape:",test_label.shape)

train_data=train_data.reshape(train_data.shape[0],pixel,pixel,1) # 三維 to 四維
test_data=test_data.reshape(test_data.shape[0],pixel,pixel,1)

print ("------------------------------------")
print ("train_data.shape:",train_data.shape)
print ("train_label.shape:",train_label.shape)
print ("test_data.shape:",test_data.shape)
print ("test_label.shape:",test_label.shape)

# In[2]:   Hyperparameter

BATCH_SIZE = 44000 #change there
NUM_EPOCHS = 50
TRAIN_DATA = 0.5
VAL = 0.2

TRAIN_DATA=int (1/TRAIN_DATA)
train_data = train_data [::TRAIN_DATA]
train_label = train_label [::TRAIN_DATA]
print ("train_data.shape:",train_data.shape)
print ("train_label.shape:",train_label.shape)

'''
BATCH_SIZE = 256 #change there
NUM_EPOCHS = 35
TRAIN_DATA = 1 
VAL = 0.2

TRAIN_DATA=int (1/TRAIN_DATA)
train_data = train_data [::TRAIN_DATA]
train_label = train_label [::TRAIN_DATA]
print ("train_data.shape:",train_data.shape)
print ("train_label.shape:",train_label.shape)
'''
# In[3]:
def model_NN(input_shape):
    X_input = Input(input_shape)
    W1 = Flatten()(X_input)
#--------------------------------D1--------------------------------------------    
    D1 = Dense(256, activation='relu', name='fc1')(W1) # Hidden layer 
    # activation can be relu 
    W1 = Dropout(0.5)(W1) 
    #D1 = Dense(64, activation='relu', name='fc2')(D1) # Hidden layer 
#--------------------------------O1--------------------------------------------    
    O1 = Dense(10, activation='softmax', name='out')(D1)    
    
    model_NN = Model(inputs = X_input, outputs = O1 ,name='NN') # spec. input output name
    return model_NN

# In[4]:  training setup
complete_time=time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()) # Record time info.
from keras.callbacks import ModelCheckpoint
if VAL !=0:
    # Training Tips: save best weights during training
    checkpoint = ModelCheckpoint("model/Best_NN_weights_{epoch:02d}_{val_loss:.2f}.h5", 
                                 monitor='val_loss', verbose=0,save_best_only=True, mode='auto', period=1)
    
NN=model_NN((pixel,pixel,1))

NN.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(NN.summary()) 
# In[4]:  training

if VAL !=0:
    NN_train_history=NN.fit(x=train_data,y={"out":train_label},validation_split=VAL, epochs=NUM_EPOCHS,
                       batch_size=BATCH_SIZE,verbose=1, callbacks=[checkpoint])
else:
    NN_train_history=NN.fit(x=train_data,y={"out":train_label},validation_split=VAL, epochs=NUM_EPOCHS,
                       batch_size=BATCH_SIZE,verbose=1)
''' 
NN_train_history=NN.fit(x=train_data,y={"out":train_label},validation_split=VAL, epochs=NUM_EPOCHS,
                       batch_size=BATCH_SIZE,verbose=1, callbacks=[checkpoint])    
'''    
complete_time=time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()) 

# In[5]:  save_weights
NN.save('NN_'+str(complete_time)+'.h5')

# In[6]:  score
# evaluate test data
NN_scores=NN.evaluate(test_data, {"out":test_label}, verbose=1)  
print('Test loss : ',' {:.5f}'.format(NN_scores[0]),' Test accuracy :',' {:.5f}'.format(NN_scores[1]))


# In[6]: plot
import matplotlib.pyplot as plt

def show_train_history(train_history):
    fig=plt.gcf()
    fig.set_size_inches(8, 3)
    plt.subplot(121)
    plt.plot(train_history.history["acc"])
    plt.plot(train_history.history["val_acc"])
    plt.title("Training Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(train_history.history["loss"])
    plt.plot(train_history.history["val_loss"])
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
def show_train_history_NVAL(train_history):
    fig=plt.gcf()
    fig.set_size_inches(8, 3)
    plt.subplot(121)
    plt.plot(train_history.history["acc"])
    plt.title("Training Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train"], loc="upper left")
    plt.subplot(122)
    plt.plot(train_history.history["loss"])
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train"], loc="upper left")
    plt.show()
    
if VAL !=0: show_train_history(NN_train_history) 
else: show_train_history_NVAL(NN_train_history) 

# In[6]: info
print(NN.summary()) 
NN_weights = np.array(NN.get_weights(),dtype=object)
print('weights shape :',NN_weights.shape)
for i in range (0,NN_weights.shape[0],2):
    print('weights 0 shape :',NN_weights[i].shape)
    print('bias 0 shape :',NN_weights[i+1].shape)  
# In[6]: info
print ('\n\n')
print ('#######################################################')
print ('          Convolutional neural network (CNN)           ')
print ('#######################################################')
print ('\n\n')


# In[3]:
def model_CNN(input_shape):
     X_input = Input(input_shape)

 #--------------------------------W1-------------------------------------------- 
     W1 = Conv2D(8, (5, 5), strides = (1, 1), activation = 'relu', name = 'conv1',padding='same')(X_input)
     W1 = MaxPooling2D((2, 2), name='max_pool1_W1')(W1)
     W1 = Conv2D(16, (3, 3), strides = (1, 1), activation = 'relu', name = 'conv2')(W1)

 #--------------------------------W3--------------------------------------------         
     W1 = Flatten()(W1)  
     W1 = Dropout(0.25)(W1)   
 #--------------------------------D1--------------------------------------------
     D1 = Dense(128, activation='relu', name='fc2')(W1)
 #--------------------------------O1--------------------------------------------    
     O1 = Dense(10, activation='softmax', name='out')(D1)    
     
     model_CNN = Model(inputs = X_input, outputs = O1 ,name='NN')
     return model_CNN
    
# In[4]:  training setup
from keras.callbacks import ModelCheckpoint
complete_time=time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()) 
if VAL !=0:
    checkpoint = ModelCheckpoint("model/Best_CNN_weights_{epoch:02d}_{val_loss:.2f}.h5", 
                                 monitor='val_loss', verbose=0,save_best_only=True, mode='auto', period=1)

CNN=model_CNN((pixel,pixel,1))
print(CNN.summary()) 

# In[4]:  training
CNN.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
if VAL !=0:
    CNN_train_history=CNN.fit(x=train_data,y={"out":train_label},validation_split=VAL, epochs=NUM_EPOCHS,
                       batch_size=BATCH_SIZE,verbose=1, callbacks=[checkpoint])
else:
    CNN_train_history=CNN.fit(x=train_data,y={"out":train_label},validation_split=VAL, epochs=NUM_EPOCHS,
                       batch_size=BATCH_SIZE,verbose=1)
'''    
CNN_train_history=CNN.fit(x=train_data,y={"out":train_label},validation_split=VAL, epochs=NUM_EPOCHS,
                       batch_size=BATCH_SIZE,verbose=1, callbacks=[checkpoint])
'''
complete_time=time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()) 
# In[6]: info
print(CNN.summary()) 
CNN_weights = np.array(CNN.get_weights(),dtype=object)
print('weights shape :',CNN_weights.shape)
for i in range (0,CNN_weights.shape[0],2):
    print('weights 0 shape :',CNN_weights[i].shape)
    print('bias 0 shape :',CNN_weights[i+1].shape)  

# In[5]:  save_weights
CNN.save('CNN_'+str(complete_time)+'.h5')

# In[6]:  score
CNN_scores=CNN.evaluate(test_data, {"out":test_label}, verbose=0)  
print('Test loss : ',' {:.5f}'.format(CNN_scores[0]),' Test accuracy :',' {:.5f}'.format(CNN_scores[1]))

if VAL !=0: show_train_history(CNN_train_history) 
else: show_train_history_NVAL(CNN_train_history) 


    
    
# In[6]: Comparison   
print ('\n\n')
print ('#######################################################')
print ('                       SUMMARY                         ')
print ('#######################################################')
print ('\n\n')

print ('/-------------------------------------------------------')
print ('NN.summary():')
print (NN.summary()) 
print ('\n\n')
print ('/-------------------------------------------------------')
print ('CNN.summary():')
print (CNN.summary()) 
print ('\n\n')
print ('/-------------------------------------------------------')
print('NN Test loss : ',' {:.5f}'.format(NN_scores[0]),' NN Test accuracy :',' {:.5f}'.format(NN_scores[1]))
print('CNN Test loss : ',' {:.5f}'.format(CNN_scores[0]),' CNN Test accuracy :',' {:.5f}'.format(CNN_scores[1]))
print ('\n\n')
print ('/-------------------------------------------------------')
def Comparison_train_history(NN,CNN):
    fig=plt.gcf()
    fig.set_size_inches(8, 3)
    plt.subplot(121)
    plt.plot(NN.history["acc"])
    plt.plot(NN.history["val_acc"])
    plt.plot(CNN.history["acc"])
    plt.plot(CNN.history["val_acc"])
    plt.title("Training Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["NN_train", "NN_validation","CNN_train", "CNN_validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(NN.history["loss"])
    plt.plot(NN.history["val_loss"])
    plt.plot(CNN.history["loss"])
    plt.plot(CNN.history["val_loss"])
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["NN_train", "NN_validation","CNN_train", "CNN_validation"], loc="upper left")
    plt.show()
def Comparison_train_history_NVAL(NN,CNN):
    fig=plt.gcf()
    fig.set_size_inches(8, 3)
    plt.subplot(121)
    plt.plot(NN.history["acc"])
    plt.plot(CNN.history["acc"])
    plt.title("Training Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["NN_train","CNN_train"], loc="upper left")
    plt.subplot(122)
    plt.plot(NN.history["loss"])
    plt.plot(CNN.history["loss"])
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["NN_train","CNN_train"], loc="upper left")
    plt.show()

if VAL !=0: Comparison_train_history(NN_train_history,CNN_train_history)
else: Comparison_train_history_NVAL(NN_train_history,CNN_train_history)

