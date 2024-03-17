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
import predo
from keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,concatenate,Cropping2D
from keras.models import Model
import time
import matplotlib.pyplot as plt
import time

# In[2]:
mode = 'nopre'
origin_X,origin_Y,origin_Y_onehot=predo.load_data()

print ("origin_X shape:",origin_X.shape)
print ("origin_X[0]",origin_X[0])
print ("origin_X[0] shape:",origin_X[0].shape)

if mode == 'nopre': resize_x=predo.resize_img(origin_X,True,False)
elif mode == 'dn1': resize_x=predo.load_data_denoise_train2()
elif mode == 'dn2': resize_x=predo.load_data_denoise_train()
    
print ("resize_x shape:",resize_x.shape)
print ("resize_x[0] shape:",resize_x[0].shape)

img = Image.fromarray(resize_x[1234])
plt.imshow(img)
plt.show()

# In[3]:
BATCH_SIZE = 200
NUM_EPOCHS = 50 #change to 100

#WEIGHTS_FINAL = 'model-cat-final2.h5'

# In[4]: 
train_rate=0.5 #change to 0.9
num_train_data=int(5000*train_rate)

train_x_orig=resize_x.reshape(5000,50,130,-1)[0:num_train_data]
test_x_orig=resize_x.reshape(5000,50,130,-1)[num_train_data:]
    
x_train=train_x_orig.astype('float32')/255
x_test=test_x_orig.astype('float32')/255

print ("x_train shape: "+str(x_train.shape))
print ("x_test shape: "+str(x_test.shape))

y_train_onehot=origin_Y_onehot[0:num_train_data]
y_test_onehot=origin_Y_onehot[num_train_data:]

print ("y_train_onehot shape: "+str(y_train_onehot.shape))
print ("y_test_onehot shape: "+str(y_test_onehot.shape))

# In[6]: training
def model(input_shape):
    X_input = Input(input_shape)
        
#--------------------------------W1--------------------------------------------
    
    W1 = Conv2D(64, (5, 5), strides = (1, 1), activation = 'relu', padding='same', name = 'conv11_W1')(X_input)
    W1 = Conv2D(64, (3, 3), strides = (1, 1), activation = 'relu', name = 'conv12_W1')(W1)
    W1 = MaxPooling2D((2, 2), name='max_pool1_W1')(W1)
    
    W1 = Conv2D(128, (5, 5), strides = (1, 1), activation = 'relu', padding='same', name = 'conv23_W1')(W1)
    W1 = Conv2D(128, (3, 3), strides = (1, 1), activation = 'relu', name = 'conv24_W1')(W1)
    W1 = Conv2D(128, (3, 3), strides = (1, 1), activation = 'relu', name = 'conv25_W1')(W1)
    
    W1 = BatchNormalization(axis = 3, name = 'bn1_W1')(W1)
    W1 = MaxPooling2D((2, 2), name='max_pool2_W1')(W1)
    
    W1 = Conv2D(256, (1, 1), strides = (1, 1), activation = 'relu', padding='same', name = 'conv36_W1')(W1)
    W1 = Conv2D(256, (3, 3), strides = (1, 1), activation = 'relu', name = 'conv37_W1')(W1)   
    W1 = Conv2D(256, (3, 3), strides = (1, 1), activation = 'relu', name = 'conv38_W1')(W1)
    W1 = MaxPooling2D((2, 2), name='max_pool3_W1')(W1)
    
    W1 = Conv2D(512, (3, 3), strides = (1, 1), activation = 'relu', padding='same', name = 'conv49_W1')(W1)
    W1 = Conv2D(512, (1, 1), strides = (1, 1), activation = 'relu', name = 'conv410_W1')(W1)
    W1 = BatchNormalization(axis = 3, name = 'bn2_W1')(W1)
    W1 = MaxPooling2D((2, 2), name='max_pool4_W1')(W1)

    W1 = Flatten()(W1)
    W1 = Dropout(0.5)(W1)
    
    D1 = Dense(128, activation='relu', name='fc11_W1')(W1)
    D1 = Dropout(0.25)(D1)    
    D1 = Dense(128, activation='relu', name='fc12_W1')(D1)    
    D1 = Dense(19, activation='softmax', name='fc13_W1')(D1)
    
    D2 = Dense(128, activation='relu', name='fc11_W2')(W1)
    D2 = Dropout(0.25)(D2)
    D2 = Dense(128, activation='relu', name='fc12_W2')(D2)    
    D2 = Dense(19, activation='softmax', name='fc13_W2')(D2)
    
    D3 = Dense(128, activation='relu', name='fc11_W3')(W1)
    D3 = Dropout(0.25)(D3)
    D3 = Dense(128, activation='relu', name='fc12_W3')(D3)    
    D3 = Dense(19, activation='softmax', name='fc13_W3')(D3)
    
    D4 = Dense(128, activation='relu', name='fc11_W4')(W1)
    D4 = Dropout(0.25)(D4)    
    D4 = Dense(128, activation='relu', name='fc12_W4')(D4)    
    D4 = Dense(19, activation='softmax', name='fc13_W4')(D4)
            
    model = Model(inputs = X_input, outputs = [D1, D2, D3, D4],name='test')
    return model

# In[6]: training
if not os.path.isdir('model_history_save'):
    os.mkdir('model_history_save')
    
complete_time=time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()) 
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

# early_stopping : val_loss 連續不降低時，結束訓練
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1,restore_best_weights=True,mode='auto')
# reduce_lr : val_loss 連續不降低時，減少learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='auto',
                            epsilon=0.0001, cooldown=0, min_lr=0)

checkpoint = ModelCheckpoint("model_history_save/best_model_weights.{epoch:02d}_{val_loss:.2f}.h5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

if mode == 'nopre': test=model((50,130,3))
else: test=model((50,130,1))

test.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=test.fit(x=x_train,y={"fc13_W1":y_train_onehot[:,0,:],"fc13_W2":y_train_onehot[:,1,:],"fc13_W3":y_train_onehot[:,2,:],"fc13_W4":y_train_onehot[:,3,:]},
                        validation_split=0.2, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,verbose=1, callbacks=[early_stopping,checkpoint,reduce_lr])

# In[6]: info
print(test.summary()) 
test_weights = np.array(test.get_weights(),dtype=object)

# weight shape
'''
print('weights shape :',test_weights.shape)
for i in range (0,test_weights.shape[0],2):
    print('weights 0 shape :',test_weights[i].shape)
    print('bias 0 shape :',test_weights[i+1].shape)  
'''

# In[6]: save weights

complete_time=time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()) 

test.save(mode+'_test_'+str(complete_time)+'.h5')
#test.save_weights(mode+'_weights_'+str(complete_time)+'.h5')

#model.save(WEIGHTS_FINAL) 
# In[6]: evaluate
scores=test.evaluate(x_test, {"fc13_W1":y_test_onehot[:,0,:],"fc13_W2":y_test_onehot[:,1,:],"fc13_W3":y_test_onehot[:,2,:],"fc13_W4":y_test_onehot[:,3,:]}, verbose=0)  
print('Test W1 loss:', scores[1],' Test W1 accuracy:', scores[5])
print('Test W2 loss:', scores[2],' Test W2 accuracy:', scores[6])
print('Test W3 loss:', scores[3],' Test W3 accuracy:', scores[7])
print('Test W4 loss:', scores[4],' Test W4 accuracy:', scores[8])

#儲存history
np.save(mode+'_weights_'+str(complete_time)+'_history.npy',train_history.history)


# In[6]: train_history keys
#讀取history
cur_history=np.load(mode+'_weights_'+str(complete_time)+'_history.npy',allow_pickle='TRUE').item()
#cur_history=np.load('nopre_weights_2021_10_19 01_44_22_history.npy',allow_pickle='TRUE').item()
print (cur_history.keys())

# In[6]: plot

def show_train_history(cur_history):
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(cur_history["fc13_W1_acc"])
    plt.plot(cur_history["val_fc13_W1_acc"])
    plt.title("Train History W1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(cur_history["fc13_W1_loss"])
    plt.plot(cur_history["val_fc13_W1_loss"])
    plt.title("Train History W1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig('result/'+mode+"W1.png")
    plt.show()
    
    
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(cur_history["fc13_W2_acc"])
    plt.plot(cur_history["val_fc13_W2_acc"])
    plt.title("Train History W2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(cur_history["fc13_W2_loss"])
    plt.plot(cur_history["val_fc13_W2_loss"])
    plt.title("Train History W2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig('result/'+mode+"W2.png")
    plt.show()
    
    
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(cur_history["fc13_W3_acc"])
    plt.plot(cur_history["val_fc13_W3_acc"])
    plt.title("Train History W3")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(cur_history["fc13_W3_loss"])
    plt.plot(cur_history["val_fc13_W3_loss"])
    plt.title("Train History W3")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig('result/'+mode+"W3.png")
    plt.show()
    
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(cur_history["fc13_W4_acc"])
    plt.plot(cur_history["val_fc13_W4_acc"])
    plt.title("Train History W4")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(cur_history["fc13_W4_loss"])
    plt.plot(cur_history["val_fc13_W4_loss"])
    plt.title("Train History W4")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig('result/'+mode+"W4.png")
    plt.show()
    
    
    
show_train_history(cur_history) 

