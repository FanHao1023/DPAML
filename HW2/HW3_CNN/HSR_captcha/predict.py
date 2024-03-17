# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 01:56:34 2021

@author: jacky
"""

"""
Revised on Mon Oct 03 21:15:00 2022

@reviser: peggy
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy.io

from PIL import Image 
import predo
from keras import models
import time


# In[7]: predict
def one_hot_decode(data):
    data = np.squeeze(data)
    list_max=data.tolist()
    list_max.sort(key=lambda x: float(x), reverse = True)
    n = np.argwhere(data==list_max[0])
    return int(n), list_max[0]

def translate_Y(yi , show = False):
    
    num = {
        0 : "2", 1 : "3", 2 : "4", 3 : "5", 4 : "7", 
        5 : "9", 6 : "A", 7 : "C", 8 : "F", 9 : "H", 
        10: "K", 11: "M", 12: "N", 13: "P", 14: "Q",
        15: "R", 16: "T", 17: "Y", 18: "Z"  } 
    
    yi_new = num.get( yi ) 
            
    if show:
        print(yi)
        print(yi_new)            
        
    return yi_new

# In[7]: setup model
model_name = 'nopre_test_2022_10_04 18_29_27.h5'
test=models.load_model(model_name)
print(test.summary()) 

# In[7]: load test data
data = scipy.io.loadmat('database/test.mat') 
print("data.keys()",data.keys())
 
origin_X = np.array(data['x'].flat)
origin_Y = data['y'][0].reshape(3000,-1)
origin_Y_onehot= data['y_onehot'].reshape(3000,4,19)
    
print ("origin_X shape: "+str(origin_X.shape))
print ("origin_Y shape: "+str(origin_Y.shape))
print ("origin_Y_onehot shape: "+str(origin_Y_onehot.shape))

# In[7]: check input_x data
showdata_seed=12

print ("input_x shape: "+str(origin_X.shape))
print ("origin_Y_onehot shape: "+str(origin_Y_onehot.shape))


'''    '''
position="~ 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 ~\n"
plt.imshow(origin_X[showdata_seed])
print ("origin_Y : "+str(origin_Y[showdata_seed]))
print ("origin_Y_onehot : \n"+position+str(origin_Y_onehot[showdata_seed]))
    

# In[7]: predict
index=0
error=0
error_n=0
accumulate_time=0

start_total=time.time()

for i in origin_X:
    # 初始化
    print ('\n\n-------------------------------------------')
    print ('---------------- index',index,'----------------')
    predict_out=[]
    answer_out=[]
    temp=0
    start = time.time()
    # 依據使用的模型決定是否要預處理 
    # EX: model_name ='dn1_test_2021_10_19 01_47_20.h5'
    #     model_name.split('_')[0] == 'dn1'
    
    if model_name.split('_')[0] == 'nopre': 
        img = Image.fromarray(i) # array要轉為image才能resize
        img=img.resize((130,50))
    elif model_name.split('_')[0] == 'dn1':
        i=predo.img_denoise(i)
        img = Image.fromarray(i) # array要轉為image才能resize
        img=img.resize((130,50))
    elif model_name.split('_')[0] == 'dn2':
        i=predo.img_denoise2(i)
        img = Image.fromarray(i) # array要轉為image才能resize
        img=img.resize((130,50))
        
    h=np.array(img)
    #print (h.shape)
    h=h.reshape(1,50,130,-1) # 轉為 (1,50,130,3) or (1,50,130,1)
    
    prediction=test.predict(h.astype('float32')/255) # predict
    
    # 轉譯偵測結果
    N1=translate_Y(one_hot_decode(prediction[0])[0]) # one_hot_decode 將19可能依信心排序
    N2=translate_Y(one_hot_decode(prediction[1])[0]) # translate_Y 由數字 1~19轉為符號
    N3=translate_Y(one_hot_decode(prediction[2])[0])
    N4=translate_Y(one_hot_decode(prediction[3])[0])   
    
    
    
    predict_out.append((N1,N2,N3,N4)) #四字元合併
    
    end = time.time()
    accumulate_time += (end - start)
    
    # 轉譯 Ground Truth    
    a1=translate_Y(origin_Y[index][0])
    a2=translate_Y(origin_Y[index][1])
    a3=translate_Y(origin_Y[index][2])
    a4=translate_Y(origin_Y[index][3])
    
    answer_out.append((a1,a2,a3,a4))
    
    
    if N1!=a1:
        error_n+=1 #此模型錯幾個字元
        temp+=1    #此圖錯幾個字元
    if N2!=a2:
        error_n+=1
        temp+=1
    if N3!=a3:
        error_n+=1
        temp+=1
    if N4!=a4:
        error_n+=1
        temp+=1
    
    if (N1!=a1 or N2!=a2 or N3!=a3 or N4!=a4):
        '''
        img = Image.fromarray(i, 'RGB')
        plt.imshow(img)
        plt.show()
        '''
        error+=1  
        print("predict=",predict_out)
        print("answer=",answer_out,",wrong words=",temp)
        
    if index<40:
        plt.imshow(img)
        plt.show()
        print("predict=",predict_out)
        print("answer=",answer_out,",wrong words=",temp)
        
    index+=1
    '''
    #快速看demo結果用，可將此段註解#
    if (index==250): 
        break
    ###########################
    '''
# In[7]: result   
num_picture=index
acc= (num_picture-error)/num_picture
acc_n=(num_picture*4-error_n)/(num_picture*4)

print (num_picture,"picture total wrong =",error)
print ("image accuracy = ",acc)
print ("word accuracy = ",acc_n)

print ('average per execute time: %f ms' %(accumulate_time/len(origin_X)*1000))

acc_total=0
end_total=time.time()
acc_total+=end_total-start_total
print ("total execute time = %f s " %(acc_total))



