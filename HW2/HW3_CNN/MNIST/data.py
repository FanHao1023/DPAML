# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:43:42 2021

@author: jacky
"""

# In[0]:
import scipy.io
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# In[1]:
def load_original_data():
    print ('\n-------------load_original_data-------------')
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print(type(mnist))
    train_img = mnist.train.images.reshape(55000,28,28)
    train_label = mnist.train.labels.reshape(55000,-1)
    test_img = mnist.test.images.reshape(10000,28,28)
    test_label = mnist.test.labels.reshape(10000,-1)

    
    return train_img,train_label,test_img,test_label
    


# In[2]:
def show_data (train_img,train_label,test_img,test_label,index):
    print ('\n-----------------show_data------------------')
    
    
    print ('img pixel type :',type(train_img[0][0][0]))
    print('\n')
    
    print ('train_img shape :',train_img.shape)
    print ('train_label shape :',train_label.shape)
    print('\n')
    
    print ('test_img shape :',test_img.shape)
    print ('test_label shape :',test_label.shape)
    picture=train_img[index]
    print ('--------------------------------------------')
    print ('train_img',index,'shape :' ,picture.shape)
    plt.imshow(picture)
    plt.show()
    
# In[3]:
def save_mat_data(train_data,train_label,test_data,test_label): 
    print ('\n---------------save_mat_data----------------\n')
    scipy.io.savemat('data.mat',{'train_data':train_data,'train_label':train_label,
                                 'test_data':test_data,'test_label':test_label})
    print ('save_mat_data finish !!!')

# In[4]:
def resize_img (o_data,new_astype,new_size):  
    print ('\n----------------resize_img------------------')
    print ('o_data shape :',o_data.shape)
    index=0
    p_data=[]
    for i in o_data:
        ############################
        if new_astype == 'uint8':
            i=i*255
            i=i.astype(np.uint8)
            '''
            print (type(i),i.shape)
            print ('img pixel type :',type(i[0][0]))
            plt.imshow(i)
            plt.show()
            '''
        ############################
        img = Image.fromarray(i)
        '''
        print ('array2img')
        plt.imshow(img)
        plt.show()
        '''
        ############################
        img=img.resize((new_size,new_size)) 
        p_data.append(np.array(img))     
        index+=1
        '''
        print ('resize_img')
        plt.imshow(img)
        plt.show()
        '''
        
        #print ('--------------------------------------------')
        
    p_data=np.array(p_data)   
    print ('p_data shape :',p_data.shape)

    return p_data
# In[5]:
def check_mat ():  
    print ('\n-----------------check_mat------------------')
    data = scipy.io.loadmat('data.mat') 
    print('data keys :',data.keys())
    train_data =np.array(data['train_data'])
    print ('train_data shape :',train_data.shape)
    print ('img pixel type :',type(train_data[0][0][0]))
    #print ('check_mat train data',data['train_data'][2])
    print ('check_mat train label',data['train_label'][2])
    plt.imshow(train_data[2])
    plt.show()
    

    
# In[6]: 
def data_convert (new_astype='float32',new_size=28):
    '''
    ##################################################################
                    ori mnist data 28x28(float32)
                    change astype using new_astype=XXX
                    change img size using new_size=XXX
    ##################################################################
    '''
    train_img,train_label,test_img,test_label=load_original_data()
    show_data (train_img,train_label,test_img,test_label,index=2)
    
    print ('----- resize train_img from 28x28 to ',new_size,'x',new_size,'with',new_astype)
    n_train_img=resize_img (train_img,new_astype,new_size)
    print ('----- resize test_img  from 28x28 to ',new_size,'x',new_size,'with',new_astype)
    n_test_img=resize_img (test_img,new_astype,new_size)
    
    show_data (n_train_img,train_label,n_test_img,test_label,index=2)
    save_mat_data(n_train_img,train_label,n_test_img,test_label)
    check_mat ()
    
# In[6]: 
def load_new_data (check):
    '''
    ##################################################################
                    load mnist data 28x28(float32)
    ##################################################################
    data name : 'data.mat'
    data keys : dict_keys('train_data', 'train_label', 'test_data', 'test_label'])
    ##################################################################
    train_img shape : (55000, 28, 28)
    train_label shape : (55000, 10)
    test_img shape : (10000, 28, 28)
    test_label shape : (10000, 10)
    ##################################################################
    '''
    data = scipy.io.loadmat('data.mat')
    train_data=data['train_data']
    train_label=data['train_label']
    test_data=data['test_data']
    test_label=data['test_label']
    if (check):
        check_mat ()
    return  train_data,train_label,test_data,test_label
    
    
    
