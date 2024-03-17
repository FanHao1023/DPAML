# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:49:35 2021

@author: jacky
"""
import predo
import scipy.io
from PIL import Image 
from matplotlib import pyplot as plt

origin_X,origin_Y,origin_Y_onehot=predo.load_data()
resize_x=predo.resize_img(origin_X,True,False)  
i= resize_x[3059]

img = Image.fromarray(i, 'RGB')
plt.imshow(img)
plt.show() 
print(predo.img_denoise(i).shape)
plt.imshow(predo.img_denoise(i))
plt.show()
           


# In[]:

resize_x=predo.resize_img(origin_X,True,False)     
denoise_x=predo.denoise_img (resize_x,True,True)   
scipy.io.savemat('database/denoise_train.mat',{'denoise_x':denoise_x})

# In[]:
origin_X,origin_Y,origin_Y_onehot=predo.load_data()
resize_x=predo.resize_img(origin_X,True,False)     
denoise_x2=predo.denoise_img2 (resize_x,True,True)   
scipy.io.savemat('database/denoise_train2.mat',{'denoise2_x':denoise_x2})