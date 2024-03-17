# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 01:56:34 2021

@author: jacky
"""
import os
import scipy.io
#import matplotlib as plt
import numpy as np
from PIL import Image 
import cv2
import matplotlib.pyplot as plt

# In[ ]:
#load data from train.mat 去迴歸

def load_data_denoise_train():
    data = scipy.io.loadmat('database/denoise_train.mat') 
    print(data.keys())
    denoise_x =np.array(data['denoise_x'])
    print ("denoise_x shape: "+str(denoise_x.shape))

    return denoise_x

# In[ ]:
#load data from train.mat 僅降躁

def load_data_denoise_train2():
    data = scipy.io.loadmat('database/denoise_train2.mat') 
    print(data.keys())
    denoise2_x =np.array(data['denoise2_x'])
    print ("denoise2_x shape: "+str(denoise2_x.shape))

    return denoise2_x

  # In[ ]:  
def load_data():
    data = scipy.io.loadmat('database/train.mat') 
    print(data.keys())
 
    origin_X = np.array(data['x'].flat)
    origin_Y = data['y'][0].reshape(5000,-1)
    origin_Y_onehot= data['y_onehot'].reshape(5000,4,19)
    
    print ("origin_X shape: "+str(origin_X.shape))
    print ("origin_Y shape: "+str(origin_Y.shape))
    print ("origin_Y_onehot shape: "+str(origin_Y_onehot.shape))
 
    return origin_X,origin_Y,origin_Y_onehot

# In[ ]:
def check_image(origin_X,origin_Y,origin_Y_onehot):
    print (origin_X.shape)
    showdata_seed=4456
    position="~ 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 ~\n"
    plt.imshow(origin_X[showdata_seed])
    print ("origin_Y : "+str(origin_Y[showdata_seed]))
    print ("origin_Y_onehot : \n"+position+str(origin_Y_onehot[showdata_seed]))

# In[ ]:  
# display data image size     
def Average(lst): 
    return sum(lst) / len(lst)


def image_size(data):
    image_high=[]
    image_width=[]

    for i in data:
        image_high.append(i.shape[0])
        image_width.append(i.shape[1])
        
    
    print ('image_high max= '+str(max(image_high)))
    print ('image_high min= '+str(min(image_high)))
    print ('image_high Average= '+str(Average(image_high)))
    
    print ('image_width max= '+str(max(image_width)))
    print ('image_width min= '+str(min(image_width)))
    print ('image_width Average= '+str(Average(image_width)))


# In[ ]:
def save_origin_image(data):
    index=0
        
    for i in data:
        name='origin_data_image/origin_'+str(index)+'.jpg'
        if os.path.isfile(name):      
            print (name+" is existed")
        else:
            img = Image.fromarray(i, 'RGB')
            img.save(name)
    print ("save origin image done")

# In[ ]:
    
def resize_img (o_data,write,save):  
    index=0
    p_data=[]
    for i in o_data:
        name='resize_data_image/resize_x_'+str(index)+'.jpg'
        img = Image.fromarray(i, 'RGB')
        img=img.resize((130,50))
        if os.path.isfile(name) and save:      
            print (name+" is existed")    
        elif save:
            img.save(name)
        if write:
            p_data.append(np.array(img))       
        index+=1
        
    p_data=np.array(p_data)   
    print (p_data.shape)
    return p_data


# In[ ]:   
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def img_denoise(img):

    dst = cv2.fastNlMeansDenoisingColored(img,None,30,30,7,21)
    ret,thresh = cv2.threshold(dst,127,255,cv2.THRESH_BINARY_INV)
    #plt.subplot(121),plt.imshow(img)
    #plt.subplot(122),plt.imshow(dst)
    #plt.show()
    #print("fastNlMeansDenoisingColored",img.shape)
    
    
    #plt.imshow(thresh)
    #plt.show()
    #print("threshold",thresh.shape)

    imggra = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    #plt.imshow(imggra)
    #plt.show()
    #print("cvtColor",imggra.shape)
    
    imgline = np.copy(imggra)
    imgline[:,3:imggra.shape[1]-3] = 0
    #plt.imshow(imgline)
    #plt.show()
    #print("imgline",imgline.shape)
    
    imgline=cv2.resize(imgline, (imgline.shape[1]*16,imgline.shape[0]*16), interpolation=cv2.INTER_CUBIC)
    imgdata = np.where(imgline == 255)
    #print(imgdata)

    #plt.scatter(imgdata[1], imgline.shape[0]-imgdata[0], s = 100,
              #  c = 'red', label = 'cluster')
    #plt.ylim(ymin = 0)
    #plt.ylim(ymax = imgline.shape[0])
    #plt.show()
    #print("imgline resize")
    
    X = np.array([imgdata[1]])
    Y = imgline.shape[0] - imgdata[0]

    poly_reg= PolynomialFeatures(degree = 2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)

    X2 = np.array([[i for i in range(0,imgline.shape[1])]])
    X2_ = poly_reg.fit_transform(X2.T)

    #print('Coefficient:{}'.format(regr.coef_) )
    #print('Intercept:{}'.format(regr.intercept_) )

    newimg = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    newimg = cv2.resize(newimg, (newimg.shape[1]*16,newimg.shape[0]*16), interpolation=cv2.INTER_CUBIC)

    for ele in np.column_stack([regr.predict(X2_).round(0),X2[0],] ):
        pos = imgline.shape[0]-int(ele[0])
        w = 39
        newimg[pos-w:pos+w,int(ele[1])] = 255 - newimg[pos-w:pos+w,int(ele[1])]
    
    #plt.imshow(thresh)
    #plt.show()
    #print("thresh",thresh.shape)
    
    #plt.imshow(newimg)
    #plt.show()
    #print("newimg",newimg.shape)
    
    '''
    ret,thresh = cv2.threshold(newimg,40,255,cv2.THRESH_BINARY)
    plt.imshow(thresh)
    plt.show()
    print("thresh1",thresh.shape)
    '''
    
    newimg = cv2.resize(newimg, (int(newimg.shape[1]/16),int(newimg.shape[0]/16)), interpolation=cv2.INTER_CUBIC)
    #ret,thresh = cv2.threshold(newimg,40,255,cv2.THRESH_BINARY)
    #plt.imshow(thresh)
    #plt.show()
    #print("thresh2",thresh.shape)
    
    
    ndst = cv2.fastNlMeansDenoising(newimg,None,50,7,21)  
    #plt.imshow(ndst)
    #plt.show()
    #print("thresh3",ndst.shape)
    data = np.copy(ndst)
    
    return data

def denoise_img (o_data,write,save):  
    """

    Parameters
    ----------
    o_data : TYPE
        input image.
    write : Boolen
        write data to output.
    save : Boolen
        save img.

    Returns
    -------
    p_data : TYPE
        DESCRIPTION.

    """
    index=0
    p_data=[]
    for i in o_data:
        print (index)
        name='denoise_data_image/denoise_x_'+str(index)+'.jpg'
        i=img_denoise(i)
        img = Image.fromarray(i)
        img=img.resize((130,50))
        if os.path.isfile(name) and save:      
            print (name+" is existed")    
        elif save:
            if not os.path.isdir('denoise_data_image'):
                os.mkdir('denoise_data_image')
            img.save(name)
        if write:
            p_data.append(np.array(img))       
        index+=1
        
    p_data=np.array(p_data)   
    print (p_data.shape)
    return p_data
         

# In[ ]:   
def img_denoise2(img):
    dst = cv2.fastNlMeansDenoisingColored(img,None,30,30,7,21)
    ret,thresh = cv2.threshold(dst,127,255,cv2.THRESH_BINARY_INV)
    #plt.subplot(121),plt.imshow(img)
    #plt.subplot(122),plt.imshow(dst)
    #plt.show()
    #print("fastNlMeansDenoisingColored",img.shape)
    
    
    #plt.imshow(thresh)
    #plt.show()
    #print("threshold",thresh.shape)

    imggra = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    #plt.imshow(imggra)
    #plt.show()
    #print("cvtColor",imggra.shape)
    
    data = np.copy(imggra)
    
    return data

def denoise_img2 (o_data,write,save):  
    index=0
    p_data=[]
    for i in o_data:
        print (index)
        name='denoise_data_image2/denoise2_x_'+str(index)+'.jpg'
        i=img_denoise2(i)
        img = Image.fromarray(i)
        img=img.resize((130,50))
        if os.path.isfile(name) and save:      
            print (name+" is existed")    
        elif save:
            if not os.path.isdir('denoise_data_image2'):
                os.mkdir('denoise_data_image2')
            img.save(name)
            img.save(name)
        if write:
            p_data.append(np.array(img))       
        index+=1
        
    p_data=np.array(p_data)   
    print (p_data.shape)
    return p_data
         
