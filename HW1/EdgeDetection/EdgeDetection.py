import cv2 
import matplotlib.pyplot as plt 


def Edge_Detection(img,method,gray=True,XY=(1,1),Weight=(0.5,0.5,0),th=(30,150),size=3,save_image=True,show_image=True):
    # RGB to grayscale
    if gray:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            print ('Error!!! check your [img] shape/dtype')
            return
       
    # Edge_Detection
    if method=='Sobel':
        img_x = cv2.Sobel(img, cv2.CV_16S, XY[0], 0, ksize=size) # 計算水平
        img_y = cv2.Sobel(img, cv2.CV_16S, 0, XY[1], ksize=size) # 計算垂直

    elif method=='Scharr':
        img_x = cv2.Scharr(img,cv2.CV_64F, XY[0], 0)
        img_y = cv2.Scharr(img,cv2.CV_64F, 0, XY[1])
    
    elif method=='Laplacian':
        post_img = cv2.Laplacian(img, cv2.CV_16S, ksize=size)
        post_img = cv2.convertScaleAbs(post_img)
        
    elif method=='Canny':
        if gray:
            post_img = cv2.Canny(img, th[0], th[1], apertureSize=size)
        else:
            print ('Error!!! Canny must be converted to grayscale, set gray=True')
            return
    else:
        print ('method have to be Sobel, Scharr, Laplacian or Canny')
        
    
    # addWeight for Sobel and Scharr
    if method=='Sobel' or method=='Scharr':
        absX = cv2.convertScaleAbs(img_x)
        absY = cv2.convertScaleAbs(img_y)
        post_img=cv2.addWeighted(absX, Weight[0], absY, Weight[1], Weight[2])
       
    # save image or not    
    if save_image: cv2.imwrite('ouutput/lenna_'+method+'.jpg', post_img) 
    
    print ('edge detect method:',method)
    print ('post_img.shape:',post_img.shape)
    print ('post_img.dtype',post_img.dtype)
    print ("//-------------------------------------------")
    
    # show image or not 
    if show_image:
        cv2.imshow(str(method)+"_post_img", post_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return post_img



def Image_Filter (img,method,size=3,show_image=False): 
    if method=='blur':
        img = cv2.blur(img, (size,size))
    elif method=='GaussianBlur':
        img = cv2.GaussianBlur(img, (size, size), 0)
    elif method=='medianBlur':
        img = cv2.medianBlur(img, size)
    elif method=='bilateralFilter':
        img = cv2.bilateralFilter(img,size,75,75)
    else:
        print ('method have to be blur, GaussianBlur, medianBlur or bilateralFilter')
    if show_image:
        cv2.imshow("Image_Filter", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img



##################################################################################
# Edge dectection using Canny
img = cv2.imread("EdgeDetection/input/cat.jpg")

# turn to grayscale first
pre_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# pre-filter
pre_img = Image_Filter(pre_img,'GaussianBlur',size=3,show_image=True)

# edge detection
post_img=cv2.Canny(pre_img, 0, 2)

# post-filter
post_img=Image_Filter(post_img,'GaussianBlur',show_image=True)

# Binarization
ret, th1 = cv2.threshold(post_img,80,255,cv2.THRESH_BINARY)



cv2.imshow('post_img', post_img)
cv2.waitKey(0)
cv2.imshow('th1_post_img', th1)
cv2.destroyAllWindows()
cv2.imwrite('EdgeDetection/output/cat_final_canny.jpg', th1)
##################################################################################



##################################################################################
# Edge dectection using Sobel
img = cv2.imread("EdgeDetection/input/cat.jpg")

# turn to grayscale first
pre_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# pre-filter
pre_img = Image_Filter(pre_img,'GaussianBlur',size=3,show_image=True)

# edge detection
post_img=Edge_Detection(pre_img,'Sobel',gray=False,show_image=True, XY=(1, 1))

# post-filter
post_img=Image_Filter(post_img,'GaussianBlur',show_image=True)

# Binarization
ret, th1 = cv2.threshold(post_img,50,255,cv2.THRESH_BINARY)



cv2.imshow('post_img', post_img)
cv2.waitKey(0)
cv2.imshow('th1_post_img', th1)
cv2.destroyAllWindows()
cv2.imwrite('EdgeDetection/output/cat_final_sobel.jpg', th1)
##################################################################################




##################################################################################
# Edge dectection using Scharr
img = cv2.imread("EdgeDetection/input/cat.jpg")

# turn to grayscale first
pre_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# pre-filter
pre_img = Image_Filter(pre_img,'GaussianBlur',size=3,show_image=True)

# edge detection
post_img=Edge_Detection(pre_img,'Scharr',gray=False,show_image=True, XY=(1, 1))

# post-filter
post_img=Image_Filter(post_img,'GaussianBlur',show_image=True)

# Binarization
ret, th1 = cv2.threshold(post_img,135,255,cv2.THRESH_BINARY)



cv2.imshow('post_img', post_img)
cv2.waitKey(0)
cv2.imshow('th1_post_img', th1)
cv2.destroyAllWindows()
cv2.imwrite('EdgeDetection/output/cat_final_scharr.jpg', th1)
##################################################################################



##################################################################################
# Edge dectection using Laplacian 
img = cv2.imread("EdgeDetection/input/cat.jpg")

# turn to grayscale first
pre_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# pre-filter
pre_img = Image_Filter(pre_img,'GaussianBlur',size=3,show_image=True)

# edge detection
post_img=Edge_Detection(pre_img,'Laplacian',gray=False,show_image=True, XY=(1, 1))

# post-filter
post_img=Image_Filter(post_img,'GaussianBlur',show_image=True)

# Binarization
ret, th1 = cv2.threshold(post_img,30,255,cv2.THRESH_BINARY)



cv2.imshow('post_img', post_img)
cv2.waitKey(0)
cv2.imshow('th1_post_img', th1)
cv2.destroyAllWindows()
cv2.imwrite('EdgeDetection/output/cat_final_laplacian.jpg', th1)
##################################################################################