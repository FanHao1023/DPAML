import cv2
import ImageProcess
import numpy as np

def order_points_new(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0,1]!=leftMost[1,1]:
        leftMost=leftMost[np.argsort(leftMost[:,1]),:]
    else:
        leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
    (tl, bl) = leftMost
    if rightMost[0,1]!=rightMost[1,1]:
        rightMost=rightMost[np.argsort(rightMost[:,1]),:]
    else:
        rightMost=rightMost[np.argsort(rightMost[:,0])[::-1],:]
    (tr,br)=rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="int")




##############################################################################################################
#                                             first version                                                  #
##############################################################################################################

image_list = ['LicensePlate/image/01.jpg', 'LicensePlate/image/02.jpg', 'LicensePlate/image/03.jpg']
result = []

for image in image_list:
    ori_img = cv2.imread(image)

    cv2.imshow('img', ori_img)
    cv2.waitKey(0)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    img=ImageProcess.Image_Filter(img,'bilateralFilter',show_image=True,size=2)
    post_img=ImageProcess.Edge_Detection(img,'Sobel',gray=False, show_image=True)
    # post_img=ImageProcess.Image_Filter(post_img,'GaussianBlur',show_image=True,size=4)
    ret, th1 = cv2.threshold(post_img,110,255,cv2.THRESH_BINARY)


    cv2.imshow('post_img', post_img)
    cv2.imshow('th1_post_img', th1)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    #cv2.imwrite('output/02_final.jpg', th1)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #執行影象形態學

    # 閉運算
    closed = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    # 膨脹2次，讓輪廓突出
    closed = cv2.dilate(closed, None, iterations=2)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    # 腐蝕2次，去掉細節
    closed = cv2.erode(closed, None, iterations=2)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # (_, cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    possible_img = ori_img.copy()
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    Box = np.int0(cv2.boxPoints(rect))
    Final_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
    cv2.imshow('Final_img', Final_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


    possible_img = ori_img.copy()

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        #print (c)
      
        rect = cv2.minAreaRect(c)
        #print ('rectt', rect)
        Box = np.int0(cv2.boxPoints(rect))
        #print ('Box', Box)    
        Box=order_points_new(Box) # return  左上/右上/右下/左下 (x,y)
        #print ('Box2',Box)
    
        possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
        cv2.imshow('possible_img', possible_img)
     
    #cv2.imshow('possible_img', possible_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


    possible_img = ori_img.copy()

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        #print (c)
      
        rect = cv2.minAreaRect(c)
        #print ('rectt', rect)
        Box = np.int0(cv2.boxPoints(rect))
        #print ('Box', Box)    
        Box=order_points_new(Box) # return  左上/右上/右下/左下 (x,y)
        #print ('Box2',Box)
    
        # determine by 工人智慧,指定方框條件設定,同學們可以在這裡調整條件
    
        if (((Box[1][0]-Box[0][0])-(Box[3][1]-Box[0][1]))>40) and 20<abs(Box[1][1]-Box[2][1])<100  and 60<abs(Box[0][0]-Box[1][0])<200  and abs(Box[0][1]-Box[1][1])<20  :
            possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
            cv2.imshow('possible_img', possible_img)
            break
    
    cv2.imshow('possible_img', possible_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    result.append(possible_img)

cv2.imwrite('LicensePlate/output/01.jpg', result[0])
cv2.imwrite('LicensePlate/output/02.jpg', result[1])
cv2.imwrite('LicensePlate/output/03.jpg', result[2])

##############################################################################################################
#                                          end of first version                                              #
##############################################################################################################



##############################################################################################################
#                                            second version                                                  #
##############################################################################################################
# first set of parameters
image_list_1 = ['LicensePlate/image/01.jpg', 'LicensePlate/image/02.jpg', 'LicensePlate/image/03.jpg']
result = []

for image in image_list_1:
    ori_img = cv2.imread(image)

    cv2.imshow('img', ori_img)
    cv2.waitKey(0)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    img=ImageProcess.Image_Filter(img,'bilateralFilter',show_image=True,size=2)
    post_img=ImageProcess.Edge_Detection(img,'Sobel',gray=False, show_image=True)
    # post_img=ImageProcess.Image_Filter(post_img,'GaussianBlur',show_image=True,size=4)
    ret, th1 = cv2.threshold(post_img,110,255,cv2.THRESH_BINARY)


    cv2.imshow('post_img', post_img)
    cv2.imshow('th1_post_img', th1)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    #cv2.imwrite('output/02_final.jpg', th1)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #執行影象形態學

    # 閉運算
    closed = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    # 膨脹2次，讓輪廓突出
    closed = cv2.dilate(closed, None, iterations=2)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    # 腐蝕2次，去掉細節
    closed = cv2.erode(closed, None, iterations=2)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # (_, cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    possible_img = ori_img.copy()
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    Box = np.int0(cv2.boxPoints(rect))
    Final_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
    cv2.imshow('Final_img', Final_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


    possible_img = ori_img.copy()

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        #print (c)
      
        rect = cv2.minAreaRect(c)
        #print ('rectt', rect)
        Box = np.int0(cv2.boxPoints(rect))
        #print ('Box', Box)    
        Box=order_points_new(Box) # return  左上/右上/右下/左下 (x,y)
        #print ('Box2',Box)
    
        possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
        cv2.imshow('possible_img', possible_img)
     
    #cv2.imshow('possible_img', possible_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


    possible_img = ori_img.copy()

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        #print (c)
      
        rect = cv2.minAreaRect(c)
        #print ('rectt', rect)
        Box = np.int0(cv2.boxPoints(rect))
        #print ('Box', Box)    
        Box=order_points_new(Box) # return  左上/右上/右下/左下 (x,y)
        #print ('Box2',Box)
    
        # determine by 工人智慧,指定方框條件設定,同學們可以在這裡調整條件
    
        if (((Box[1][0]-Box[0][0])-(Box[3][1]-Box[0][1]))>40) and 20<abs(Box[1][1]-Box[2][1])<100  and 60<abs(Box[0][0]-Box[1][0])<200  and abs(Box[0][1]-Box[1][1])<20  :
            possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
            cv2.imshow('possible_img', possible_img)
            break
    
    cv2.imshow('possible_img', possible_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    result.append(possible_img)


# second set of parameters
image_list_2 = ['LicensePlate/image/04.png', 'LicensePlate/image/05.jpg', 'LicensePlate/image/07.jpg', 'LicensePlate/image/08.jpg']

for image in image_list_2:
    ori_img = cv2.imread(image)

    cv2.imshow('img', ori_img)
    cv2.waitKey(0)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    img=ImageProcess.Image_Filter(img,'bilateralFilter',show_image=True,size=2)
    post_img=ImageProcess.Edge_Detection(img,'Sobel',gray=False, show_image=True)
    # post_img=ImageProcess.Image_Filter(post_img,'GaussianBlur',show_image=True,size=4)
    ret, th1 = cv2.threshold(post_img,140,255,cv2.THRESH_BINARY)


    cv2.imshow('post_img', post_img)
    cv2.imshow('th1_post_img', th1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite('output/02_final.jpg', th1)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #執行影象形態學

    # 閉運算
    closed = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    # 膨脹3次，讓輪廓突出
    closed = cv2.dilate(closed, None, iterations=3)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    # 腐蝕3次，去掉細節
    closed = cv2.erode(closed, None, iterations=3)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # (_, cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    possible_img = ori_img.copy()
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    Box = np.int0(cv2.boxPoints(rect))
    Final_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
    cv2.imshow('Final_img', Final_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
 

    possible_img = ori_img.copy()

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        #print (c)
      
        rect = cv2.minAreaRect(c)
        #print ('rectt', rect)
        Box = np.int0(cv2.boxPoints(rect))
        #print ('Box', Box)    
        Box=order_points_new(Box) # return  左上/右上/右下/左下 (x,y)
        #print ('Box2',Box)
    
        possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
        cv2.imshow('possible_img', possible_img)
     
    #cv2.imshow('possible_img', possible_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


    possible_img = ori_img.copy()

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        #print (c)
      
        rect = cv2.minAreaRect(c)
        #print ('rectt', rect)
        Box = np.int0(cv2.boxPoints(rect))
        #print ('Box', Box)    
        Box=order_points_new(Box) # return  左上/右上/右下/左下 (x,y)
        #print ('Box2',Box)
    
        # determine by 工人智慧,指定方框條件設定,同學們可以在這裡調整條件
    
        if (((Box[1][0]-Box[0][0])-(Box[3][1]-Box[0][1]))>5) and 10<abs(Box[1][1]-Box[2][1])<40  and 30<abs(Box[0][0]-Box[1][0])<200  and abs(Box[0][1]-Box[1][1])<20  :
            possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
            cv2.imshow('possible_img', possible_img)
            break
    
    cv2.imshow('possible_img', possible_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    result.append(possible_img)


# third set of parameters
image_list_3 = ['LicensePlate/image/06.jpg']
for image in image_list_3:
    ori_img = cv2.imread(image)

    cv2.imshow('img', ori_img)
    cv2.waitKey(0)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    img=ImageProcess.Image_Filter(img,'bilateralFilter',show_image=True,size=2)
    post_img=ImageProcess.Edge_Detection(img,'Sobel',gray=False, show_image=True)
    # post_img=ImageProcess.Image_Filter(post_img,'GaussianBlur',show_image=True,size=4)
    ret, th1 = cv2.threshold(post_img,100,255,cv2.THRESH_BINARY)


    cv2.imshow('post_img', post_img)
    cv2.imshow('th1_post_img', th1)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    #cv2.imwrite('output/02_final.jpg', th1)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #執行影象形態學

    # 閉運算
    closed = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    # 膨脹5次，讓輪廓突出
    closed = cv2.dilate(closed, None, iterations=5)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    # 腐蝕2次，去掉細節
    closed = cv2.erode(closed, None, iterations=2)
    cv2.imshow('erode dilate', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # (_, cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    possible_img = ori_img.copy()
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    Box = np.int0(cv2.boxPoints(rect))
    Final_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
    cv2.imshow('Final_img', Final_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
 

    possible_img = ori_img.copy()

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        #print (c)
      
        rect = cv2.minAreaRect(c)
        #print ('rectt', rect)
        Box = np.int0(cv2.boxPoints(rect))
        #print ('Box', Box)    
        Box=order_points_new(Box) # return  左上/右上/右下/左下 (x,y)
        #print ('Box2',Box)
    
        possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
        cv2.imshow('possible_img', possible_img)
     
    #cv2.imshow('possible_img', possible_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


    possible_img = ori_img.copy()

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        #print (c)
      
        rect = cv2.minAreaRect(c)
        #print ('rectt', rect)
        Box = np.int0(cv2.boxPoints(rect))
        #print ('Box', Box)    
        Box=order_points_new(Box) # return  左上/右上/右下/左下 (x,y)
        #print ('Box2',Box)
    
        # determine by 工人智慧,指定方框條件設定,同學們可以在這裡調整條件
    
        if (((Box[1][0]-Box[0][0])-(Box[3][1]-Box[0][1]))>5) and 10<abs(Box[1][1]-Box[2][1])<40  and 55<abs(Box[0][0]-Box[1][0])<100  and abs(Box[0][1]-Box[1][1])<20  :
            possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
            cv2.imshow('possible_img', possible_img)
            break
    
    cv2.imshow('possible_img', possible_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    result.append(possible_img)



cv2.imwrite('LicensePlate/output/01_v2.jpg', result[0])
cv2.imwrite('LicensePlate/output/02_v2.jpg', result[1])
cv2.imwrite('LicensePlate/output/03_v2.jpg', result[2])
cv2.imwrite('LicensePlate/output/04_v2.jpg', result[3])
cv2.imwrite('LicensePlate/output/05_v2.jpg', result[4])
cv2.imwrite('LicensePlate/output/07_v2.jpg', result[5])   
cv2.imwrite('LicensePlate/output/08_v2.jpg', result[6]) 
cv2.imwrite('LicensePlate/output/06_v2.jpg', result[7])  

##############################################################################################################
#                                         end of second version                                              #
##############################################################################################################