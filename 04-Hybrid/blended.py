#!/usr/bin/env python3
# image blending [1]


import numpy as np
import cv2

# read images with cv2
img0 = cv2.imread('este.jpg')
img1 = cv2.imread('javi.jpg')


gausspyr0 = img0.copy() 
gausspyr1 = img1.copy()
gpimg0 = [gausspyr0]
gpimg1 = [gausspyr1]
for i in range(7):
    gausspyr0 = cv2.pyrDown(gausspyr0) 
    gpimg0.append(gausspyr0) #gaussian pyramid for img0
    gausspyr1 = cv2.pyrDown(gausspyr1)
    gpimg1.append(gausspyr1) #gaussian pyramid for img1
 

lpimg0 = [gpimg0[6]] 
lpimg1 = [gpimg1[6]]
for i in range(6,0,-1):
    lap0 = cv2.subtract(gpimg0[i-1],cv2.pyrUp(gpimg0[i]))
    lpimg0.append(lap0) #laplacian pyramid for img0
    lap1 = cv2.subtract(gpimg1[i-1],cv2.pyrUp(gpimg1[i]))
    lpimg1.append(lap1) #laplacian pyramid for img1 


LaplSum = []
for limg0,limg1 in zip(lpimg0,lpimg1):
    rows,cols,dpt = limg0.shape
    laplsum = np.hstack((limg0[:,0:int(cols/2)], limg1[:,int(cols/2):])) #add left and righ imgs parts at each level
    LaplSum.append(laplsum)

laplsumrec = LaplSum[0]
for i in range(1,7):
     laplsumrec = cv2.pyrUp(laplsumrec)
     laplsumrec = cv2.add(laplsumrec, LaplSum[i]) #reconstruction
   

dirBlend = np.hstack((img0[:,:int(cols/2)],img1[:,int(cols/2):])) #direct blending
   
cv2.imwrite('Pyrblend.jpg',laplsumrec)
cv2.imwrite('Dirblend.jpg',dirBlend)

"""
References
[1] "OpenCV: Image Pyramids", Docs.opencv.org. [Online]. Available: https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html. [Accessed: 21- Feb- 2019].
"""
