#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import cv2

#import matplotlib.pyplot as plt

img = cv2.imread('123.jpg', 0)

#濾波
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#Otsu濾波
ret2,th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#print ret2
#cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow('Image', img)
cv2.imshow('2', th1)
cv2.imshow('3', th2)
cv2.imwrite('trans1.jpg', th2)
#plt.subplot(222),plt.imshow(th1, 'gray')
#plt.subplot(223),plt.imshow(th2, 'gray')
#cv2.destroyAllWindows()
cv2.waitKey(0)
