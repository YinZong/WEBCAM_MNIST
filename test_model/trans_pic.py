#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np
from PIL import Image
#import model
#讀取已訓練好的模型
#from minst_model import *

#model = MnistModel('models/mnist2/mnist2.tf1')
#model = MnistModel('/home/kevin/桌面/mnist/model.tar.gz')

#將圖片轉成灰階格式
img = Image.open('/home/kevin/桌面/mnist/1.jpg').convert('L')

#resize圖片
if img.size[0] != 28 or img.size[1] !=28:
    img = img.resize((28,28))

#暫存像素的一維陣列
arr = []

for i in range(28):
    for j in range(28):
        #mnist裡的顏色0代表白色，1代表黑色
        pixcel = 1.0 - float(img.getpixel((j,i)))/255.0
        #pixcel = 255.0 - float(img.getpixel((j,i)))#如果是0~255的顏色值
        arr.append(pixcel)

arr1 = np.array(arr).reshape((1,28,28,1))
print(arr)
img.save('/home/kevin/桌面/mnist/0.png')
img.show(arr)

