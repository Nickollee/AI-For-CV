'''Assignments:
1.Recode all the examples.
2.Change image through YUV space.
3.Combine image crop,colour shift,rotation and perspective transform together to complete a data augmentation script.
# author:Niki
'''
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

# 读入显示
img = cv2.imread('D:\learning_materials\Second_grade\AI_for_NLP&CV_Train\courseware\Week1\\1.jpg')
#BGR转YUV显示
img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
#分离
Y,U,V = cv2.split(img_yuv)
#随机变化颜色？？
Y = Y+np.random.rand(Y.shape)
#通道合并
img_yuv = cv2.merge([Y,U,V])
#YUV转回BGR空间
img_yuv = cv2.cvtColor(img_yuv,cv2.COLOUR_BGR2YUV)


cv2.imshow('img_yuv', img_yuv)
