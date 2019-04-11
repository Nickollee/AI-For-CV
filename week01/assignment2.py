'''Assignments:2.Change image through YUV space.'''

import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

# 读入显示
img = cv2.imread('D:\learning_materials\Second_grade\AI_for_NLP&CV_Train\courseware\CV\week2\\flower.png')

def random_yuv_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(img_yuv)

    y_rand = random.randint(-50, 50)
    if y_rand == 0:
        pass
    elif y_rand > 0:
        lim = 255 - y_rand
        Y[Y > lim] = 255
        Y[Y <= lim] = (y_rand + Y[Y <= lim]).astype(img_yuv.dtype)
    elif y_rand < 0:
        lim = 0 < y_rand
        Y[Y < lim] = 0
        Y[Y >= lim] = (y_rand + Y[Y >= lim]).astype(img_yuv.dtype)

    u_rand = random.randint(-50, 50)
    if u_rand == 0:
        pass
    elif u_rand > 0:
        lim = 255 - u_rand
        U[U > lim] = 255
        U[U <= lim] = (u_rand + U[U <= lim]).astype(img_yuv.dtype)
    elif u_rand < 0:
        lim = 0 < u_rand
        U[U < lim] = 0
        U[U >= lim] = (u_rand + U[U >= lim]).astype(img_yuv.dtype)
    v_rand = random.randint(-50, 50)
    if v_rand == 0:
        pass
    elif v_rand > 0:
        lim = 255 - v_rand
        V[V > lim] = 255
        V[V <= lim] = (v_rand + V[V <= lim]).astype(img_yuv.dtype)
    elif v_rand < 0:
        lim = 0 < v_rand
        V[V < lim] = 0
        V[V >= lim] = (v_rand + V[V >= lim]).astype(img_yuv.dtype)

    img_merge = cv2.merge((Y, U, V))
    return cv2.cvtColor(img_merge, cv2.COLOR_YUV2BGR)



img_random_yuv = random_yuv_color(img)
cv2.imshow('img_random_yuv', img_random_yuv)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
