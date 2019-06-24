# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Niki
import numpy as np

def iou(box0,box1):
    r0 = box0[3] / 2#正方体
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)): overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union
def nms(output,nms_th):
    if len(output) ==0:
        return output
    output = output[np.argsort(output[:,0])]#output是检测结果，前三列是坐标，第四列是score
    bboxes = [output[0]]
    for i in np.arange(1,len(output)):
        bbox = output[i]
        flag =1
        for j in range(len(bboxes)):
            if iou(bbox[1:5],bbox[j][1:5])>=nms_th:
                flag = -1
                break
            if flag == 1:
                bboxes.append(bbox)
        bboxes = np.asarray(bboxes,np.float32)
        return bboxes

