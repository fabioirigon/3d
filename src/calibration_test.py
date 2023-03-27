#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:20:08 2023

@author: fip
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from corners import getCornersFromImg

fname = '../vids/checker_2.mp4'
cap = cv2.VideoCapture(fname)

totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

HRES, VRES = 640, 360
SQR_H, SQR_W = 19.7, 19.7



if 0:
    plt.close('all')
    img_num = 50
    cap.set(cv2.CAP_PROP_POS_FRAMES,img_num)
    ret, frame = cap.read()
    corners, frame = getCornersFromImg(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    print(corners.shape)

    for pt in corners:
        cv2.circle(frame, pt[::-1], 6, (255,0,0), 4)
    plt.imshow(frame)


if 1:

    for i in range(5):
        frames = np.random.randint(0, totalFrames-1, 30)

    fnums = list(range(20, totalFrames-1, 10))
    print(len(fnums))
    
    if 0:
        fnum = fnums[0]
    
        fnum = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES,fnum)
        ret, frame = cap.read()
    
        corners, frame = getCornersFromImg(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
        corners.shape
        #corners = corners[:, ::-1]

    pts_w = np.zeros((81, 3))
    for i in range(81):
        pts_w[i, 0] = (i//9-4)*SQR_H
        pts_w[i, 1] = (i%9-4)*SQR_W
    

    cx, cy = HRES//2, VRES//2
    fx, fy = 300, 300
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    tvec[2] = 300
    
    A = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float32)        

    if 0:
        impts = corners.astype(np.float32)
        ret, rvec, tvec = cv2.solvePnP(pts_w, impts, A, None)


    obPts = np.zeros((len(fnums), 81, 3), dtype=np.float32)
    imPts = np.zeros((len(fnums), 81, 2), dtype=np.float32)
    
    mustRem = []
    for i, fridx in enumerate(fnums):
        obPts[i] = pts_w
        cap.set(cv2.CAP_PROP_POS_FRAMES,fnums[i])
        ret, frame = cap.read()
        corners, frame = getCornersFromImg(frame)
        if len(corners) > 0:
            corners = corners[:, ::-1]        
            imPts[i] = corners.astype(np.float32)
        else:
            mustRem.append(i)

    
    for k in mustRem[::-1]:
        print(k, end=' ')
        imPts = np.delete(imPts, k, axis=0)
        obPts = np.delete(obPts, k, axis=0)
    print(' ')
    imPts.shape
    obPts.shape


    imSz = (HRES, VRES)
    
    #rval, camMat, rvecs, tvecs = cv2.calibrateCamera(obPts, imPts, imSz, A, None)
    r = cv2.calibrateCamera(obPts, imPts, imSz, A, None)

    rval, camMat, distCoefs, rvecs, tvecs = r

    print(camMat)
    found_30f = np.array([[570.15880905,   0.         ,318.31387203],
                         [  0.,         570.46383477, 179.380326  ],
                         [  0.,           0.,           1.        ]])

    camMat_97f = np.array([[565.37877495,   0.        , 321.32942885],
                           [  0.        , 565.48666909, 181.99343463],
                           [  0.        ,   0.        ,   1.        ]])

    distCoefs_97f = np.array([[ 2.22609859e-01, -1.59712209e+00, -1.58796203e-03, 6.81037555e-04,  4.74766986e+00]])



if __name__ == "__main__" and False:
    pass