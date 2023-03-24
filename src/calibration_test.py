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



if 1:
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


if 0:

    for i in range(5):
        frames = np.random.randint(0, totalFrames-1, 30)
        print(frames)

    fnums = list(range(0, totalFrames-1, 100))
    fnum = fnums[0]

    corners = corners[:, :, :, ::-1]

    pts_w_pos = np.zeros((9, 9, 3))
    for i in range(9):
        for j in range(9):
            pts_w_pos[i, j, 0] = (i-4)*SQR_H
            pts_w_pos[i, j, 1] = (j-4)*SQR_W
    

    cx, cy = fr.shape[1]//2, fr.shape[0]//2
    fx, fy = 300, 300
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    tvec[2] = 300
    
    A = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float32)        

    ret, rvec, tvec = cv2.solvePnP(pts_w, imPts, A, None)

    obPts = np.zeros((len(corn_lst), 81, 3), dtype=np.float32)
    imPts = np.zeros((len(corn_lst), 81, 2), dtype=np.float32)
    
    for i, fridx in enumerate(corn_lst):
        obPts[i] = pts_w
        imPts[i] = corners[fridx].reshape((-1, 2))

    h, w, _ = fr.shape
    imSz = (w, h)
    
    #rval, camMat, rvecs, tvecs = cv2.calibrateCamera(obPts, imPts, imSz, A, None)
    r = cv2.calibrateCamera(obPts, imPts, imSz, A, None)

    rval, camMat, distCoefs, rvecs, tvecs = r


if __name__ == "__main__" and False:
    pass