#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:20:08 2023

@author: fip
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from corners import sort_corners

HRES, VRES = 640, 480
SQR_H, SQR_W = 18, 18

with open('frames.npy', 'rb') as f:
    frames = np.load(f)
with open('corners.npy', 'rb') as f:
    corner_lst = np.load(f)



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


obPts = np.zeros((len(frames), 81, 3), dtype=np.float32)
imPts = np.zeros((len(frames), 81, 2), dtype=np.float32)

for i in range(len(frames)):
    obPts[i] = pts_w
    imPts[i] = sort_corners(corner_lst[i]) #[:, ::-1]

imSz = HRES, VRES
r = cv2.calibrateCamera(obPts, imPts, imSz, A, None)
rval, camMat, distCoefs, rvecs, tvecs = r

print(rval)    
    
if 0:

    idx = 4
    obPts[idx][0]
    obPts[idx][1]
    obPts[idx][-1]
    
    imPts[idx][0]
    imPts[idx][2]
    imPts[idx][-1]

    imgc = cv2.cvtColor(frames[idx], cv2.COLOR_GRAY2RGB)
    #pts = imPts[idx].astype(int)
    pts = corner_lst[idx].astype(int)
    for k, (x, y) in enumerate(pts):
        cv2.circle(imgc, (x, y), 3, (255, 255, 0), 2)
        if k>0:
            x0, y0 = pts[k-1]
            cv2.line(imgc, (x0, y0), (x, y), (0, 255, 0), 2)
    plt.figure();plt.imshow(imgc)
    
    
    ymin, ymax, xmin = pts[:, 1].min(), pts[:, 1].max(), pts[:, 0].min()
    d_topLeft = np.abs(pts[:, 0]-xmin) + np.abs(pts[:, 1]-ymin)
    d_botLeft = np.abs(pts[:, 0]-xmin) + np.abs(pts[:, 1]-ymax)
    tl, bl = pts[np.argmin(d_topLeft)], pts[np.argmin(d_botLeft)]
    
    srt = [pts[np.argmin(d_topLeft)]]
    dx, dy = (bl-tl)/8
    for i in range(9):
        for j in range(8):
            x0, y0 = srt[-1]
            x1, y1 = x0+dx, y0+dy
            pt_idx = np.argmin(np.abs(pts[:, 0]-x1) + np.abs(pts[:, 1]-y1))
            srt.append(pt_idx)
            print(pts[pt_idx])
        if i < 8:
            x0, y0 = pts[srt[-9]]
            x1, y1 = x0+dy, y0-dx
            pt_idx = np.argmin(np.abs(pts[:, 0]-x1) + np.abs(pts[:, 1]-y1))
            srt.append(pt_idx)
            print(pts[pt_idx])
    pts_s = pts[srt]
    len(srt)

    plt.close('all')
    plt.plot(pts_s[:, 0])
    plt.plot(pts_s[:, 1])

    x = sort_corners(corner_lst[idx])
    for i in range(10):
        plt.figure();
        plt.plot(imPts[i, :, 0])
        plt.plot(imPts[i, :, 1])

    if 0:
        impts = corners.astype(np.float32)
        ret, rvec, tvec = cv2.solvePnP(pts_w, impts, A, None)


    
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