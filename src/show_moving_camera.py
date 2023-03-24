#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 06:50:41 2023

@author: fip
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

HRES, VRES = 460, 340
SQR_H, SQR_W = 19.7, 19.7
K_LEFT = 81
K_RIGHT = 83
K_UP = 82
K_DOWN = 84
K_SPACE = 32

def invRT(RT):
    RI = np.linalg.inv(R)
    TI = np.matmul(RI, tvec.reshape(3, 1))*(-1)
    RTI = np.hstack((RI, TI))
    return RTI

def invRvecTvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    #RT = np.hstack(((R, tvec.reshape(3, 1))))

    RI = np.linalg.inv(R)
    TI = np.matmul(RI, tvec.reshape(3, 1))*(-1)
    
    irvec, _ = cv2.Rodrigues(RI)
    return irvec.flatten(), TI.flatten()

def camToWldPts(rvec, tvec, A):
    
    imgPts = np.ones((3, 5))
    imgPts[:, 0] = cx, cy, 0
    imgPts[:, 1] = 0, 0, 1
    imgPts[:, 2] = HRES, 0, 1
    imgPts[:, 3] = HRES, VRES, 1
    imgPts[:, 4] = 0, VRES, 1

    imgPts = imgPts*10

    AI = np.linalg.inv(A)
    wPts = np.matmul(AI, imgPts)
    wPts.shape

    irvec, itvec = invRvecTvec(rvec, tvec)
    R, _ = cv2.Rodrigues(irvec)
    RT = np.hstack(((R, itvec.reshape((3, 1)))))
    w = np.vstack((wPts, np.ones(5)))
    RT.shape, w.shape
    nw = np.matmul(RT, w)
    return nw

#camPts = wPts
def drawCamPts(camPts, rvec, tvec, A, fr):
    projPts, jac = cv2.projectPoints(camPts, rvec, tvec, A, None)
    projPts = projPts.reshape((-1, 2)).astype(int)
    
    if 1:
        for i in range(1,5):
            cv2.line(fr, projPts[0], projPts[i], (255, 0, 0), 1)
            if i<4:
                cv2.line(fr, projPts[i], projPts[i+1], (255, 0, 0), 1)
            else:
                cv2.line(fr, projPts[i], projPts[1], (255, 0, 0), 1)
    else:
        cnts = []        
        for i in range(1,5):
            j  = i+1 if i<4 else (i+1) % 5 + 1
            cnt = np.array( [projPts[0], projPts[i], projPts[j]] )
            cnts.append(cnt)
        cv2.drawContours(fr, cnts, 0, (0,255,0), -1)

    return fr

        
pts_w_pos = np.zeros((9, 9, 3))
for i in range(9):
    for j in range(9):
        pts_w_pos[i, j, 0] = (i-4)*SQR_H
        pts_w_pos[i, j, 1] = (j-4)*SQR_W

#cx, cy = fr.shape[1]//2, fr.shape[0]//2
cx, cy = HRES//2, VRES//2
fx, fy = 300, 300
rvec = np.zeros(3)
tvec = np.zeros(3)
tvec[2] = 300

A = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float32)

pts_w = pts_w_pos.reshape((-1, 3))
zp = np.array([0, 0, 100], dtype=np.float32)

projPts, jac = cv2.projectPoints(pts_w, rvec, tvec, A, None)
projPts = projPts.reshape((-1, 2))
projPts = projPts.astype(int)

fr0 = np.zeros((HRES, VRES, 3), dtype=np.uint8)
for pt in projPts:
    cv2.circle(fr0, pt, 6, (0, 255, 0), 2)

camPts = []

while 1:

    cv2.imshow("Display window", fr0)
    k = cv2.waitKey(0)
    if k == ord("q"):
        break
    if k == ord("w"):
        tvec[1] += 10 
    if k == ord("s"):
        tvec[1] -= 10 
    if k == ord("a"):
        tvec[0] -= 10 
    if k == ord("d"):
        tvec[0] += 10
    if k == ord("z"):
        tvec[2] -= 10 
    if k == ord("x"):
        tvec[2] += 10
    if k == ord("i"):
        rvec[1] += 0.1
    if k == ord("k"):
        rvec[1] -= 0.1
    if k == ord("l"):
        rvec[0] += 0.1
    if k == ord("j"):
        rvec[0] -= 0.1
    if k == K_SPACE:
        camPts.append(camToWldPts(rvec, tvec, A))
        print('point added')

    print(k)

    projPts, jac = cv2.projectPoints(pts_w, rvec, tvec, A, None)
    projPts = projPts.reshape((-1, 2))
    projPts = projPts.astype(int)

    pp, jac = cv2.projectPoints(zp, rvec, tvec, A, None)
    pp = pp.reshape((2)).astype(int)

    
    fr0 = np.zeros((460, 340, 3), dtype=np.uint8)
    for pt in projPts:
        cv2.circle(fr0, pt, 6, (0, 255, 0), 2)
    cv2.circle(fr0, pp, 6, (255, 0, 0), 2)
    
    if len(camPts) > 0:
        fr0 = drawCamPts(camPts[0], rvec, tvec, A, fr0)
        
    print('rvec:', rvec)
    print('tvec:', tvec)
    
#rvec: [1.2 0.  0. ]
#tvec: [  0. -80. 300.]
    
cv2.destroyAllWindows()

if 0:
    pts_w.shape
    projPts.shape
    projPts[0]
    wpt = np.ones((4, 1))
    wpt[:3, 0] = pts_w[0]
    
    tvec
    R, _ = cv2.Rodrigues(rvec)
    RT = np.hstack(((R, tvec.reshape((3, 1)))))
    
    nwpt = np.matmul(RT, wpt)
    nproj = np.matmul(A, nwpt)
    nproj / nproj[2]
    
    wpt2 = pts_w[0].reshape(3, 1)
    nwpt1 = np.matmul(R, wpt2) + tvec.reshape(3, 1)
    
    RI = np.linalg.inv(R)
    TI = np.matmul(RI, tvec.reshape(3, 1))*(-1)
    RTI = np.hstack((RI, TI))
    nwpt2 = np.ones((4, 1))
    nwpt2[:3] = nwpt
    oldPt = np.matmul(RTI, nwpt2)
    
    irvec, _ = cv2.Rodrigues(RI)
    