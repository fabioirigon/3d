#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:01:10 2023

@author: fip
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from corners import getCornersFromImg

A = np.array([[565.37877495,   0.        , 321.32942885],
              [  0.        , 565.48666909, 181.99343463],
              [  0.        ,   0.        ,   1.        ]])

distCoefs_97f = np.array([[ 2.22609859e-01, -1.59712209e+00, -1.58796203e-03, 6.81037555e-04,  4.74766986e+00]])
SQR_H, SQR_W = 19.7, 19.7



fname = '../vids/checker_2.mp4'
cap = cv2.VideoCapture(fname)
plt.close('all')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 300,
                       qualityLevel = 0.05,
                       minDistance = 45,
                       blockSize = 10 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap.set(cv2.CAP_PROP_POS_FRAMES,800)
ret, frame = cap.read()
corners0, gray0 = getCornersFromImg(frame)
frame0 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2RGB)
f0_bkp = frame0.copy()

cap.set(cv2.CAP_PROP_POS_FRAMES,830)
ret, frame = cap.read()
corners1, gray1 = getCornersFromImg(frame)
frame1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2RGB)
f1_bkp = frame1.copy()


for pt in corners0:
    cv2.circle(frame0, pt.astype(int), 6, (255,0,0), 4)
for pt in corners1:
    cv2.circle(frame1, pt.astype(int), 6, (255,0,0), 4)

plt.figure(1)
plt.subplot(121)
plt.imshow(frame0)
plt.subplot(122)
plt.imshow(frame1)


p0 = cv2.goodFeaturesToTrack(gray0, mask = None, **feature_params)
p1, st, err = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk_params)
p1g = p1[st==1]
p0g = p0[st==1]

for i, (new, old) in enumerate(zip(p1g, p0g)):
    a, b = new.ravel()
    c, d = old.ravel()
    frame0 = cv2.line(frame0, (int(a), int(b)), (int(c), int(d)), (255,0,0), 2)
    frame0 = cv2.circle(frame0, (int(c), int(d)), 5, (0,0,255), -1)
    frame1 = cv2.circle(frame1, (int(a), int(b)), 5, (0,0,255), -1)

plt.figure(2)
plt.subplot(121)
plt.imshow(frame0)
plt.subplot(122)
plt.imshow(frame1)

HRES, VRES = 640, 360

pts_w = np.zeros((81, 3))
for i in range(81):
    pts_w[i, 0] = (i//9-4)*SQR_H
    pts_w[i, 1] = (i%9-4)*SQR_W


ret0, rvec0, tvec0 = cv2.solvePnP(pts_w, corners0, A, None)
ret1, rvec1, tvec1 = cv2.solvePnP(pts_w, corners1, A, None)


projM0 = np.hstack((cv2.Rodrigues(rvec0)[0], tvec0))
projM1 = np.hstack((cv2.Rodrigues(rvec1)[0], tvec1))

p0g_u = cv2.undistortPoints(p0g, A, distCoefs_97f)
p1g_u = cv2.undistortPoints(p1g, A, distCoefs_97f)

points3d = cv2.triangulatePoints(projM0, projM1, p0g_u, p1g_u)

p3 = points3d[:3] / points3d[3, None] 
p3.shape

ppts, jc = cv2.projectPoints(p3, rvec0, tvec0, A, distCoefs_97f)
ppts.shape
ppts = ppts[:,0,:]

ppts[0]

plt.figure(3)
for pt in ppts[14:18]:
    cv2.circle(frame0, pt.astype(int), 3, (0,255,255), 2)
plt.imshow(frame0)

trackWpts = np.array([[   4.4759927,  305.8374   , -214.79953  ,  212.62338  ],
                       [-382.79422  , -113.78936  , -142.68068  , -142.18954  ],
                       [ 239.43945  ,   92.293106 ,  -10.213252 ,  -13.1673565]],
                      dtype=np.float32)


if 0:
    plt.close('all')
    fnums = np.arange(850,950,10)
    frnum = 100
    for frnum in fnums:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frnum)
        ret, frame = cap.read()
        corners0, gray0 = getCornersFromImg(frame)
        frame0 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2RGB)
    
        ret0, rvec0, tvec0 = cv2.solvePnP(pts_w, corners0, A, None)
        ppts0, jc0 = cv2.projectPoints(trackWpts, rvec0, tvec0, A, distCoefs_97f)

        ppts0 = ppts0[:,0,:]
        plt.figure()
        for pt in ppts0:
            cv2.circle(frame0, pt.astype(int), 6, (255,0,255), 4)
        plt.imshow(frame0)
