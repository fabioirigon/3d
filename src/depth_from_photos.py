#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 08:32:08 2023

@author: fip
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import proj_tools as prt

plt.close('all')

A = np.array([[565.37877495,   0.        , 321.32942885],
              [  0.        , 565.48666909, 181.99343463],
              [  0.        ,   0.        ,   1.        ]])

distCoefs_97f = np.array([[ 2.22609859e-01, -1.59712209e+00, -1.58796203e-03, 6.81037555e-04,  4.74766986e+00]])

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 0.05))

def getCorrespondences(g0, g1):

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 500,
                           qualityLevel = 0.1,
                           minDistance = 10,
                           blockSize = 7)

    p0 = cv2.goodFeaturesToTrack(g0, mask = None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk_params)
    
    if p1 is None: 
        print('track error')
    
    p1g = p1[st==1]
    p0g = p0[st==1]
    print('lens: ', len(p0), len(p0g))
    return p0g, p1g

def getCorrFast(g0, g1):
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()
    # find and draw the keypoints
    kp = fast.detect(g0, None)
    len(kp)
    if 0:
        img2 = cv2.drawKeypoints(g0, kp, None, color=(255,0,0))
        plt.figure(30); plt.imshow(img2)
    dir(kp[0])
    kp[0].pt
    p0 = np.array([kp[i].pt for i in range(len(kp ))], dtype=np.float32)
    p0.shape, p0.dtype

    p1, st, err = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk_params)
    p1.shape, p1.dtype, st.shape


    p1g = p1[st[:, 0]==1]
    p0g = p0[st[:, 0]==1]
    print('lens: ', len(p0), len(p0g))
    return p0g, p1g

    
f0_bkp = cv2.imread("../imgs/sala_5.jpeg")
f1_bkp = cv2.imread("../imgs/sala_4.jpeg")
f0_bkp = cv2.resize(f0_bkp, (640, 360))
f1_bkp = cv2.resize(f1_bkp, (640, 360))
print(f0_bkp.shape, f1_bkp.shape)

plt.close('all')
plt.figure(1)
plt.subplot(121)
plt.imshow(f0_bkp)
plt.subplot(122)
plt.imshow(f1_bkp)


g0 = cv2.cvtColor(f0_bkp, cv2.COLOR_BGR2GRAY)
g1 = cv2.cvtColor(f1_bkp, cv2.COLOR_BGR2GRAY)

if 0:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g0 = clahe.apply(g0)
    g1 = clahe.apply(g1)

p0g, p1g = getCorrespondences(g0, g1)
#p0g, p1g = getCorrFast(g0, g1)


c0 = cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)
c1 = cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)
# draw the tracks
for (new, old) in zip(p0g, p1g):
    a, b = new.ravel()
    c, d = old.ravel()
    cv2.line(c1, (int(a), int(b)), (int(c), int(d)), (255,0,0), 1)
    cv2.circle(c1, (int(c), int(d)), 5, (0,255,0), 2)
    cv2.circle(c0, (int(a), int(b)), 5, (0,255,0), 2)
plt.figure(10); plt.imshow(c1)
plt.figure(9); plt.imshow(c0)

# set a constant depth
Z = np.ones(len(p0g))*400
X, Y = prt.ptsToWld(p0g[:, 0], p0g[:, 1], Z, A)
wpts = np.vstack((X, Y, Z)).T
wpts = wpts.astype(np.float32)
wpts.shape, p1g.shape, wpts.dtype, p1g.dtype

# estimate cam pos
retv, rvec, tvec, inliers = cv2.solvePnPRansac(wpts, p1g, A, None)
print(retv, len(inliers))

# PROJECT AND PLOT POINTS
P_0_1 = prt.vecs2P(rvec, tvec)
P = prt.vecs2P(np.zeros(3), np.zeros(3))

projx, projy = prt.projPts(X, Y, Z, P_0_1, A)
p1r = np.vstack((projx, projy)).T
err=prt.reprErr(p1g, p1r)
print("REP: ", np.median(err))

c1 = cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)
for i in range(len(p1g)):
    a, b = int(p1g[i, 0]), int(p1g[i, 1])
    c, d = int(projx[i]), int(projy[i])
    cv2.circle(c1, (a, b), 5, (0,255,0), 2)
    cv2.circle(c1, (c, d), 4, (0,0,255), 2)
plt.figure(11); plt.imshow(c1)

# TRIANGULATE
p0g_rs = p0g.reshape((-1, 1, 2))
p1g_rs = p1g.reshape((-1, 1, 2))
nP = np.matmul(A, P)
nP_0_1 = np.matmul(A, P_0_1)

points3d = cv2.triangulatePoints(nP, nP_0_1, p0g_rs, p1g_rs)
points3d.shape
p3 = points3d[:3] / points3d[3, None] 

# REPROJECT TRIANGULATED POINTS
projx, projy = prt.projPts(p3[0, :], p3[1, :], p3[2, :], P_0_1, A)
p1r = np.vstack((projx, projy)).T
err = prt.reprErr(p1g, p1r)
print("REP: ", np.median(err))

for i in range(len(p1g)):
    c, d = int(projx[i]), int(projy[i])
    cv2.circle(c1, (c, d), 4, (255,0,0), 2)
plt.figure(12); plt.imshow(c1)

for kk in range(4):
    # ESTIMATE CAM POS:
    thresh = np.median(err)
    p3f, p1gf = p3[:, err<thresh].copy(), p1g[err<thresh].copy()
    #p3f.shape, p3.shape
    retv, rvec, tvec, inliers = cv2.solvePnPRansac(p3f.T, p1gf, A, None)
    P_0_1 = prt.vecs2P(rvec, tvec)
    nP_0_1 = np.matmul(A, P_0_1)
    
    # SHOW ERROR:
    projx, projy = prt.projPts(p3[0, :], p3[1, :], p3[2, :], P_0_1, A)
    p1r = np.vstack((projx, projy)).T
    print("REP PnP: ", np.median(prt.reprErr(p1g, p1r)))

    #TRIANGULATE
    points3d = cv2.triangulatePoints(nP, nP_0_1, p0g_rs, p1g_rs)
    p3 = points3d[:3] / points3d[3, None] 

    # SHOW ERROR:
    projx, projy = prt.projPts(p3[0, :], p3[1, :], p3[2, :], P_0_1, A)
    p1r = np.vstack((projx, projy)).T
    err = prt.reprErr(p1g, p1r)
    print("REP Tri: ", np.median(err))

# plot depth
err = prt.reprErr(p1g, p1r)
plt.figure(19), plt.plot(err)

z = p3[2, :].copy()
plt.figure(20), plt.plot(z)
thr = np.median(z)
print('thr: ', thr)
z[z<thr*0.7] = thr*0.7
z[z>thr*1.4] = thr*1.4
z = (255*(z-z.min())/(z.max()-z.min()))
z = z.astype(np.uint8)
z = z.ravel()
z.shape, np.median(z)

c1 = cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)
for i in range(len(p0g)):
    if err[i]>np.median(err)*1.5:
        continue
    a, b = int(p0g[i, 0]), int(p0g[i, 1])
    c = int(z[i])
    cv2.circle(c1, (a, b), 5, (c,255-c,0), 2)
plt.figure(15); plt.imshow(c1)


if 0:
    P0, P1 = prt.vecs2P(rvec0, tvec0), prt.vecs2P(rvec1, tvec1)
    P_0_1 = prt.composeP(prt.invP(P0) , P1)
    P = prt.vecs2P(np.zeros(3), np.zeros(3))
    
    
    P0, P1 = prt.vecs2P(rvec0, tvec0), prt.vecs2P(rvec1, tvec1)
    P_0_1 = prt.composeP(prt.invP(P0) , P1)
    P = prt.vecs2P(np.zeros(3), np.zeros(3))
    
    u0, v0 = prt.projPts(X, Y, Z, P, A)
    u1, v1 = prt.projPts(X, Y, Z, P_0_1, A)
    
    uv0 = np.vstack((u0, v0)).T
    
    
    if 1:
        uv0_u = cv2.undistortPoints(uv0, A, distCoefs_97f)
        uv1f_u = cv2.undistortPoints(uv1f, A, distCoefs_97f)
        uv1a_u = cv2.undistortPoints(uv1a, A, distCoefs_97f)
    
        uv0_s = uv0.reshape((-1, 1, 2))
        uv1f_s = uv1f.reshape((-1, 1, 2))
        uv1a_s = uv1a.reshape((-1, 1, 2))
    
        nP = np.matmul(A, P)
        nP_0_1 = np.matmul(A, P_0_1)
    
        #points3d = cv2.triangulatePoints(nP, nP_0_1, uv0_s, uv1a_s)
        #points3d = cv2.triangulatePoints(nP_0_1, nP, uv1a_s, uv0_s)
        points3d = cv2.triangulatePoints(nP_0_1, nP, uv1f_s, uv0_s)
        #points3d[:, :5]
    
        #points3d = cv2.triangulatePoints(P, P_0_1, uv0_u, uv1f_u)
        #points3d = cv2.triangulatePoints(P0, P1, uv0_u, uv1f_u)
        p3 = points3d[:3] / points3d[3, None] 
        #p3[:, :5]
        
        #recovered depth
        p3.shape
        rd = p3[2, :]
        print(rd.max(), rd.min(), np.median(rd))
        rd[rd>2*np.median(rd)] = 2*np.median(rd)
        rd[rd<500] = 500
        
        col = 255*(rd-rd.min())/(rd.max()-rd.min())
        col = col.astype(int)
        print(col.max(), col.min(), np.median(col))
        
        xy = uv0.astype(int)
        imgc = cv2.cvtColor(g0, cv2.COLOR_GRAY2RGB)
        for i, (u, v) in enumerate(xy):
            color_ = np.array([col[i], 255-col[i], 0], dtype=np.int32)
            #color_ = np.array([50, 205, 0], dtype=np.int32)
            color_ = (50, 205, 0)
            color_ = (col[i], 0, 0)
            color_ = ((np.asscalar(col[i])),np.asscalar(255-col[i]),0)#HERE
            #cv2.rectangle(imgc, (u-10, v-10), (u+10, v+10), (50, 205, 0))
            cv2.rectangle(imgc, (u-5, v-5), (u+5, v+5), color_)
            #cv2.circle(imgc, (u, v), 4, (col[i], 255-col[i], 0), 2)
    
        plt.figure(50); plt.imshow(imgc)
       
