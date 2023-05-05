#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 07:37:55 2023

@author: fip
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from corners import sort_corners

import proj_tools as prt

#imp, objp, rv, tv = imPts[0], obPts[0], rvecs[0], tvecs[0]
def getReprErr(params):
    imp, objp, rv, tv, A = params
    
    P = prt.vecs2P(rv, tv)
    prx, pry = prt.projPts(objp[:, 0], objp[:, 1], objp[:, 2], P, A)
    pr = np.stack((prx, pry), axis=1)
    return pr-imp


def levMarOpt(imp, objp, A):
    rvi, tvi = np.zeros(3), np.zeros(3)
    tvi[2] = 1
    lamb = 0.05

    for i in range(20):
        params = imp, objp, rvi, tvi, A
        Jr = np.zeros((len(imp)*2, 6))
        r = getReprErr(params)
        print(np.mean(np.abs(r)))
        
        for i in range(3):
            tvi[i] += 1E-8
            params = imp, objp, rvi, tvi, A
            r0 = getReprErr(params)
            dr = r0-r
            Jr[:, i] = (dr/1E-8).flatten()
            tvi[i] -= 1E-8
    
        for i in range(3):
            rvi[i] += 1E-6
            params = imp, objp, rvi, tvi, A
            r0 = getReprErr(params)
            dr = r0-r
            Jr[:, i+3] = (dr/1E-6).flatten()
            rvi[i] -= 1E-6
    
            
        J0 = np.matmul(Jr.T, Jr)
        if 0:
            dump = np.zeros((6, 6))
            np.fill_diagonal(dump, np.diag(J0))
            J0 = J0 + dump*lamb
        J1 = np.linalg.inv(J0)
        J2 = np.matmul(J1, Jr.T)
        d =  np.matmul(J2, r.reshape((-1, 1))).flatten()
        tvi, rvi = tvi-d[0:3], rvi-d[3:6]
    
    
    return

if 1:
    HRES, VRES = 640, 480
    SQR_H, SQR_W = 18, 18
    
    plt.close('all')

    with open('frames.npy', 'rb') as f:
        frames = np.load(f)
    with open('corners.npy', 'rb') as f:
        corner_lst = np.load(f)
    
    
    # help(cv2.calcOpticalFlowPyrLK)
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
    



if __name__ == "__main__":

    pass

if 0:
    plt.close('all')
    a, b, c = 0.02, -0.3, 7
    
    x = np.linspace(-20, 40, 100)
    y = a*x*x + b*x + c
    y_n = y + np.random.normal(scale=3, size=len(x))
    
    plt.plot(x, y)
    plt.plot(x, y_n)
    
    # LEAST SQUARES
    X = np.vstack((x*x, x, np.ones(len(x))))
    X = X.T
    X.shape
    XS = np.matmul(X.T, X)
    XI = np.linalg.inv(XS)
    XA = np.matmul(XI, X.T)
    XA.shape
    b = np.matmul(XA, y_n)
    print(b)
    
    
    # GAUSS NEWTON
    print("\n Gauss - Newton")
    bh = np.ones(3)*20
    Jr = np.ones((len(x), 3))
    
    for k in range(2):
        yp = bh[0]*x*x + bh[1]*x + bh[2]
        r = yp-y_n
        for i in range(3):
            bh[i] += 1E-8
            yp = bh[0]*x*x + bh[1]*x + bh[2]
            r0 = yp-y_n
            dr = r0-r
            Jr[:, i] = dr/1E-8
            bh[i] -= 1E-8
        
        J0 = np.matmul(Jr.T, Jr)
        J1 = np.linalg.inv(J0)
        J2 = np.matmul(J1, Jr.T)
        d =  np.matmul(J2, r)
        bh = bh-d
        print(bh)
    
    
    #Levenberg–Marquardt
    print("\n Levenberg-Marquardt")
    bh = np.ones(3)*20
    Jr = np.ones((len(x), 3))
    lamb = 0.05
    
    for k in range(7):
        yp = bh[0]*x*x + bh[1]*x + bh[2]
        r = yp-y_n
        for i in range(3):
            bh[i] += 1E-8
            yp = bh[0]*x*x + bh[1]*x + bh[2]
            r0 = yp-y_n
            dr = r0-r
            Jr[:, i] = dr/1E-8
            bh[i] -= 1E-8
        
        J0 = np.matmul(Jr.T, Jr)
        dump = np.zeros((3, 3))
        np.fill_diagonal(dump, np.diag(J0))
        J0 = J0 + dump*lamb
        J1 = np.linalg.inv(J0)
        J2 = np.matmul(J1, Jr.T)
        d =  np.matmul(J2, r)
        bh = bh-d
        print(bh)
    

