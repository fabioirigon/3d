#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 06:50:41 2023

@author: fip
"""

import cv2
import numpy as np


def vecs2P(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    P = np.hstack((R, tvec.reshape(3, 1)))
    return P

def invRT(R, T):
    RI = np.linalg.inv(R)
    TI = np.matmul(RI, T.reshape(3, 1))*(-1)
    RTI = np.hstack((RI, TI))
    return RTI

def invP(P):
    RI = np.linalg.inv(P[:, :3])
    TI = np.matmul(RI, P[:, 3].reshape(3, 1))*(-1)
    PI = np.hstack((RI, TI))
    return PI

def invRvecTvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    #RT = np.hstack(((R, tvec.reshape(3, 1))))

    RI = np.linalg.inv(R)
    TI = np.matmul(RI, tvec.reshape(3, 1))*(-1)
    
    irvec, _ = cv2.Rodrigues(RI)
    return irvec.flatten(), TI.flatten()


def composeRvecTvec(rvec0, tvec0, rvec1, tvec1):
    R0, _ = cv2.Rodrigues(rvec0)
    R1, _ = cv2.Rodrigues(rvec1)

    R2 = np.matmul(R0, R1)
    T2 = np.matmul(R1, tvec0.reshape(3, 1))
    tvec2 = T2.flatten() + tvec1
    
    rvec2, _ = cv2.Rodrigues(R2)
    return rvec2, tvec2

def composeP(P0, P1):
    R2 = np.matmul(P0[:, :3], P1[:, :3])
    T2 = np.matmul(P1[:, :3], P0[:, 3].reshape(3, 1))
    T2 = T2.flatten() + P1[:, 3]
    P2 = np.hstack((R2, T2.reshape(3, 1)))
    return P2

def ptsToWld(u, v, z, A):
    fx, fy, cx, cy = A[0, 0], A[1, 1], A[0, 2], A[1, 2]
    x = (u-cx)*z/fx
    y = (v-cy)*z/fy
    return x, y

#P = P0
def projPts(X, Y, Z, P, A):
    W = np.vstack((X, Y, Z, np.ones(len(X))))
    W.shape
    nP = np.matmul(A, P)
    nP.shape
    uvw = np.matmul(nP, W)
    uvw = uvw/uvw[2, None]
    return uvw[0], uvw[1]

def fastTri(P):
    #k = matmul(R, uv0)
    #u1 = (kx*d+tx)/(kz*d + tz)
    #v1 = (ky*d+ty)/(kz*d + tz)
    #num = (tx-u1*tz)*(kz*tx-kx*tz)+(ty-v1*tz)*(kz*ty-ky*tz)
    #den = (kx-kz*u1)(kx*tz-kz*tx)+(ky-kz*v1)*(ky*tz-kz*ty)
    pass

def reprErr(pts0, pts1):
    err = pts1-pts0
    err = err*err
    err = err[:, 0]+err[:, 1]
    return err**0.5
    
# def camToWldPts(rvec, tvec, A):
    
#     imgPts = np.ones((3, 5))
#     imgPts[:, 0] = cx, cy, 0
#     imgPts[:, 1] = 0, 0, 1
#     imgPts[:, 2] = HRES, 0, 1
#     imgPts[:, 3] = HRES, VRES, 1
#     imgPts[:, 4] = 0, VRES, 1

#     imgPts = imgPts*10

#     AI = np.linalg.inv(A)
#     wPts = np.matmul(AI, imgPts)
#     wPts.shape

#     irvec, itvec = invRvecTvec(rvec, tvec)
#     R, _ = cv2.Rodrigues(irvec)
#     RT = np.hstack(((R, itvec.reshape((3, 1)))))
#     w = np.vstack((wPts, np.ones(5)))
#     RT.shape, w.shape
#     nw = np.matmul(RT, w)
#     return nw

