#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 06:50:41 2023

@author: fip
"""

import cv2
import numpy as np


def invRT(R, T):
    RI = np.linalg.inv(R)
    TI = np.matmul(RI, T.reshape(3, 1))*(-1)
    RTI = np.hstack((RI, TI))
    return RTI

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

