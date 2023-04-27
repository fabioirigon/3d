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

def searchEpLine(im0, im1, pt0, pt1a, pt1b, hSz):

    x, y = int(pt0[0]), int(pt0[1])
    fig0 = im0[y-hSz: y+hSz, x-hSz:x+hSz].astype(np.float32)
    sz = 2*hSz
    x, y = int(pt1a[0]), int(pt1a[1])
    fig1 = im1[y-sz: y+sz, x-sz:x+sz].astype(np.float32)
    du, dv = pt1a[0] - pt1b[0], pt1a[1] - pt1b[1]
    da = (du*du + dv*dv)**0.5
    du, dv = du/da, dv/da

    tx, ty = -hSz, -hSz
    for k in range(10):
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        figc = cv2.warpAffine(fig1, M, (sz, sz))
        
        M = np.array([[1, 0, tx+du], [0, 1, ty+dv]], dtype=np.float32)
        figd = cv2.warpAffine(fig1, M, (sz, sz))
        
        r = figc-fig0
        r0 = figd-fig0
        d = r0-r
        st = d.ravel()/np.sum(d*d)
        st = np.sum(st*r.ravel())
        tx, ty = tx-st*du, ty-st*dv, 
        print(st, np.mean(np.abs(r)))
        if abs(st)<0.2:
            break
    return tx+hSz, ty+hSz

def imgGrad(img, pt, hSz):
    x, y = int(pt0[0]), int(pt0[1])
    crop = img[y-hSz: y+hSz, x-hSz:x+hSz].astype(np.float32)
    cropx = img[y-hSz: y+hSz, x-hSz+1:x+hSz+1].astype(np.float32)
    cropy = img[y-hSz+1: y+hSz+1, x-hSz:x+hSz].astype(np.float32)
    dx = np.mean(crop-cropx)
    dy = np.mean(crop-cropy)
    da = (dx*dx + dy*dy)**0.5
    return dx/da, dy/da, da
    

    
plt.close('all')
plt.figure(1)
plt.subplot(121)
plt.imshow(f0_bkp)
plt.subplot(122)
plt.imshow(f1_bkp)

g0 = cv2.cvtColor(f0_bkp, cv2.COLOR_RGB2GRAY)
g1 = cv2.cvtColor(f1_bkp, cv2.COLOR_RGB2GRAY)
f0 = f0_bkp.copy()
f1 = f1_bkp.copy()

if 0:
    g0f, g1f = g0.astype(float), g1.astype(float)
    diff = np.abs(g0f-g1f)
    diff.shape, diff.max(), diff.min()
    plt.figure(100)
    plt.imshow(diff.astype(np.uint8), cmap='gray')
    plt.figure(101)
    plt.imshow(diff, cmap='gray')


edges = cv2.Canny(g0,100,200)
py, px = np.where(edges == np.max(edges))
print(len(px))
pos = list(range(0, len(px), len(px)//100))
py, px  = py[pos], px[pos]

Z = np.ones(len(px))*400
X, Y = prt.ptsToWld(px, py, Z, A)

P0, P1 = prt.vecs2P(rvec0, tvec0), prt.vecs2P(rvec1, tvec1)
P_0_1 = prt.composeP(prt.invP(P0) , P1)
P = prt.vecs2P(np.zeros(3), np.zeros(3))

u0, v0 = prt.projPts(X, Y, Z, P, A)
u1, v1 = prt.projPts(X, Y, Z, P_0_1, A)
u2, v2 = prt.projPts(X, Y, Z*0.9, P_0_1, A)
u3, v3 = prt.projPts(X, Y, Z*0.8, P_0_1, A)

pt_idx = 3

cv2.circle(f0, (int(u0[pt_idx]), int(v0[pt_idx])), 4, (255, 0, 0))
cv2.circle(f1, (int(u1[pt_idx]), int(v1[pt_idx])), 4, (255, 0, 0))
cv2.circle(f1, (int(u2[pt_idx]), int(v2[pt_idx])), 4, (255, 255, 0))
cv2.circle(f1, (int(u3[pt_idx]), int(v3[pt_idx])), 4, (255, 0, 255))

plt.close('all')
plt.figure(3)
plt.subplot(121)
plt.imshow(f0)
plt.subplot(122)
plt.imshow(f1)


if 1:
    x, y = int(u0[3]), int(v0[3])
    fig0 = g0[y-10: y+10, x-10:x+10].astype(np.float32)
    x, y = int(u1[3]), int(v1[3])
    print(x, y)
    fig1 = g1[y-20: y+20, x-20:x+20].astype(np.float32)
    du, dv = u1[3] - u2[3], v1[3] - v2[3]
    da = (du*du + dv*dv)**0.5
    du, dv = du/da, dv/da
    print('dudv: ', du, dv)
    print('fg: ', np.mean(fig0), np.mean(fig1))
    
    print("test")
    
    tx, ty = -10, -10
    for k in range(10):
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        figc = cv2.warpAffine(fig1, M, (20, 20))
        
        M = np.array([[1, 0, tx+du], [0, 1, ty+dv]], dtype=np.float32)
        figd = cv2.warpAffine(fig1, M, (20, 20))
        
        r = figc-fig0
        r0 = figd-fig0
        d = r0-r
        st = d.ravel()/np.sum(d*d)
        st = np.sum(st*r.ravel())
        tx, ty = tx-st*du, ty-st*dv, 
        print(st, np.mean(np.abs(r)))
        if abs(st)<0.2:
            break
    
    print("\nsrch")
    #pt0, pt1a, pt1b = (int(u0[3]), int(v0[3])), (int(u1[3]), int(v1[3])), (int(u2[3]), int(v2[3]))
    pt0, pt1a, pt1b = (u0[3], v0[3]), (u1[3],v1[3]), (u2[3],v2[3])
    searchEpLine(g0, g1, pt0, pt1a, pt1b, 10)

dx, dy, da = imgGrad(g0, pt0, 8)
