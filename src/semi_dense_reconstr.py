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
SQR_H, SQR_W = 19.7, 19.7

f0_bkp = cv2.imread("../meta_data/frame0.jpg")
f1_bkp = cv2.imread("../meta_data/frame1.jpg")
arr = np.loadtxt("../meta_data/camsPos.txt")
rvec0, tvec0, rvec1, tvec1 = arr.T


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
    

    plt.figure(40)
    plt.imshow(fig_ref, cmap='gray')
    plt.figure(41)
    plt.imshow(fig_a, cmap='gray')
    plt.figure(42)
    plt.imshow(fig_b, cmap='gray')

    im1c = cv2.cvtColor(im1p, cv2.COLOR_GRAY2RGB)
    cv2.circle(im1c, (int(nx), int(ny)), 4, (0, 255, 0), 2)
    plt.figure(43)
    plt.imshow(im1c, cmap='gray')


def searchSinglePt(fig_ref, img_2, x, y, dx, dy, hSz):

    tx, ty = hSz-x, hSz-y
    sz = 2*hSz
    for k in range(10):
        M = np.array([[1, 0, tx],[0,1, ty]], dtype=np.float32)
        fig_a = cv2.warpAffine(img_2, M, (sz, sz))
    
        M = np.array([[1, 0, tx+dx],[0,1, ty+dy]], dtype=np.float32)
        fig_b = cv2.warpAffine(img_2, M, (sz, sz))

        fig_a, fig_b, fig_ref = fig_a.astype(float), fig_b.astype(float), fig_ref.astype(float)
        r = fig_a-fig_ref
        r0 = fig_b-fig_ref
        d = r0-r
        st = d.ravel()/np.sum(d*d)
        st = np.sum(st*r.ravel())
        tx, ty = tx-st*dx, ty-st*dy,
        print(st, np.mean(np.abs(r)))
        if abs(st)<0.2:
            break
    return hSz-tx, hSz-ty
    

def search_epLine_pyr(im0, im1, uv0, uv1a, uv1b, hSz):
    im0p, im1p = cv2.pyrDown(im0), cv2.pyrDown(im1)
    im0pp, im1pp = cv2.pyrDown(im0p), cv2.pyrDown(im1p)
    
    brd = 40
    im0ppb = cv2.copyMakeBorder(im0pp, brd, brd, brd, brd, cv2.BORDER_REPLICATE)
    im1ppb = cv2.copyMakeBorder(im1pp, brd, brd, brd, brd, cv2.BORDER_REPLICATE)

    uv0ppb, uv1appb, uv1bppb = uv0/4+brd, uv1a/4+brd, uv1b/4+brd
    sz = hSz*2
    uv1f = uv0.copy()

    du, dv = uv1appb[:, 0] - uv1bppb[:, 0], uv1appb[:, 1] - uv1bppb[:, 1]
    da = (du*du + dv*dv)**0.5
    du, dv = du/da, dv/da

    for pt_idx in range(len(uv0)):

        M = np.array([[1, 0, hSz-uv0ppb[pt_idx][0]],[0,1, hSz-uv0ppb[pt_idx][1]]], dtype=np.float32)
        fig_ref = cv2.warpAffine(im0ppb, M, (sz, sz))
    
        nx, ny = searchSinglePt(fig_ref, im1ppb, uv1appb[pt_idx][0], uv1appb[pt_idx][1], 
                       du[pt_idx], dv[pt_idx], hSz)

        uv1f[pt_idx] = (nx-brd)*4, (ny-brd)*4

    for pt_idx in range(len(uv0)):

        M = np.array([[1, 0, hSz-uv0[pt_idx][0]],[0,1, hSz-uv0[pt_idx][1]]], dtype=np.float32)
        fig_ref = cv2.warpAffine(im0, M, (sz, sz))
    
        nx, ny = searchSinglePt(fig_ref, im1, uv1f[pt_idx][0], uv1f[pt_idx][1], 
                       du[pt_idx], dv[pt_idx], hSz)

        uv1f[pt_idx] = nx, ny

    return uv1f

def searchEpLine_2(im0, im1, uv0, uv1a, uv2a, hSz):

    im0p, im1p = cv2.pyrDown(im0), cv2.pyrDown(im1)
    uv0p, uv1p = uv0/2, uv1a/2

    im0p, im1p = cv2.pyrDown(im0p), cv2.pyrDown(im1p)
    uv0p, uv1p = uv0p/2, uv1p/2

    pt_idx = 3
    sz = hSz*2

    M = np.array([[1, 0, hSz-uv0p[pt_idx][0]],[0,1, hSz-uv0p[pt_idx][1]]], dtype=np.float32)
    fig_ref = cv2.warpAffine(im0p, M, (sz, sz))

    du, dv = uv1a[:, 0] - uv1b[:, 0], uv1a[:, 1] - uv1b[:, 1]
    da = (du*du + dv*dv)**0.5
    du, dv = du/da, dv/da

    nx, ny = searchSinglePt(fig_ref, im1p, uv1p[pt_idx][0], uv1p[pt_idx][1], 
                   du[pt_idx], dv[pt_idx], hSz)
    return nx, ny


def imgGrad(img, u, v, hSz):
    #u, v = u0, v0
    ret = np.zeros((len(u), 3))
    for i, (x, y) in enumerate(zip(u.astype(int), v.astype(int))):
        crop = img[y-hSz:y+hSz, x-hSz:x+hSz].astype(float)
        cropx = img[y-hSz:y+hSz, x-hSz+1:x+hSz+1].astype(float)
        cropy = img[y-hSz+1:y+hSz+1, x-hSz:x+hSz].astype(float)
        dx, dy = np.mean(crop-cropx), np.mean(crop-cropy)
        da = (dx*dx+dy*dy)**0.5
        ret[i] = dx/da, dy/da, da
    return ret

def gradFilter(imgrad, u1, v1, u2, v2):
    igrad = imgGrad(g0, u0, v0, 10)
    dx = u1 - u2
    dy = v1 - v2
    da = (dx*dx+dy*dy)**0.5
    dx, dy = dx/da, dy/da

    vx = igrad[:,0]*dx
    vy = igrad[:,1]*dy
    vv = (vx*vx+vy*vy)**0.5
    return vv, dx, dy

def DrawGrad(img, xa, ya, dxa, dya, hsz, color=(255,0,0)):
    if len(img.shape) < 3:
        imgc = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        imgc = img.copy()
    for u, v, du, dv in zip(xa, ya, dxa, dya):
        u, v = int(u), int(v)
        u2, v2 = int(u+hsz*du), int(v+hsz*dv)
        cv2.line(imgc,  (u, v), (u2, v2), color)
    return imgc

def drawSquares(img, xa, ya, hsz, color=(255,0,0)):
    if len(img.shape) < 3:
        imgc = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        imgc = img.copy()
    for u, v in zip(xa, ya):
        u, v = int(u), int(v)
        cv2.rectangle(imgc, (u-hsz, v-hsz), (u+hsz, v+hsz), color)

    return imgc

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
px, py= px[px>10], py[px>10]
px, py= px[py>10], py[py>10]
px, py= px[px<g0.shape[1]-10], py[px<g0.shape[1]-10]
px, py= px[py<g0.shape[0]-10], py[py<g0.shape[0]-10]
print(len(px))
pos = list(range(0, len(px), len(px)//300))
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

plt.figure(3)
plt.subplot(121)
plt.imshow(f0)
plt.subplot(122)
plt.imshow(f1)

if 1:
    imgrad = imgGrad(g0, u0, v0, 10)
    vv, dx, dy = gradFilter(imgrad, u1, v1, u2, v2)

    uv0 = np.vstack((u0, v0)).T
    uv1a = np.vstack((u1, v1)).T
    uv1b = np.vstack((u2, v2)).T
    
    thr = 0.3
    uv0 = uv0[vv>=thr]
    uv1a = uv1a[vv>=thr]
    uv1b = uv1b[vv>=thr]

    uv1f = search_epLine_pyr(g0, g1, uv0, uv1a, uv1b, 10)

if 0:
    img0s = drawSquares(g0, uv0[:, 0], uv0[:, 1], 4, color=(255,0,0))
    img1s = drawSquares(g1, uv1a[:, 0], uv1a[:, 1], 4, color=(255,0,0))
    img1s = drawSquares(img1s, uv1f[:, 0], uv1f[:, 1], 4, color=(0,255,0))

    plt.figure(40); plt.imshow(img0s)
    plt.figure(41); plt.imshow(img1s)


if 0:    
    imgc = DrawGrad(g0, u0, v0, imgrad[:, 0], imgrad[:, 0], 10, (255,0,0))
    imgc = DrawGrad(imgc, u0[vv<thr], v0[vv<thr], dx[vv<thr], dy[vv<thr], 10, (0,255,0))
    imgc = DrawGrad(imgc, u0[vv>=thr], v0[vv>=thr], dx[vv>=thr], dy[vv>=thr], 10, (0,0,255))
    plt.figure(30); plt.imshow(imgc)

if 0:

    #results
    MI = np.array([[1, 0, hSz-uv1appb[pt_idx][0]],[0,1, hSz-uv1appb[pt_idx][1]]], dtype=np.float32)
    MR = np.array([[1, 0, hSz-nx],[0,1, hSz-ny]], dtype=np.float32)
    fig_I = cv2.warpAffine(im1ppb, MI, (sz, sz))
    fig_R = cv2.warpAffine(im1ppb, MR, (sz, sz))

    plt.figure(33); plt.imshow(fig_I, cmap='gray')
    plt.figure(34); plt.imshow(fig_R, cmap='gray')
    

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
    
    print("\nsrch")
    #pt0, pt1a, pt1b = (int(u0[3]), int(v0[3])), (int(u1[3]), int(v1[3])), (int(u2[3]), int(v2[3]))
    pt0, pt1a, pt1b = (u0[3], v0[3]), (u1[3],v1[3]), (u2[3],v2[3])
    searchEpLine(g0, g1, pt0, pt1a, pt1b, 10)

    #dx, dy, da = imgGrad(g0, pt0, 8)

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
   
