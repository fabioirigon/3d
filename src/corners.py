#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:20:08 2023

@author: fip
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def correlate(fr0):
    kernel = np.ones((7, 7), dtype = float)
    kernel[:3, :3] = -1
    kernel[4:, 4:] = -1
    kernel[:, 3], kernel[3, :] = 0, 0
    img = cv2.filter2D(fr0.astype(float), -1, kernel)
    img = np.abs(img)

    img.shape, img.dtype, img.max(), img.min()
    img = (img-img.min())/(img.max()-img.min())
    img = (img*255).astype(np.uint8)
    return img

def getPts(im, thresh=180):
    pts = []
    while(im.max() > thresh and len(pts) < 100):
        ind = np.unravel_index(np.argmax(im, axis=None), im.shape)
        pts.append(ind)
        h, w = im.shape
        t, l = max(0, ind[0]-8), max(0, ind[1]-8)
        b, r = min(h, ind[0]+8), min(w, ind[1]+8) 
        im[t:b, l: r] = 0
    return pts

def filterElem(p0, p1, c):
    dy, dx = abs(p0[0]-p1[0]), abs(p0[1]-p1[1])
    if dy > dx:
        c = c[np.argsort(c[:, 0])]
        v = c[:, 0]
    else:
        c = c[np.argsort(c[:, 1])]
        v = c[:, 1]
    df = np.diff(v)
    i=0
    thr = np.median(df)*1.5
    rem = []
    while(df[i]>thr):
        rem.append(i)
        i = i+1
    i = -1
    while(df[i]>thr):
        rem.append(i)
        i = i-1
    for i in rem:
        c = np.delete(c, i, axis=0)
    return c


def distToLine(x0, y0, x1, y1, x2, y2):
    num = abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1))
    den = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return num/den

def getLinePts(p0, p1, pts, thr=6):
    y0, x0 = p0
    y1, x1 = p1
    ds = np.array([distToLine(p[1], p[0], x0, y0, x1, y1) for p in pts])
    #ind = (np.where(ds<10)[0]).astype(int)
    pts = np.array(pts)
    good = pts[np.where(ds<thr, True, False)].copy()
    bad = pts[np.where(ds>thr, True, False)].copy()
    good = filterElem(p0, p1, good)
    return good, bad

def getCentralCol(pts, figh, figw):

    #central point
    cy, cx = figh//2, figw//2
    d = [(cy-x[0])**2+(cx-x[1])**2 for x in pts]
    cp = pts[np.argmin(d)]

    # vertical neighbour:
    d = [((cp[0]-x[0])**2)+((cp[1]-x[1])**2)*10 for x in pts]
    d[np.argmin(d)] = np.max(d)
    neigv = pts[np.argmin(d)]

    colc, bad = getLinePts(cp, neigv, pts)
    colc = colc[np.argsort(colc[:, 0])]
    return colc

def getTopBotRows(colc, pts):

    res = []
    for cp in [colc[0], colc[-1]]:

        dc = [((cp[0]-x[0])**2)*10+(cp[1]-x[1])**2 for x in pts]
        dc[np.argmin(dc)] = np.max(dc)
        neig = pts[np.argmin(dc)]

        row, bad = getLinePts(cp, neig, pts)
        res.append(row.copy())

    return res[0], res[1]

def getFigure(n):
    if n >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES,n)
    ret, frame2 = cap.read()
    RED_FAC = np.min(frame2.shape[:2])//300

    fr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    fr = fr[::RED_FAC, ::RED_FAC].copy()
    th = cv2.adaptiveThreshold(fr,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,37,6)
    return fr, th

def getEveryColumn(lin0, lin_n, pts):
    # get every column:
    lin0 = lin0[np.argsort(lin0[:, 1])]
    lin_n = lin_n[np.argsort(lin_n[:, 1])]
    if len(lin0) != 9 or len(lin_n) != 9:
        return []

    cols = []
    for i in range(len(lin0)):
        p0, p1 = lin0[i], lin_n[i]
        col, bad = getLinePts(p0, p1, pts)
        if len(col) != 9:
            return []
        cols.append(col)
    return cols

def getCorners(img_num):
    fr, th = getFigure(img_num)
    img = correlate(th)
    pts = getPts(img)
    colc = getCentralCol(pts, fr.shape[0], fr.shape[1])
    topRow, botRow = getTopBotRows(colc, pts)

    cols = getEveryColumn(topRow, botRow, pts)
    return cols, fr

def getCornersFromImg(img):
    red_fac = np.min(img.shape[:2])//300
    fr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fr = fr[::red_fac, ::red_fac].copy()
    th = cv2.adaptiveThreshold(fr,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,37,6)

    img = correlate(th)
    pts = getPts(img)
    colc = getCentralCol(pts, fr.shape[0], fr.shape[1])
    if len(colc) < 2:
        return [], fr
    topRow, botRow = getTopBotRows(colc, pts)

    cols = getEveryColumn(topRow, botRow, pts)
    if len(cols)>0:
        cols = np.array(cols).reshape((81, 2))
        cols = cols[:, ::-1].astype(np.float32)

    return cols, fr

def sort_corners(pts):
    pts = pts.astype(float)
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
            srt.append(pts[pt_idx])
            #print(pts[pt_idx])
        if i < 8:
            x0, y0 = srt[-9]
            x1, y1 = x0+dy, y0-dx
            pt_idx = np.argmin(np.abs(pts[:, 0]-x1) + np.abs(pts[:, 1]-y1))
            srt.append(pts[pt_idx])
            #print(pts[pt_idx])
    pts_s = np.array(srt)
    return pts_s


if 0:
    img_num = 50
    cols, fr = getCorners(img_num)
    
    print(len(cols))
    cc = np.array(cols)
    
    gray = cv2.cvtColor(fr, cv2.COLOR_GRAY2RGB)
    for i in range(9):
        for j in range(9):
            pt = cc[i][j]
            color = (255, 0, 255) if i==2 and j==4 else (0, 255, 0)
            cv2.circle(gray, pt[::-1], 6, color, 4)
    plt.imshow(gray)


if __name__ == "__main__":
    
    fname = '../vids/checker_2.mp4'
    
    cap = cv2.VideoCapture(fname)
    
    # get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    ret, fr = cap.read()
    
    
    cnt = 0
    for k in range(int(totalFrames)-1):

        cols, gray = getCorners(-1)
        
        if len(cols) < 5:
            cnt += 1
            continue
        
        cc = np.array(cols)        
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
        for i in range(9):
            for j in range(9):
                pt = cc[i][j]
                color = (255, 0, 255) if i==2 and j==4 else (0, 255, 0)
                cv2.circle(gray, pt[::-1], 6, color, 4)
        
        
        cv2.imshow("Display window", gray)
        k = cv2.waitKey(10)
        if k == ord("q"):
            break
    
    print(totalFrames, cnt)
    cv2.destroyAllWindows()

if 0:
    tst = np.zeros((985, 81, 2), dtype=np.float32)
    tst.tofile("tst.bt")
        