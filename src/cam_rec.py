#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:31:31 2023

@author: fip
"""

import cv2
from corners import getCornersFromImg
import numpy as np

frames = np.zeros((10, 480, 640), dtype=np.uint8)
corners = np.zeros((10, 81, 2), dtype=np.uint16)

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness=2

def drawPts(img, pts):
    imgc = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if len(pts) > 0:
        pts = pts.astype(int)
        for (x, y) in pts:
            cv2.circle(imgc, (x, y), 3, (255, 255, 0), 2)
    return imgc

vidcap = cv2.VideoCapture(0)
cap_flag = False
fr_cnt = 0

if vidcap.isOpened():

    while(fr_cnt<10):
        ret, frame = vidcap.read()  #capture a frame from live video
        pts, img = getCornersFromImg(frame)
        
        if len(pts) > 0 and cap_flag:
            frames[fr_cnt] = img
            corners[fr_cnt] = pts
            cap_flag = False
            fr_cnt += 1
            print(pts.shape)
        
        img = drawPts(img, pts)
        img = cv2.putText(img, 'cap: %d, n: %d'% (cap_flag, fr_cnt), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("Frame",img)   #show captured frame
        
        
        #press 'q' to break out of the loop
        kp = cv2.waitKey(10) & 0xFF
        if kp  == ord('q'):
            break
        elif kp == 32:
            cap_flag = True
        elif kp != 255:
            print (kp)
        
else:
    print("Cannot open camera")
    
vidcap.release()
cv2.destroyAllWindows()

if 0:
    with open('frames.npy', 'wb') as f:
        np.save(f, frames)
    with open('corners.npy', 'wb') as f:
            np.save(f, corners)
    
    with open('frames.npy', 'rb') as f:
        a = np.load(f)
    with open('corners.npy', 'rb') as f:
        b = np.load(f)
    a.shape
    b.shape
    