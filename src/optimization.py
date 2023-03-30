#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 07:37:55 2023

@author: fip
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
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
    

