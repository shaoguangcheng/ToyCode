#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def expdist() :
    xx = np.linspace(0,0.99,99)
    z = 1-xx
    y = np.zeros(99)
    i = 0
    for x in z :
        y[i] = -1*np.math.log(x)
        i += 1

    plt.figure(figsize=(8,4))
    plt.plot(xx)
    plt.plot(y)
    plt.show()

def gaussdist() :
    xx1 = np.linspace(0,1,100)
    xx2 = np.linspace(0,1,100)
    z1 = 2*xx1 - 1
    z2 = 2*xx2 - 1

    plt.figure(figsize=(8,4))
    plt.hold(True)
    p = np.zeros((2,99))
    i = j = 0
    n = 0
    while i < 2 :
        while j < 99 :
            if z1[i]**2 + z2[j]**2 <= 1 :
                p[0][n] = z1[i]
                p[1][n] = z2[j]
                n += 1
            i += 1
            j += 1
    print(n)
    plt.plot(p)
    plt.show()
    
if __name__ == "__main__" :
    gaussdist()
