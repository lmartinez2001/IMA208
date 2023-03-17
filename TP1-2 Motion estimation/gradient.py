#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 18:06:43 2022

@author: ckervazo
"""
import numpy as np

def gradient(M,stepX=1.,stepY=1.):
# Computes the gradient of an image, over the rows and the column directions. StepY is the assumed gap between the rows and StepX is the assumed gap between the columns

    gy = np.gradient(M,stepY,axis=0)
    gx = np.gradient(M,stepX,axis=1)
    
    
    return gx,gy
