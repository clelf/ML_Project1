# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    '''poly = np.zeros((x.shape[0], degree+1))
    poly[:,0]=np.ones((x.shape[0],))
    
    for d in range(1,degree+1): #bc it stops just before degree+1 -> last value = degree
        poly[:,d]=x**d
    return poly'''
    
    poly = np.ones((x.shape[0], 1))
    for d in range(1, degree+1):
        poly = np.c_[poly, np.power(x, d)]
    return poly