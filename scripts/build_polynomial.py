# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    poly = np.zeros((x.shape[0], degree+1))
    poly[:,0]=np.ones((x.shape[0],))
    
    for d in range(1,degree+1): #bc it stops just before degree+1 -> last value = degree
        poly[:,d]=x**d
    return poly
    # ***************************************************
    #raise NotImplementedError