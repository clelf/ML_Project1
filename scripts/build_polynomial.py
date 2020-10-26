# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    
    poly = np.ones((x.shape[0], 1))
    for d in range(1, degree+1):
        poly = np.c_[poly, np.power(x, d)]
    return poly

def feature_expansion(tx_, degree, categorical_index):
    tx_categorical = tx_[:,categorical_index]
    
    tx_no_categorical = np.delete(tx_, [categorical_index], 1)
    tx_poly = build_poly(tx_no_categorical, degree)
    
    tx = np.c_[tx_poly, tx_categorical]
    return tx