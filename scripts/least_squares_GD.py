# -*- coding: utf-8 -*-
"""Gradient Descent"""
from costs import compute_rmse


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx@w
    grad = -1/(tx.shape[0])*(tx.T)@e
    
    return grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_rmse(y, tx, w)
        w = w-gamma*grad

        # store w and loss
        ws.append(w)
        losses.append(loss)

    return ws[-1], losses[-1]