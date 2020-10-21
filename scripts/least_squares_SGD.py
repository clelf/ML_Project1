# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
from proj1_helpers import batch_iter
from costs import compute_mse

def compute_stoch_gradient(y, tx, w):
    #here tx and y have already been minibatched
    e = y - tx@w
    grad = -1/(y.shape[0])*(tx.T)@e
    
    return grad


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad  = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w-gamma*grad
            loss=compute_mse(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
    return ws[-1], losses[-1]