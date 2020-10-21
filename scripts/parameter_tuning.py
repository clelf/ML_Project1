import numpy as np

from least_squares_SGD import *
from least_squares_GD import *
from proj1_helpers import batch_iter

def gamma_tuning_SGD(tx, y, initial_w, max_iters, batch_size):
    #Compute weights and loss for each gamma
    losses = []
    # Values above range 10^-5 make the algorithm diverge
    gammas = np.logspace(-10, -5, 15)
    for gamma in gammas:
        _, loss = least_squares_SGD(y,tx,initial_w, batch_size, max_iters,gamma)
        losses.append(loss)
    gamma_opt = gammas[np.argwhere(losses == min(losses))] # returns an array
    return gamma_opt.item() #returns a scalar


def degree_tuning(y,tx):
    degrees = [1,2,3,4]
    losses=[]
    
    for i in degrees:
        x=build_poly(y,tx,degree)
        _,loss = least_squares_SGD(y,tx,initial_w, 1000, max_iters,gamma)
        losses.eppend(loss)
        
    best_degree=degrees[degrees.index(min(losses))]
    return best_degree