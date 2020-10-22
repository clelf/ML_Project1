import numpy as np

from least_squares_SGD import *
from least_squares_GD import *
from proj1_helpers import batch_iter
from ridge_regression import *
from build_polynomial import build_poly
from least_squares import *

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


def degree_tuning_LS(y,tx):
    degrees = np.linspace(1, 5, 5, dtype=int)
    losses=[]
    
    for degree in degrees:
        x=build_poly(tx,degree)
        _,loss = least_squares(y,x)
        losses.append(loss)
        
    best_degree=degrees[np.argwhere(losses == min(losses))]
    return best_degree.item()


def lambda_tuning_ridge(y, tx):
    # Compute weights and loss for each gamma
    losses = []
    # Boundaries are set very low because it systematically takes the lowest boundary when lower limit is above -10
    lambdas = np.logspace(-20, -10, 25)
    for lambda_ in lambdas:
        _, loss = ridge_regression(y, tx, lambda_)
        losses.append(loss)
    lambda_opt = lambdas[np.argwhere(losses == min(losses))]    
    return lambda_opt.item()
