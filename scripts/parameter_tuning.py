import numpy as np

from least_squares_SGD import *
from least_squares_GD import *
from proj1_helpers import batch_iter
from ridge_regression_edited import ridge_regression
from build_polynomial import build_poly
from least_squares import *
from logistic_regression import *
from reg_logistic_regression import *
from cross_validation import cross_val, build_k_indices

def gamma_tuning_SGD(y, tx, initial_w, max_iters, batch_size):
    #Compute weights and loss for each gamma
    losses = []
    # Values above range 10^-5 make the algorithm diverge
    gammas = np.logspace(-10, -5, 15)
    for gamma in gammas:
        _, loss = least_squares_SGD(y,tx,initial_w, batch_size, max_iters,gamma)
        losses.append(loss)
    gamma_opt = gammas[np.argwhere(losses == min(losses))] # returns an array
    return gamma_opt.item() #returns a scalar

def gamma_tuning_GD(y, tx, initial_w, max_iters):
    #Compute weights and loss for each gamma
    losses = []
    # Values above range 10^-5 make the algorithm diverge
    gammas = np.logspace(-10, -5, 15)
    for gamma in gammas:
        _, loss = least_squares_GD(y,tx,initial_w, max_iters,gamma)
        losses.append(loss)
    gamma_opt = gammas[np.argwhere(losses == min(losses))] # returns an array
    return gamma_opt.item() #returns a scalar

def gamma_tuning_log(y, tx, initial_w, max_iters):
    #Compute weights and loss for each gamma
    losses = []
    # Values above range 10^-5 make the algorithm diverge
    gammas = np.logspace(-10, -5, 15)
    
    for gamma in gammas:
        _, loss = logistic_regression(y,tx,initial_w, max_iters,gamma)
        losses.append(loss)
    gamma_opt = gammas[np.argwhere(losses == min(losses))] # returns an array
    print(gamma_opt)
    return gamma_opt.item() #returns a scalar

def degree_tuning_LS(y,tx):
    degrees = np.linspace(1, 10, 10, dtype=int)
    losses=[]
    
    for degree in degrees:
        x=build_poly(tx,degree)
        _,loss = least_squares(y,x)
        losses.append(loss)
        
    best_degree=degrees[np.argwhere(losses == min(losses))]
    return best_degree.item()

def lambda_tuning_ridge(y, tx):
    # Compute weights and loss for each gamma
    #print(losses)
    losses = []
    
    # Boundaries are set very low because it systematically takes the lowest boundary when lower limit is above -10
    lambdas = np.logspace(-20, -10, 200)
    
    for lambda_ in lambdas:
        _, loss = ridge_regression(y, tx, lambda_)
        losses.append(loss)
        #losses[i] = loss
    lambda_opt = lambdas[np.argwhere(losses == min(losses))]  
    lambda_=lambda_opt[-1]
    return lambda_.item()
    #return lambdas, losses
    
def lambda_tuning_ridge_cv(y, tx_, k_fold, lambdas, seed):
    # Build split indices
    k_indices = build_k_indices(y, k_fold, seed)
    
    #Prepare matrices for iteration
    k_rmse_tr = np.zeros([k_fold])
    k_rmse_te = np.zeros([k_fold])
    
    w_k_folds = np.zeros([k_fold,tx_.shape[1]])
    w_ave_lambda = np.zeros([len(lambdas),tx_.shape[1]])
    
    rmse_tr = np.zeros([len(lambdas)])
    rmse_te = np.zeros([len(lambdas)])
    #k_var = np.zeros([len(lambdas)])
    
    # Grid search with k-fold CV
    for i, lambda_ in enumerate(lambdas):
        for k in range(k_fold):
            loss_tr_hold, loss_te_hold, w_hold = cross_val(y, tx_, k_indices, k, lambda_)
            
            w_k_folds[k,:] = w_hold
            k_rmse_tr[k] = loss_tr_hold
            k_rmse_te[k] = loss_te_hold
            
        w_ave_lambda[i,:] = np.mean(w_k_folds, 0)
        rmse_tr[i] = np.mean(k_rmse_tr)
        rmse_te[i] = np.mean(k_rmse_te)
        #k_var[i] = np.var(k_rmse_te)
    
    losses = rmse_te
    lambda_opt = lambdas[np.argwhere(rmse_te == min(rmse_te))].ravel().item()
    w_opt = w_ave_lambda[np.where(rmse_te==min(rmse_te)),:].ravel()
    return lambda_opt, w_opt, rmse_tr, rmse_te

    
def param_tuning_log(y, tx, initial_w, max_iters, lambda_):
    #Compute weights and loss for each gamma
    losses = []
    # Values above range 10^-5 make the algorithm diverge
    gammas = np.logspace(-10, -5, 16)
    for gamma in gammas:
        _, loss = logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_)
        losses.append(loss)
    gamma_opt = gammas[np.argwhere(losses == min(losses))] # returns an array
    gamma_opt = gamma_opt[-1]
    return gamma_opt.item() #returns a scalar


def param_tuning_reg_log(y, tx, initial_w, max_iters, lambda_):
    #Compute weights and loss for each gamma
    losses = []
    # Values above range 10^-5 make the algorithm diverge
    gammas = np.logspace(-10, -5, 16)
    for gamma in gammas:
        _, loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
        losses.append(loss)
    gamma_opt = gammas[np.argwhere(losses == min(losses))] # returns an array
    gamma_opt = gamma_opt[-1]
    return gamma_opt.item() #returns a scalar