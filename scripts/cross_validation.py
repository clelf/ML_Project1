from costs import compute_rmse
from ridge_regression_edited import ridge_regression 
import numpy as np

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def split_data(x, y, k_indices, k=0):
    ### Splits according to indices ###
    ### For just a simple split just 
    
    x_test = x[k_indices[k],:]
    y_test = y[k_indices[k]]
    
    x_train = np.delete(x, list(k_indices[k]), 0)
    y_train = np.delete(y, list(k_indices[k]))
    return x_train, y_train, x_test, y_test

def cross_val(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    # ***************************************************
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    x_train, y_train, x_test, y_test = split_data(x,y,k_indices,k)
    # ***************************************************
    # form data with polynomial degree: TODO
    # ***************************************************
    #x_test_poly = build_poly(x_test, degree)
    #x_train_poly = build_poly(x_train, degree)
    ## ***************************************************
    ## ridge regression: TODO
    ## ***************************************************
    w_tr, rmse_tr = ridge_regression(y_train, x_train, lambda_)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    rmse_te = compute_rmse(y_test, x_test, w_tr)
    loss_tr = rmse_tr
    loss_te = rmse_te
    return loss_tr, loss_te, w_tr