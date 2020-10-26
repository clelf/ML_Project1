import numpy as np
from costs import compute_rmse

def ridge_regression(y, tx, lambda_):
    lambda_2 = lambda_*2*tx.shape[0]
    w = np.linalg.solve(tx.T@tx+lambda_2*np.identity(tx.shape[1]), tx.T@y)
    
    #lambda_simplified = lambda_*2*tx.shape[0]
    #cov_X = np.dot(tx.T, tx)
    #inner_first = np.linalg.inv(cov_X + lambda_simplified*np.identity(cov_X.shape[0]))
    #all_X = np.dot(inner_first, tx.T)
    #w = np.dot(all_X, y)
    
    loss = compute_rmse(y, tx, w)
    return w, loss