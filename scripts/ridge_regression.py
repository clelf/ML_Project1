import numpy as np
from costs import compute_rmse


def ridge_regression(y, tx, lambda_):
    lambda_2 = lambda_*2*tx.shape[0]
    w = np.linalg.solve(tx.T@tx+lambda_2*np.identity(tx.shape[1]), tx.T@y)
    loss = compute_rmse(y, tx, w)
    return w, loss