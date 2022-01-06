import numpy as np
from utils import *

#########################
###### ML METHODS #######
#########################

## Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Computes linear regression using gradient descent
    Input parameters:
    initial_w = initial starting point of weights for the iterations
    max_iters = maximum number of iterations
    gamma = learning rate
    """

    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_rmse(y, tx, w)
        w = w-gamma*grad
    return ws, loss


## Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Computes linear regression using stochastic gradient descent
    Input parameters:
    initial_w = initial starting point of weights for the iterations
    batch_size = size of the batch considered
    max_iters = maximum number of iterations
    gamma = learning rate
    """
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad  = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w-gamma*grad
            loss=compute_rmse(y, tx, w)
    return w, loss

## Least squares regression using normal equations
def least_squares(y, tx): 
    """
    Computes least squares regression using normal equations
    """
    A=np.transpose(tx).dot(tx)
    B=np.transpose(tx).dot(y)
    w = np.linalg.solve(A,B)
    loss = compute_rmse(y,tx,w)
    return w, loss

## Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """
    Computes ridge regression using normal equations
    Input parameter:
    lambda_ = complexity of the model
    """
    lambda_2 = lambda_*2*tx.shape[0]
    w = np.linalg.solve(tx.T@tx+lambda_2*np.identity(tx.shape[1]), tx.T@y)
    loss = compute_rmse(y, tx, w)
    return w, loss

## Logistic regression using gradient descent
def logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    losses = []
    w = initial_w
    threshold = pow(10,-8)

    for n_iter in range(max_iters):
        gradient = compute_gradient_log(y, tx, w)
        loss = calculate_loss_log(y,tx,w, lambda_)
        w = w-gamma*gradient
        losses.append(loss)
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2])) < threshold:
            break
    return w, losses[-1]


## Regularized logistic regression using gradient descent
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_log(y, tx, w) + lambda_ *w
        loss = calculate_loss_log(y,tx,w, lambda_)
        w = w-gamma*gradient
    return w, loss
