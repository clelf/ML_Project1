import numpy as np

def compute_gradient_log(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y)

def calculate_loss_log(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def sigmoid(t):
    """apply sigmoid function on t."""
    print(np.exp(-t))
    sigmoid = 1./(1. + np.exp(-t))
    return sigmoid

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        gradient = compute_gradient_log(y, tx, ws[n_iter])
        w = ws[n_iter]-gamma*gradient
        loss = calculate_loss_log(y,tx,w)
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return ws[-1], losses[-1]