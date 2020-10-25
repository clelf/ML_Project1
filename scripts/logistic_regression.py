import numpy as np

def compute_gradient_log(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    k = 1.0 / tx.shape[0]
    return tx.T.dot(pred - y)*k


def calculate_loss_log(y, tx, w, lambda_):
    """compute the cost by negative log likelihood."""
    a= sigmoid(tx.dot(w))
    #loss = y.T.dot(np.log(probLabel)) + (1-y).T.dot(np.log(1-probLabel))
    loss = - (1 / tx.shape[0]) * np.sum((y * np.log(a)) + ((1 - y) * np.log(1 - a)))
    return loss + np.squeeze(w.T.dot(w))*lambda_

def sigmoid(value):
    a = np.exp(-value)
    return 1.0/ (1.0 + a)
    

def logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_log(y, tx, w)
        loss = calculate_loss_log(y,tx,w, lambda_)
        w = w-gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
    return ws[-1], losses[-1]
