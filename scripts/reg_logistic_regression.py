import matplotlib.pyplot as plt
import numpy as np
from logistic_regression import*

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    print("hullo")
    for n_iter in range(max_iters):
        gradient = compute_gradient_log(y, tx, w) + 2* lambda_ *w
        loss = calculate_loss_log(y,tx,w, lambda_)
        w = w-gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
    plt.plot(losses)    
    return ws[-1], losses[-1]

