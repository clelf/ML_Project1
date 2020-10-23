import numpy as np
from costs import compute_rmse

#Least squares regression using normal equations
def least_squares(y, tx):
    # *************************************************
    #using the course equation for optimal_weights = (((X^T)X)^-1).(X^T).y 
    A=np.transpose(tx).dot(tx)
    B=np.transpose(tx).dot(y)
    w = np.linalg.solve(A,B)
    
    loss = compute_rmse(y,tx,w)
    
    return w, loss