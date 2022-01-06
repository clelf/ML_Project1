import numpy as np
import csv


###################
###### UTILS ######
###################

# Contains gradient, costs, data processing, polynomial expansion, hyper-parameter tuning and additional helper functions

#### Gradients ####

def compute_gradient(y, tx, w):
    e = y - tx@w
    grad = -1/(tx.shape[0])*(tx.T)@e
    return grad

def compute_stoch_gradient(y, tx, w):
    e = y - tx@w
    grad = -1/(y.shape[0])*(tx.T)@e
    return grad

def compute_gradient_log(y, tx, w):
    """compute the gradient of logistic loss"""
    pred = sigmoid(tx.dot(w))
    k = 1.0 / tx.shape[0]
    return tx.T.dot(pred - y)*k

#### Costs #####

def compute_mse(y, tx, w):
    """ Computes mean square error"""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2*len(e))
    return mse

def compute_rmse(y, tx, w):
    """ Computes root mean square error"""
    return np.sqrt(2 * compute_mse(y, tx, w))

def calculate_loss_log(y, tx, w, lambda_):
    """Computes the loss by negative logistic likelihood"""
    a= sigmoid(tx.dot(w))
    loss = - (1 / tx.shape[0]) * np.sum((y * np.log(a)) + ((1 - y) * np.log(1 - a)))
    return loss + 0.5*np.squeeze(w.T.dot(w))*lambda_





#### Logistic regression additional helper functions

def sigmoid(value):
    a = np.exp(-value)
    return 1.0/ (1.0 + a)




#### Data wrangling ####

def clean_nan(tx_, replacement, skip_index):
    """ Replaces nan with a value"""
    tx = tx_.copy()
    for j in range(tx.shape[1]):
        if (j!=skip_index):
            tx[:,j] = np.nan_to_num(tx[:,j], nan=replacement[j])
            
    return tx


def shift_zeros(tx_, indices, new_zero):
    """Replaces 0 with a value"""
    tx = tx_.copy()
    for ind in indices:
        tX_hold = tx[:,ind]
        tX_hold[tX_hold==0.0] = new_zero
        tx[:,ind] = tX_hold

    return tx


def log_transform(tx_, indices):
    """ log transform data on indices provided"""
    tX_logged = tx_.copy()
    for ind in indices:
        tX_logged[:,ind] = np.log(tx_[:,ind])
    return tX_logged


def compute_statistics(tx_):
    """ Calculate statistics for each feature"""
    feature_details = np.zeros([7, tx_.shape[1]])
    for i in range(tx_.shape[1]):
        feature_details[0, i] = np.nanmean(tx_[:,i])
        feature_details[1, i] = np.nanvar(tx_[:,i])
        feature_details[2, i] = np.nanstd(tx_[:,i])
        feature_details[3, i] = np.nanmin(tx_[:,i])
        feature_details[4, i] = np.nanmax(tx_[:,i])
        feature_details[5, i] = np.isnan(tx_[:,i]).sum()
        feature_details[6, i] = np.nanmedian(tx_[:,i])
    return feature_details


def standardize_features(tx, skip_index, mean, std):
    """ Standardize select features """
    for i in range(tx.shape[1]):
        if (i!=skip_index):
            tx[:,i] = (tx[:,i] - mean[i])/std[i]

    return tx


def data_process(tX, train, train_mean, train_std, train_median):
    """ 
    Proceeds to various modifictions of data sets
    Input parameters:
    train = true if tX is train data, false if tX is test data
    train_mean, train_std, train_median = facultative feature statistics used to standardize test data
    """
    # Replaces outliers with nan
    tX[tX==-999] = np.nan
    
    if (train):
        #Compute statistics and store median, mean and std
        feature_statistics = compute_statistics(tX)
        mean = feature_statistics[0, :]
        std = feature_statistics[2, :]
        median = feature_statistics[6, :]
    else:
        mean = train_mean
        std = train_std
        median = train_median
        
    # Enhancement of feature distributions
    ## have to shift 0 with 0.0005 in order to avoid log(0)
    zero_shift = 0.0005 # Done to avoid computing log(0)
    categorical_index = 22 # index to avoid standardization because this feature is categorical    
    tX_nan_replaced = clean_nan(tX, median, categorical_index) # replace nan values by the median of each feature
    tx = standardize_features(tX_nan_replaced, categorical_index, mean, std)
    return tx, mean, std, median





#### Polynomial ####
def build_poly(x, degree):
    
    poly = np.ones((x.shape[0], 1))
    for d in range(1, degree+1):
        poly = np.c_[poly, np.power(x, d)]
    return poly

def feature_expansion(tx_, degree, categorical_index):
    tx_categorical = tx_[:,categorical_index]
    
    tx_no_categorical = np.delete(tx_, [categorical_index], 1)
    tx_poly = build_poly(tx_no_categorical, degree)
    
    tx = np.c_[tx_poly, tx_categorical]
    return tx


### Cross-validation
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
    x_train, y_train, x_test, y_test = split_data(x,y,k_indices,k)
    w_tr, rmse_tr = ridge_regression(y_train, x_train, lambda_)
    rmse_te = compute_rmse(y_test, x_test, w_tr)
    loss_tr = rmse_tr
    loss_te = rmse_te
    return loss_tr, loss_te, w_tr



### Hyper-parameter tuning


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

    
def lambda_tuning_ridge_cv(y, tx_):
    # Set parameters
    lambdas = np.logspace(-7, -3, 50)
    seed = 42
    k_fold = 5
    
    
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
    
    lambda_opt = lambdas[np.argwhere(rmse_te == min(rmse_te))].ravel().item()
    return lambda_opt

    
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


#### Helpers ####

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
            
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
            
            
from implementations import *