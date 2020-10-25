import numpy as np
#### functions for wrangling data ####

###########################
## Replaces nan with a value
##
###########################
def clean_nan(tx_, replacement, skip_index):
    tx = tx_.copy()
    for j in range(tx.shape[1]):
        if (j!=skip_index):
            tx[:,j] = np.nan_to_num(tx[:,j], nan=replacement[j])
            
    return tx


###########################
## Replaces 0 with a value
##
###########################
def shift_zeros(tx_, indices, new_zero):
    tx = tx_.copy()
    for ind in indices:
        tX_hold = tx[:,ind]
        tX_hold[tX_hold==0.0] = new_zero
        tx[:,ind] = tX_hold

    return tx


###########################
## log transform data on
## indices provided
###########################
def log_transform(tx_, indices):
    tX_logged = tx_.copy()
    for ind in indices:
        tX_logged[:,ind] = np.log(tx_[:,ind])
    return tX_logged

###########################
## Calculate statistics
## for each feature
###########################
def compute_statistics(tx_):
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

###########################
## Standardize select
## features
###########################
def standardize_features(tx_, skip_index, mean, std):
    tx = tx_.copy()
    for i in range(tx.shape[1]):
        if (i!=skip_index):
            tx[:,i] = (tx_[:,i] - mean[i])/std[i]
    return tx