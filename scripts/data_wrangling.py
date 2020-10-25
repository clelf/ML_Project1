import numpy as np
#### functions for wrangling data ####

###########################
## Replaces nan with a value
##
###########################
def clean_nan(tx_, replacement):
    tx = tx_.copy()
    for j in range(tx.shape[1]):
        if (j!=22):
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


