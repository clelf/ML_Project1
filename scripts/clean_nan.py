import numpy as np

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

