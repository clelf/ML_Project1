import numpy as np
#### Replaces NaN's with imported 

def clean_nan(tx_, replacement):
    tx = tx_.copy()
    for j in range(tx.shape[1]):
        if (j!=22):
            tx[:,j] = np.nan_to_num(tx[:,j], nan=replacement[j])
            
    return tx

