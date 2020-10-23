import numpy as np

def clean_nan(tx_, feature):
    tx = tx_.copy()
    for i in range(tx.shape[1]):
        if (i!=22):
            tx[np.argwhere(np.isnan(tx)),i] = feature[i]
            

    #tx_ = np.where(tx == nan, np.ma.array(tX, mask= (tX == nan)).mean(axis=0), tx)
    
    '''
    for i in range(tx.shape[0]):
        for j in range(tx.shape[1]):
            if (np.isnan(tx[i, j])): 
                tx[i, j] = feature[j]
    '''

    return tx
