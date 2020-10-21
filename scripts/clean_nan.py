import numpy as np

def clean_nan(tx, feature):
    tx_ = tx
    #tx_ = np.where(tx == nan, np.ma.array(tX, mask= (tX == nan)).mean(axis=0), tx)
    
    for i in range(tx_.shape[1]):
        tx_[np.argwhere(np.isnan(tx_)),i] = feature[i]
    return tx_
