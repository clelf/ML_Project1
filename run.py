import numpy as np
from utils import *
from implementations import *

# Extract training data
from proj1_helpers import *
DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


# Process and adjust training data
tx, mean, std, median  = data_process(tX, True, 0, 0, 0)

# Build polynomial expansion: choose degree, set categorical_index to 22
#feature_expansion(tx, degree, categorical_index)

# Tune parameters of the chosen method, HERE = ridge regression
lambda_ = lambda_tuning_ridge_cv(y, tx)

# Compute weights of method: ...
weights, loss = ridge_regression(y, tx, lambda_)

# Extract test data
DATA_TEST_PATH = '../data/test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Process and adjust test data
tx_test, _, _, _ = data_process(tX_test, False, mean, std, median)

# Generate ouput
OUTPUT_PATH = '../data/sample-submission.csv' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(weights, tx_test) # CAREFUL IT'S W3
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)