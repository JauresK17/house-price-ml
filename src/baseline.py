import numpy as np 
def baseline (y_test,y_train):
    y_pred_baseline = np.full_like(y_test , y_train.mean(),dtype=float)
    return y_pred_baseline 