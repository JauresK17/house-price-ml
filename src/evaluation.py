import numpy as np 
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error 

def evaluate(y_true,y_pred):
    mse = mean_squared_error(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    r2  = r2_score(y_true,y_pred)
    rmse = np.sqrt(mse)
    
    relative_error = np.mean(abs(y_true - y_pred ) / y_true )
    return mse,mae,r2,rmse,relative_error 