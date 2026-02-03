from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LinearRegression,Ridge 
from sklearn.metrics import make_scorer 
from sklearn.model_selection import KFold,cross_val_score
import numpy as np 
def cross_validate(X_train,y_train):
    """
    Pipeline de la cross-validation 1 (scale data + model(linearRegression))
    """
    pipeline_1 = Pipeline([
        ("scaler",StandardScaler()),
        ("model",LinearRegression())
    ])
    pipeline_2 = Pipeline([
        ("scaler",StandardScaler()),
        ("model",Ridge(alpha=1.0))

    ])
    ##KFold
    cv = KFold(n_splits = 5 , shuffle=True , random_state = 42 )
    ##Cross-validation(Evaluation/LinearRegression) 1 : 
       #Mean_absolute_error(MAE)
    mae_score_1 = cross_val_score(pipeline_1,X_train,y_train,cv=cv,scoring='neg_mean_absolute_error')
       #Mean_squared_error(MSE)
    mse_score_1 = cross_val_score(pipeline_1,X_train,y_train,cv=cv,scoring='neg_mean_squared_error')
       #Root_mean_squared_error(RMSE)
    rmse_score_1 = cross_val_score(pipeline_1,X_train,y_train,cv=cv,scoring="neg_root_mean_squared_error")
       #R2_score
    r2_score_1 = cross_val_score(pipeline_1,X_train,y_train,cv=cv,scoring="r2")
    ##Cross-validation(Evaluation/Ridge) 2 :
       #Mean_absolute_error(MAE)
    mae_score_2 = cross_val_score(pipeline_2,X_train,y_train,cv=cv,scoring='neg_mean_absolute_error')
       #Mean_squared_error(MSE)
    mse_score_2 = cross_val_score(pipeline_2,X_train,y_train,cv=cv,scoring='neg_mean_squared_error')
       #Root_mean_squared_error(RMSE)
    rmse_score_2 = cross_val_score(pipeline_2,X_train,y_train,cv=cv,scoring = 'neg_root_mean_squared_error')
       #R2_score
    r2_score_2 = cross_val_score(pipeline_2 , X_train,y_train,cv=cv,scoring ="r2")
    
    ##Affichage des evaluations 
    print("LES METRICS D'EVALUATIONS DU MODEL DE REGRESSION LINEAIRE ")
    print("-------------------------------------------------")
    print(f"MAE : {mae_score_1.mean()} | {mae_score_1.std()}")
    print(f"MSE : {mse_score_1.mean()} | {mse_score_1.std()}")
    print(f"RMSE: {rmse_score_1.mean()}| {rmse_score_1.std()}")
    print(f"R2  : {r2_score_1.mean()}  | {r2_score_1.std()}")
    print("--------------------------------------------------")
    print("LES METRICS D'EVALUATIONS DU MODEL RIDGE")
    print(f"MAE : {mae_score_2.mean()} | {mae_score_2.std()}")
    print(f"MSE : {mse_score_2.mean()} | {mse_score_2.std()}")
    print(f"RMSE: {rmse_score_2.mean()}| {rmse_score_2.std()}")
    print(f"R2  : {r2_score_2.mean()}  | {r2_score_2.std()}")
    
