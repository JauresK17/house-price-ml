from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline 
def scale_data(X_train,X_val,X_test):
    scaler = StandardScaler()
    """
    Standardiser les features
    Fit sur le train seulement 
    """
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled,X_val_scaled,X_test_scaled
def final_training (X_train_scaled,y_train):
    """
    Entraine le modele final 
    """
    model_final = Pipeline([
        ("model",LinearRegression())
    ])
    model_final.fit(X_train_scaled,y_train)
    return model_final