from sklearn.model_selection import train_test_split 

def split_data(X,y):
    #70% train 
    X_train,X_temp,y_train,y_temp = train_test_split(
        X,y,
        test_size = 0.30,
        random_state = 42
    )
    #15% val , 15% test 
    X_val,X_test,y_val,y_test = train_test_split(
        X_temp,y_temp,
        test_size=0.50,
        random_state = 42 

    )
    return X_train , X_val ,X_test,y_train,y_val,y_test 