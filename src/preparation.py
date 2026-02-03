def prepare_data(data):
    """
    Separe les features et la cible 
    """
    X = data.drop("price",axis=1).values 
    y = data["price"].values.reshape(-1,1)
    return X, y 
