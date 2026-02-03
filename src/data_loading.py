import pandas as pd 

def load_data(path):
    """ 
    Charge le dataset depuis un fichier CSV
    """
    data = pd.read_csv(path)
    return data 