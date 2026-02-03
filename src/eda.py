import matplotlib.pyplot as plt 

def info(data):
    return data.info()
def desc(data):
    return data.describe()
def corr(data):
    cor = data.corr()
    print(cor["price"].sort_values(ascending = False))
    

def plot_target_distribution(data  ):
    plt.hist(data["price"],bins=50)
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.title("Price distribution ")
    plt.show()
def outlier(data):
    plt.boxplot(data["price"],vert = False)
    plt.xlabel("Price")
    plt.title("Price outlier")