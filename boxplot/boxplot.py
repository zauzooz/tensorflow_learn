import matplotlib.pyplot as plt
import numpy as np

def example_1():
    # Creating dataset
    np.random.seed(10)
    data = np.random.normal(100, 20, 200)
    
    fig = plt.figure(figsize =(10, 7))
    
    # Creating plot
    plt.boxplot(data)
    
    # show plot
    plt.show()

def example_2():
    # Creating dataset
    np.random.seed(10)
    
    data_1 = np.random.uniform(-1, 1, 100).tolist()
    data_2 = np.random.uniform(-1, 1, 90).tolist()
    data_3 = np.random.uniform(-1, 1, 80).tolist()
    data_4 = np.random.uniform(-1, 1, 70).tolist()
    data_5 = np.random.uniform(-1, 1, 60).tolist()
    data_6 = np.random.uniform(-1, 1, 50).tolist()
    data = [data_1, data_2, data_3, data_4, data_5, data_6]
    
    fig = plt.figure(figsize =(10, 7))
    
    # Creating axes instance
    ax = fig.add_axes([-1, -1, 1, 1])
    
    # Creating plot
    bp = ax.boxplot(data)
    
    # show plot
    plt.show()