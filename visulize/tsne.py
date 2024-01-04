import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":

    x = np.random.uniform(0,1, size=(1000, 2))

    plt.scatter(x[[1,2,4],0], x[[1,2,4],1], c='r')
    plt.scatter(x[[3,5,6],0], x[[3,5,6],1], c='b')
    plt.scatter(x[[7,8,9],0], x[[7,8,9],1], c='g')
    plt.show()