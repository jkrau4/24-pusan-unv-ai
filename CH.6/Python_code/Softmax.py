import numpy as np

def Softmax(X):
    a = np.exp(X)
    sum = np.sum(a)
    y = a / sum 
    return y