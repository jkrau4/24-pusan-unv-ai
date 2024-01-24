import numpy as np

def Softmax(X):
    a = np.exp(X)
    b = np.sum(a)
    y = a / b
    return y