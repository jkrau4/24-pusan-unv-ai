import numpy as np
from MultiClass import MultiClass
from Softmax import Softmax
from Sigmoid import Sigmoid

def TestMultiClass():
    X = np.zeros(5, 5, 5)

    X[:, :, 0] = np.array([[0, 0, 1, 1, 0],
                        [0, 0, 1, 1, 0],
                        [0, 1, 0, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 1, 1, 1, 0]])

    X[:, :, 1] = np.array([[1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1]])

    X[:, :, 2] = np.array([[1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 1, 1, 1, 0],
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0]])

    X[:, :, 3] = np.array([[0, 1, 1, 1, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 1, 1, 1, 0]])

    X[:, :, 4] = np.array([[0, 1, 1, 1, 1],
                        [0, 1, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 1, 0],
                        [1, 1, 1, 1, 0]])
    
    N = 5
    for k in range(N):
        a = X[:, :, k]
        x = a.reshape(25,1)
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v = np.matmul(W2, y1)
        y = Softmax(v)
        print(y)

TestMultiClass()
