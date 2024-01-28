import numpy as np
from DeepDropout import DeepDropout
from Sigmoid import Sigmoid
from Softmax import Softmax
from Dropout import Dropout


def TestDeepDropout():
    X = np.zeros((5, 5, 5))
    
    X[:, :, 0] = np.array([[0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 0]])
    
    X[:, :, 1] = np.array([[1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 1, 1, 1, 0],
                  [1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1]])
    
    X[:, :, 2] = np.array([[1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 0]])
    
    X[:, :, 3] = np.array([[0, 0, 0, 1, 0],
                  [0, 0, 1, 1, 0],
                  [0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 0]])
    
    X[:, :, 4] = np.array([[1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 0]])
    
    D = np.array([[[1, 0, 0, 0, 0]],
                  [[0, 1, 0, 0, 0]],
                  [[0, 0, 1, 0, 0]],
                  [[0, 0, 0, 1, 0]],
                  [[0, 0, 0, 0, 1]]])
    
    W1 = 2*np.random.random((20, 25)) - 1
    W2 = 2*np.random.random((20, 20)) - 1
    W3 = 2*np.random.random((20, 20)) - 1
    W4 = 2*np.random.random(( 5, 20)) - 1

    for epoch in range(20000):
        W1, W2, W3, W4 = DeepDropout(W1, W2, W3, W4, X, D)


    N = 5
    for k in range(N):
        a = X[:, :, k]
        x = a.reshape(25,1)
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        y1 = y1 * Dropout(y1, 0.2)

        v2 = np.matmul(W2, y1)
        y2 = Sigmoid(v2)
        y2 = y2 * Dropout(y2, 0.2)

        v3 = np.matmul(W3, y2)
        y3 = Sigmoid(v3)
        y3 = y3 * Dropout(y3, 0.2)

        v = np.matmul(W4, y3)
        y = Softmax(v)
        print(y)

TestDeepDropout()  

