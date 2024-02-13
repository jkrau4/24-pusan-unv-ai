import numpy as np
from MultiClass import MultiClass
from Softmax import Softmax
from Sigmoid import Sigmoid

import numpy as np

def TestMultiClass():
    X = np.zeros((5, 5, 5))
    
    X[:, :, 0] = [[0,1,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,1,1,1,0]]
    
    X[:, :, 1] = [[1,1,1,1,0],
                  [0,0,0,0,1],
                  [0,1,1,1,0],
                  [1,0,0,0,0],
                  [1,1,1,1,1]]
    
    X[:, :, 2] = [[1,1,1,1,0],
                  [0,0,0,0,1],
                  [0,1,1,1,0],
                  [0,0,0,0,1],
                  [1,1,1,1,0]]
    
    X[:, :, 3] = [[0,0,0,1,0],
                  [0,0,1,1,0],
                  [0,1,0,1,0],
                  [1,1,1,1,1],
                  [0,0,0,1,0]]
    
    X[:, :, 4] = [[1,1,1,1,1],
                  [1,0,0,0,0],
                  [1,1,1,1,0],
                  [0,0,0,0,1],
                  [1,1,1,1,0]]
    
    D = np.array([[[1,0,0,0,0]],
                  [[0,1,0,0,0]],
                  [[0,0,1,0,0]],
                  [[0,0,0,1,0]],
                  [[0,0,0,0,1]]])
    
    
    W1 = 2*np.random.random((50, 25)) - 1
    W2 = 2*np.random.random((5, 50)) - 1
    
    for epoch in range(10000):
        W1, W2 = MultiClass(W1, W2, X, D)
    
        
    N = 5
    for k in range(N):
        a = X[:, :, k]
        x  = a.reshape(25,1)
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v  = np.matmul(W2, y1)
        y  = Softmax(v)
        print(y)
    
    return W1, W2


TestMultiClass()