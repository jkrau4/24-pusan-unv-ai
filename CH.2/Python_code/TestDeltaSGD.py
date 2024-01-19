import numpy as np
import DeltaSGD
import Sigmoid

def TestDeltaSGD():
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    D = np.array([[0],
                  [0],
                  [1],
                  [1]])
        
    W = 2*np.random.random((1, 3)) - 1
        
    for epoch in range(10000):
        W = DeltaSGD(W, X, D)
                
    N = 4
    for k in range(N):
        x = X[k,:].T
        v = np.matmul(W, x)
        y = Sigmoid(v)
        print(y)

TestDeltaSGD()