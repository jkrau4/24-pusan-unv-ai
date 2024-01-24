import numpy as np
from Sigmoid import Sigmoid

def BackpropMmt(W1, W2, X, D):
    alpha = 0.9 # scalar
    beta = 0.9 # scalar

    mmt1 = np.zeros_like(W1) # (4, 3)
    mmt2 = np.zeros_like(W2) # (1, 4)

    N = 4
    for k in range(N):
        
        x = X[k, :].T # (3, 1)
        d = D[k] # scalar

        v1 = np.matmul(W1, x) # (1, 4)
        y1 = Sigmoid(v1) # (1, 4)
        v = np.matmul(W2, y1) # scalar
        y = Sigmoid(v) # scalar

        e = d - y # scalar
        delta = y * (1-y) * e # scalar

        e1 = W2 * delta # (1, 4)
        delta1 = y1 * (1 - y1) *e1 # (1, 4)

        dW1 = (alpha * delta1).T * x.T #(4, 3)
        mmt1 = dW1 + beta*mmt1 # (4, 3)
        W1 = W1 + mmt1 # (4, 3)

        dW2 = alpha * delta * y1.T # (1, 4)
        mmt2 = dW2 + beta*mmt2 # (1, 4)
        W2  = W2 + mmt2 # (1, 4)

    return W1, W2   # (4, 3), (1, 4)
