import numpy as np
from Sigmoid import Sigmoid

def BackpropXOR(w1,W2,X,D):
    alpha = 0.9

    N = 4
    for k in range(N):
        x = X[k,:]
        d = D[k]

        v1 = w1 * x
        y1 = Sigmoid(v1)
        v = W2 * y1
        y = Sigmoid(v)

        e = d - y
        delta = y * (1-y) * e

        e1 = W2 * delta
        delta1 = y1 * (1-y1) * e1
        
        dw1 = alpha * delta1 * x
        w1 = w1 + dw1

        dw2 = alpha * delta * y1
        w2 = w2 + dw2
    return w1, w2