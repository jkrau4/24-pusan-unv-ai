import numpy as np
import matplotlib.pyplot as plt
from Sigmoid import Sigmoid
from DeltaSGD import DeltaSGD
from DeltaBatch import DeltaBatch

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

D = np.array([[0],
              [0],
              [1],
              [1]])

E1 = []
E2 = []

W1 = 2*np.random.random((1, 3)) - 1
W2 = np.array(W1)

for epoch in range(1000):
    W1 = DeltaSGD(W1, X, D)
    W2 = DeltaBatch(W2, X, D)
    
    es1 = 0
    es2 = 0
    N   = 4
    for k in range(N):
        x = X[k, :].T
        d = D[k]
        
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        es1 = es1 + (d - y1)**2
        
        v2 = np.matmul(W2, x)
        y2 = Sigmoid(v2)
        es2 = es2 + (d - y2)**2
        
    E1.append(es1/N)
    E2.append(es2/N)

SGD, = plt.plot(E1, 'r')
Batch, = plt.plot(E2, 'b:')
plt.xlabel("Epoch")
plt.ylabel("Average of Training Error")
plt.legend([SGD, Batch], ['SGD', 'Batch'])
plt.show()