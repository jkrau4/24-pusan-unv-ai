import numpy as np
from time import time, sleep
start = time()
def Sigmoid(x):
    return 1 / (1 +np.exp(-x))

def BackpropCE(W1, W2, X, D):
    alpha = 0.9
    N = 4
    for k in range(N):
        x = X[k, :].T
        d = D[k]

        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v = np.matmul(W2, y1)
        y = Sigmoid(v)

        e = d - y
        delta = e

        e1 = np.matmul(W2.T, delta)
        delta1 = y1 * (1 - y1) * e1

        dW1 = (alpha * delta1).reshape(4,1) * x.T
        W1 = W1 + dW1

        dW2 = alpha * delta * y1
        W2  = W2 + dW2

    return W1, W2

def TestBackpropCE():
  X = np.array([[0, 0, 1],
               [0, 1, 1],
               [1, 0, 1],
               [1, 1, 1]])

  D = np.array([[0],
              [1],
              [1],
              [0]])

  W1 = 2*np.random.random((4, 3)) - 1
  W2 = 2*np.random.random((1, 4)) - 1

  for epoch in range(10000):
     W1, W2 = BackpropCE(W1, W2, X, D)

  N = 4
  for k in range(N):
      x = X[k ,:].T
      v1 = np.matmul(W1, x)
      y1 = Sigmoid(v1)
      v = np.matmul(W2, y1)
      y = Sigmoid(v)
      print(y)
TestBackpropCE()

end= time()
print('time elapsed CE:', end - start)

start = time()

def Sigmoid(x):
    return 1 / (1 +np.exp(-x))

def BackpropXOR(W1,W2,X,D):
    alpha = 0.9

    N = 4
    for k in range(N):
        x = X[k,:].T
        d = D[k]

        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v = np.matmul(W2, y1)
        y = Sigmoid(v)

        e = d - y
        delta = y * (1-y) * e

        e1 = W2 * delta
        delta1 = y1 * (1-y1) * e1
        
        dW1 = (alpha * delta1).reshape(4, 1) * x.T
        W1 = W1 + dW1

        dW2 = alpha * delta * y1.T
        W2 = W2 + dW2
    return W1, W2

def TestBackpropXOR():
  X = np.array([[0, 0, 1],
               [0, 1, 1],
               [1, 0, 1],
               [1, 1, 1]])

  D = np.array([[0],
              [1],
              [1],
              [0]])

  W1 = 2*np.random.random((4, 3)) - 1
  W2 = 2*np.random.random((1, 4)) - 1

  for epoch in range(10000):
     W1, W2 = BackpropXOR(W1, W2, X, D)

  N = 4
  for k in range(N):
      x = X[k ,:].T
      v1 = np.matmul(W1, x)
      y1 = Sigmoid(v1)
      v = np.matmul(W2, y1)
      y = Sigmoid(v)
      print(y)
TestBackpropXOR()

end= time()
print('time elapsed SSE:', end - start)
