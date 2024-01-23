def BackpropMnt(W1, W2, X, D):
    alpha = 0.9
    beta = 0.9

    mnt1 = np.zeros(W1)
    mnt2 = np.zeros(W2)

    N = 4
    for k in range(N):
        x = X[k, :].T
        d = D[k]

        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v = np.matmul(W2, y1)
        y = Sigmoid(v)

        e = d - y
        delta = y*(1-y)*e

        e1 = W2*delta
        delta1 = y1*(1-y1)*e1
        dW1 = alpha*delta*x.T
        mmt1 = dW1 + beta*mmt1
        W1 = W1 + mnt1

        dW2 = alpha*delta*y1.T
        mmt2 = dW2 + beta*mmt2
        W2 = W2 + mnt2
    return W1, W2
