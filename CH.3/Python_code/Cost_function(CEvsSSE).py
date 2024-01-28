import numpy as np
import matplotlib.pyplot as plt

def b(x):
    y = (1/2 - x)**2/2
    return y


def c(x):
    y = (-1* np.log(x))/2 - (1 * np.log(1 - x))/2
    return y

x = np.linspace(0.001, 1, 1000)

CE, = plt.plot(c(x), 'r')
SSE, = plt.plot(b(x), 'b')
plt.xlabel('y')
plt.ylabel('E')
plt.legend([CE, SSE], ['CE', 'SSE'])
plt.show()

