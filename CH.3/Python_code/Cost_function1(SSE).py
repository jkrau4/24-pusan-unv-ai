import numpy as np
import matplotlib.pyplot as plt

def b(x):
    y = x**2/2
    return y

x = np.linspace(-1, 1, 1000)

plt.plot(x,b(x))
plt.xlabel('d - y')
plt.ylabel('E')
plt.show()