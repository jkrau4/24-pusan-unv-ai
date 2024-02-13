import numpy as np
import matplotlib.pyplot as plt

def b(x):
    y = -1 * np.log(1 - x)
    return y

x = np.linspace(0.001, 1, 1000)

plt.plot(x,b(x))
plt.xlabel('y')
plt.ylabel('E')
plt.show()