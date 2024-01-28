import numpy as np
import matplotlib.pyplot as plt

def a(x):
    y = -1* np.log(x)
    return y

x = np.linspace(0.001, 1, 1000)

plt.plot(x,a(x))
plt.xlabel('y')
plt.ylabel('E')
plt.show()