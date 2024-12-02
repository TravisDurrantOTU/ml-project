import matplotlib.pyplot as plt
import numpy as np

gridpts = np.arange(-5, 5.0000000001, step=0.001)
figure,axis = plt.subplots()
axis.plot(gridpts, np.maximum(0, gridpts), '-k')
axis.set_title('y = ReLU(x)')
axis.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
plt.show()
figure,axis = plt.subplots()
axis.plot(gridpts, np.tanh(gridpts), '-k')
axis.set_title('y = tanh(x)')
axis.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
plt.show()