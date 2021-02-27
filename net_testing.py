
import Net as n
import random
import math
import numpy as np
import matplotlib.pyplot as pp

lr = 0.05
a = n.Net([2, 4, 3, 1])
a.activate[-1] = n.tanh
print(a.activate)
# a.set_activation(n.tanh)

print(a.get_out([[0], [0]]))
print(a.get_out([[0], [1]]))
print(a.get_out([[1], [0]]))
print(a.get_out([[1], [1]]))
print(a.get_out([[0.75], [0]]))
print()

for i in range(10000):
    random.choice([a.train(np.array([[0], [0]]), np.array([[0]]), lr),
                   a.train(np.array([[1], [0]]), np.array([[1]]), lr),
                   a.train(np.array([[0], [1]]), np.array([[1]]), lr),
                   a.train(np.array([[1], [1]]), np.array([[0]]), lr)])
    if i % 50 == 0:
        pp.plot(i, a.get_out([[0], [1]]), 'bo')

print(a.get_out([[0], [0]]))
print(a.get_out([[0], [1]]))
print(a.get_out([[1], [0]]))
print(a.get_out([[1], [1]]))
print(a.get_out([[0.75], [0]]))
pp.show()