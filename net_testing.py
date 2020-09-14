import Net as n
import random
import math
import numpy as np
import matplotlib.pyplot as pp

lr = 0.05
a = n.Net(4, [2, 4, 3, 1])
a.set_activation(n.tanh)

print(a.get_out([[0], [0]]))
print(a.get_out([[0], [1]]))
print(a.get_out([[1], [0]]))
print(a.get_out([[1], [1]]))
print(a.get_out([[0.75], [0]]))
print()

for i in range(5000):
    random.choice([a.train([[0], [0]], [[0]], lr),
                   a.train([[1], [0]], [[1]], lr),
                   a.train([[0], [1]], [[1]], lr),
                   a.train([[1], [1]], [[0]], lr)])
    if i % 50 == 0:
        pp.plot(i, a.get_out([[0], [1]]), 'bo')

print(a.get_out([[0], [0]]))
print(a.get_out([[0], [1]]))
print(a.get_out([[1], [0]]))
print(a.get_out([[1], [1]]))
print(a.get_out([[0.75], [0]]))
pp.show()

