import Net as n
import random
import math

lr = 0.1
a = n.Net(5, [2, 4, 4, 3, 1])
# a.set_activation(lambda x: math.log10(abs(x)+0.1))

for i in range(50000):
    random.choice([a.train([0, 0], [1], lr),
                   a.train([0, 1], [0], lr),
                   a.train([1, 0], [0], lr),
                   a.train([1, 1], [1], lr)])

print(a.get_out([0, 0]))
print(a.get_out([0, 1]))
print(a.get_out([1, 0]))
print(a.get_out([1, 1]))
print(a.get_out([0.5, 0.5]))
print()
print(a)

# b = n.Net(3, [2, 4, 1])
#
# print(b.get_out([0, 0]))
# print(b.get_out([0, 1]))
# print(b.get_out([1, 0]))
# print(b.get_out([1, 1]))
# print()
#
# b = n.make_net(3, [2, 4, 1], a.cons, a.biases)
#
# print(b.get_out([0, 0]))
# print(b.get_out([0, 1]))
# print(b.get_out([1, 0]))
# print(b.get_out([1, 1]))
# print()
