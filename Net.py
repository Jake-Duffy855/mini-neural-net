import random
import math
import numpy as np
import matplotlib.pyplot as pp


# default function to initialize weights and biases
def rand():
    return random.uniform(-1, 1)


# basic activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# default activation function
def tanh(x):
    return 1 - sigmoid(-2 * x)


# relu
def relu(x):
    return max(0, x)


# calculates gradient of a function at a specific value, used for descent
def gradient(func, x, delta=0.001):
    vectorized_func = np.vectorize(func)
    return (func(x + delta) - func(x)) / delta


# manually initialize weights and biases
# def make_net(layers, nodes, cons, biases, act=relu):
#     net = Net(layers, nodes, act)
#     net.cons = cons
#     net.biases = biases
#
#     return net


class Net:
    def __init__(self, layers, nodes, act=tanh, r=rand):
        self.layers = layers - 1
        self.nodes = nodes
        self.cons = []
        self.biases = []
        self.activate = act
        for layer in range(self.layers):
            self.cons.append(np.random.uniform(1, 1, (nodes[layer + 1], nodes[layer])))
            self.biases.append(np.random.uniform(1, 1, (nodes[layer + 1], 1)))

    # change the activation function
    def set_activation(self, fun):
        self.activate = fun

    # returns a string containing formatted weights and biases arrays
    def __str__(self):
        return str(self.cons) + "\n\n" + str(self.biases)

    # feeds forward an input np array and returns the calculated output
    def get_out(self, inp):
        vectorized_activation = np.vectorize(self.activate)
        for layer in range(self.layers):
            inp = vectorized_activation(inp)
            inp = np.matmul(self.cons[layer], inp) + self.biases[layer]
        return vectorized_activation(inp)

    # trains the net on a given input array, desired output array, and learning rate (default 0.1)
    def train(self, inp, exp, lr=0.1):
        vectorized_activation = np.vectorize(self.activate)
        inputs = [inp]
        for layer in range(self.layers):
            inp = vectorized_activation(inp)
            inp = np.matmul(self.cons[layer], inp) + self.biases[layer]
            inputs.append(inp)

        # inputs has all node values from inp to out (not activated)

        # has been activated
        out = vectorized_activation(inp)

        # errors will have all errors from out to first hidden in that order (activated)
        errors = [exp - out]

        for layer in range(self.layers - 1):
            errors.append(np.matmul(np.transpose(self.cons[-(layer + 1)]), errors[layer]))

        for layer in range(self.layers):
            # adjust connection weights
            self.cons[-(layer + 1)] += np.matmul(
                lr * errors[layer] * gradient(vectorized_activation, inputs[-(layer + 1)]),
                np.transpose(inputs[-(layer + 2)]))
            # adjust bias weights
            self.biases[-(layer + 1)] += lr * errors[layer] * gradient(vectorized_activation, inputs[-(layer + 1)])
