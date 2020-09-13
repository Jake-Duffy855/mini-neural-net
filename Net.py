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
            self.cons.append(np.random.rand(nodes[layer], nodes[layer + 1]))
            self.biases.append(np.random.rand(nodes[layer + 1], 1))

    # change the activation function
    def set_activation(self, fun):
        self.activate = fun

    # returns a string containing formatted weights and biases arrays
    def __str__(self):
        result = "["
        for i in range(len(self.cons)):
            result += "["
            for j in range(len(self.cons[i])):
                result += str(self.cons[i][j])
                if j != len(self.cons[i]) - 1:
                    result += ",\n"
            result += "]"
            if i != len(self.cons) - 1:
                result += ",\n"
        result += "]"

        result += "\n\n["
        for i in range(len(self.biases)):
            result += str(self.biases[i])
            if i != len(self.biases) - 1:
                result += ",\n"
        result += "]"

        return result

    # feeds forward an input np array and returns the calculated output
    def get_out(self, inp):
        for layer in range(self.layers):
            inp = self.activate(inp)
            inp = np.matmul(self.cons[layer], inp) + self.biases[layer]
        return self.activate(inp)

    # trains the net on a given input array, desired output array, and learning rate (default 0.1)
    def train(self, inp, exp, lr=0.1):
        inputs = [inp]
        for layer in range(self.layers):
            inp = list(map(self.activate, inp))
            nxt = []
            for i in range(self.nodes[layer + 1]):
                s = self.biases[layer][i]
                for j in range(self.nodes[layer]):
                    s += self.cons[layer][i][j] * inp[j]
                nxt.append(s)
            inp = nxt
            inputs.append(nxt)

        # inputs has all node values from inp to out (not activated)

        # has been activated
        out = list(map(self.activate, nxt))

        # errors will have all errors from out to first hidden in that order (activated)
        errors = [[(exp[i] - out[i]) for i in range(max(len(out), len(exp)))]]

        for layer in range(self.layers - 1):
            err = []
            for i in range(self.nodes[-(layer + 2)]):
                s = 0
                for j in range(self.nodes[-(layer + 1)]):
                    s += self.cons[-(layer + 1)][j][i] * errors[layer][j]
                err.append(s)
            errors.append(err)

        for layer in range(self.layers):
            for i in range(self.nodes[-(layer + 1)]):
                for j in range(self.nodes[-(layer + 2)]):
                    self.cons[-(layer + 1)][i][j] += \
                        lr * errors[layer][i] * gradient(self.activate, inputs[-(layer + 1)][i])\
                        * self.activate(inputs[-(layer + 2)][j])
                self.biases[-(layer + 1)][i] += \
                    lr * errors[layer][i] * gradient(self.activate, inputs[-(layer + 1)][i])
