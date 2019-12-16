import random
import math
import matplotlib.pyplot as pp


def rand():
    return random.uniform(-1, 1)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def gradient(func, x, delta=0.001):
    return (func(x + delta) - func(x)) / delta


def make_net(layers, nodes, cons, biases, act=sigmoid):
    net = Net(layers, nodes, act)
    net.cons = cons
    net.biases = biases

    return net


class Net:
    def __init__(self, layers, nodes, act=sigmoid):
        self.layers = layers - 1
        self.nodes = nodes
        self.cons = []
        self.biases = []
        self.activate = act
        for layer in range(self.layers):
            self.cons.append([[rand() for i in range(nodes[layer])] for j in range(nodes[layer + 1])])
            self.biases.append([rand() for i in range(nodes[layer + 1])])

    def set_activation(self, fun):
        self.activate = fun

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

    def get_out(self, inp):

        for layer in range(self.layers):
            inp = list(map(self.activate, inp))
            nxt = []
            for i in range(self.nodes[layer + 1]):
                s = self.biases[layer][i]
                for j in range(self.nodes[layer]):
                    s += self.cons[layer][i][j] * inp[j]
                nxt.append(s)
            inp = nxt

        return list(map(self.activate, nxt))

    def train(self, inp, exp, lr):
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
        errors = [[(exp[i] - out[i]) for i in range(len(out))]]

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