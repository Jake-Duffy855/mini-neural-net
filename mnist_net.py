import Net as nn
import numpy as np
import mnist
import matplotlib.pyplot as plt
import random
from tkinter import *
import math

master = Tk()
canvas_width = 280
canvas_height = 280

# get the images and labels
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()


# plt.imshow(test_images[0], cmap='gray')
# plt.show()

# normalize pixel values from [0, 255] tp [0, 1]
train_images = train_images / 255
test_images = test_images / 255

# make the images 1D lists
train_images = list(train_images.reshape(-1, 784, 1))
test_images = list(test_images.reshape(-1, 784, 1))

# make the labels lists
train_labels = list(train_labels)
test_labels = list(test_labels)

my_net = nn.Net(4, [784, 128, 128, 10])


# get the desired output array for the given number
def get_output_array(n):
    result = [[0] for i in range(10)]
    result[n][0] = 1
    return np.array(result)


# get the answer from the net output
def answer(output):
    output = list(output.reshape(10))
    return output.index(max(output))


# return the accuracy of the net over all test images
def get_accuracy(net, divisor=20):
    correct = 0
    for i in range(int(len(test_images) / divisor)):
        if answer(net.get_out(test_images[i])) == test_labels[i]:
            correct += 1
    return divisor * correct / len(test_images)


# train the given net on the training images
def train_net(net, iterations=1, divisor=1):
    trains = int(len(train_images) / divisor)
    for iteration in range(iterations):
        for i in range(trains):
            for j in range(1):
                net.train(train_images[i], get_output_array(train_labels[i]),
                          0.004 * trains / (trains + trains * iteration + i))
            if i % 1000 == 0:
                acc = get_accuracy(my_net, 40)
                print(str(iteration) + ", " + str(i) + " : " + str(acc))
                plt.plot(i, acc, 'bo')



train_net(my_net, 1, 4)
print(get_accuracy(my_net, 1))

f = open("net.txt", "w")
f.write(my_net.__str__())
f.close()

for i in range(2):
    print(answer(my_net.get_out(test_images[i]))) #, test_labels[i])
    picture = test_images[i]
    first_image = np.array(test_images[i], dtype=float)
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


# -----------------------------------------------------------------------

inp = np.zeros((784, 1))


def clear():
    global inp
    c.delete(ALL)
    inp = np.zeros((784, 1))


def submit():
    print(answer(my_net.get_out(inp)))


def paint(event):
    color = 'white'
    x1, y1 = (event.x, event.y)
    x1 = int(x1 / 10)
    y1 = int(y1 / 10)
    x2 = x1 + 1
    y2 = y1 + 1
    c.create_rectangle(x1 * 10, y1 * 10, x2 * 10, y2 * 10, fill=color, outline=color)
    c.create_rectangle(x1 * 10, (y1 + 1) * 10,
                       x2 * 10, (y2 + 1) * 10, fill=color, outline=color)
    c.create_rectangle(x1 * 10, (y1 - 1) * 10,
                       x2 * 10, (y2 - 1) * 10, fill=color, outline=color)
    c.create_rectangle((x1 + 1) * 10, y1 * 10,
                       (x2 + 1) * 10, y2 * 10, fill=color, outline=color)
    c.create_rectangle((x1 - 1) * 10, y1 * 10,
                       (x2 - 1) * 10, y2 * 10, fill=color, outline=color)
    inp[x1 + y1 * 28, 0] = 1
    inp[x1 + (y1 + 1) * 28, 0] = min(1, inp[x1 + (y1 + 1) * 28, 0] + 0.2)
    inp[x1 + (y1 - 1) * 28, 0] = min(1, inp[x1 + (y1 - 1) * 28, 0] + 0.2)
    inp[(x1 + 1) + y1 * 28, 0] = min(1, inp[x1 + (y1 + 1) * 28, 0] + 0.2)
    inp[(x1 - 1) + y1 * 28, 0] = min(1, inp[x1 + (y1 - 1) * 28, 0] + 0.2)


c = Canvas(master,
           width=canvas_width,
           height=canvas_height,
           bg='black')
c.pack(expand=YES, fill=BOTH)
c.bind('<B1-Motion>', paint)
clear_button = Button(master, command=clear, text="Clear")
clear_button.pack()
submit_button = Button(master, command=submit, text="Submit")
submit_button.pack()
mainloop()
