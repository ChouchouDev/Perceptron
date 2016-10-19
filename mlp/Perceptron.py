# encoding: utf-8
'''
Created on 13 oct. 2016

@author: Miao1
'''
import numpy as np
import matplotlib.pyplot as plt

'''
Having .txt:

1	4.383	-4.114
-1	-2.223	1.915
1	2.793	3.335
-1	0.386	-4.508
1	1.649	-3.579
-1	-2.638	-4.973
1	3.69	-4.941
-1	-4.46	-1.574

'''

class Perceptron:
    def __init__(self, learningRate, nbIteration, dataFicher):
        self.learningRate = learningRate
        self.nbIteration = nbIteration
        self.data = np.loadtxt(dataFicher)

        self.x = self.data[:, 1:3]
        self.weight = np.random.random((2, 1))
        self.bias = 0
        self.z = self.data[:, 0:1]

    def train(self):
        for iter in range(self.nbIteration):
            # forward
            l0 = self.x
            l1 = sign(np.dot(l0, self.weight) + self.bias)

            # loss
            loss = self.z - l1

            if all(item[0] == 0 for item in loss):
                print("finish if we find a possible solution")
                break

            # back, update weights et bias
            self.weight = self.weight + np.dot(l0.T, loss * self.learningRate)
            self.bias = self.bias + sum(loss)*self.learningRate

        print("After our trainment:")
        print('l1')
        print(l1)
        print('weight, bias')
        print(str(self.weight) + str(self.bias))

    def dessiner(self):
        fig = plt.figure()
        positive = []
        negative = []
        for i in range(0, len(self.z)):
            if self.z[i] == 1:
                positive.append(self.x[i,:])
            else:
                if self.z[i] == -1:
                    negative.append(self.x[i,:])

        plt.plot(np.array(positive)[:, 0], np.array(positive)[:, 1], 'or')
        plt.plot(np.array(negative)[:, 0], np.array(negative)[:, 1], 'og')

        x1 = np.linspace(-10, 10, 20)
        x2 = -(x1 * self.weight[0] + self.bias) / self.weight[1]
        plt.plot(x1, x2, 'b')
        plt.show()

    def sign(x):
        resultat = []
        for i in x:
            if i >= 0:
                resultat.append(1)
            elif i < 0:
                resultat.append(-1)

        return np.array([resultat]).T


if __name__  == "__main__":
    p=Perceptron(0.01,1000,"point4.txt")
    p.train()
    p.dessiner()

