import numpy as np
import math
import matplotlib.pyplot as plt
from functions import percentage


class LogisticRegression:
    def __init__(self, x, y, epoch, alpha):
        self.k = len(x[0])
        self.x = x
        self.y = y
        self.epoch = epoch
        self.alpha = alpha

        rowsAmount = len(x)

        self.xTrain = x[:percentage(rowsAmount, 70)]

        self.xValidation = x[percentage(
            rowsAmount, 70):percentage(rowsAmount, 90)]

        self.xTest = x[percentage(rowsAmount, 90):]

    def hypothesis(self, w, x):
        np.append(x, 1)
        # print(len(x))
        return np.dot(w, x)

    def derivate(self, w, x):
        m = len(x)

        dw = [0] * self.k

        for i in range(self.k):
            for j in range(m):
                dw[i] += self.s(w, x[j]) - self.y[j][0]

            dw[i] *= (x[i] / m)

        return dw

    def s(self, w, xi):
        return 1 / (1 + math.e ** (-self.hypothesis(w, xi)))

    def cost(self, w, x):
        err = 0
        m = len(x)

        for i in range(m):
            err += (self.y[i][0] * math.log(self.s(w, x[i]))) + \
                (1 - y[i][0] * math.log(1 - self.s(w, x[i])))

        err *= (-1 / m)
        return err

    def update(self, w, dw):
        for i in range(self.k):
            w[i] -= (self.alpha * dw[i])

        return w

    def train(self):
        w = [np.random.rand() for i in range(self.k)]
        print(len(w))
        b = np.random.rand()
        print(len(w))

        w.append(b)
        # for row in self.xTrain:
        # np.append(row, 1)

        errTrain = self.cost(w, self.xTrain)
        errValidation = self.cost(w, self.xValidation)
        errTest = self.cost(w, self.xTest)

        errorListTrain = [errTrain]
        errorListValidation = [errValidation]
        errorListTest = [errTest]

        for i in range(self.epoch):
            dw = self.derivate(w, self.xTrain)

            w = self.update(w, dw)

            errTrain = self.cost(w, self.xTrain)
            errValidation = self.cost(w, self.xValidation)
            errTest = self.cost(w, self.xTest)

            # Animation
            """
            plt.scatter(i, errTrain, label="Training", color="red")
            plt.scatter(i, errValidation, label="Validation", color="green")
            plt.scatter(i, errTest, label="Testing", color="blue")
            plt.pause(0.0001)
            """

            # Print
            """
            print("epoch:", i)
            print("train:", errTrain)
            print("validation:", errValidation)
            print("test:", errTest)
            print()
            """

            errorListTrain.append(errTrain)
            errorListValidation.append(errValidation)
            errorListTest.append(errTest)

        # Graph
        plt.plot(errorListTrain, label="Training")
        plt.plot(errorListValidation, label="Validation")
        plt.plot(errorListTest, label="Testing")

        plt.legend()

        """
        ys = [self.hypothesis(w, b, xi) for xi in self.x]
        plt.plot(self.y, '*')
        plt.plot(ys, '*')
        """
        plt.show()
