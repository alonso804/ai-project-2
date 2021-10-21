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

        self.yTrain = y[:percentage(rowsAmount, 70)]
        self.yValidation = y[percentage(
            rowsAmount, 70):percentage(rowsAmount, 90)]
        self.yTest = y[percentage(rowsAmount, 90):]

    def hypothesis(self, w, x):
        return np.dot(w, x)

    def derivate(self, w, x, y):
        m = len(x)
        print(m)

        dw = [0] * (self.k + 1)

        for i in range(self.k + 1):
            for j in range(m):
                dw[i] += self.s(w, x[j]) - y[j][0]

            dw[i] *= (x[i] / m)

        print(dw)
        return dw

    def s(self, w, xi):
        return 1 / (1 + math.e ** (-self.hypothesis(w, xi)))

    def cost(self, w, x, y):
        # print(x)
        err = 0
        m = len(x)

        # Por aqui esta el error =======================================
        for i in range(m):
            s = self.s(w, x[i])
            # if i == 0:
            # print(w)
            # print(x[i])
            # print(i, s)
            # print()
            err += (y[i][0] * math.log(s)) + \
                (1 - y[i][0] * math.log(1 - s))

        err *= (-1 / m)

        return err

    def update(self, w, dw):
        for i in range(self.k + 1):
            w[i] -= (self.alpha * dw[i])

        return w

    def train(self):
        w = [np.random.rand() for i in range(self.k)]
        b = np.random.rand()

        w.append(b)

        self.xTrain = [np.append(row, 1) for row in self.xTrain]

        errorListTrain = [self.cost(w, self.xTrain, self.yTrain)]
        # errorListValidation = [
        # self.cost(w, self.xValidation, self.yValidation)]
        # errorListTest = [self.cost(w, self.xTest, self.yTest)]

        for i in range(self.epoch):
            dw = self.derivate(w, self.xTrain, self.yTrain)

            w = self.update(w, dw)

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

            errorListTrain.append(self.cost(w, self.xTrain, self.yTrain))
            # errorListValidation.append(self.cost(w, self.xValidation, self.yValidation))
            # errorListTest.append(self.cost(w, self.xTest, self.yTest))

        # Graph
        # plt.plot(errorListTrain, label="Training")
        # plt.plot(errorListValidation, label="Validation")
        # plt.plot(errorListTest, label="Testing")

        # plt.legend()

        """
        ys = [self.hypothesis(w, b, xi) for xi in self.x]
        plt.plot(self.y, '*')
        plt.plot(ys, '*')
        """
        # plt.show()
