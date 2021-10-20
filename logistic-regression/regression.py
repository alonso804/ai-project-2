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

    def hypothesis(self, w, b, x):
        return np.dot(w, x) + b

    def derivate(self, w, b, x):
        m = len(x)

        db = 0

        for i in range(m):
            db += (self.y[i][0] - self.hypothesis(w, b, x[i])) * (-1)

        db /= m

        dw = [0] * self.k

        for i in range(self.k):
            for j in range(m):
                dw[i] += self.s(w, b, x[j]) - self.y[j][0]

            dw[i] *= (x[i] / m)

        return db, dw

    def s(self, w, b, xi):
        return 1 / (1 + math.e ** (-self.hypothesis(w, b, xi)))

    def error(self, w, b, x):
        err = 0
        m = len(x)

        for i in range(m):
            err += (self.y[i][0] * math.log(self.s(w, b, x[i])) + (1 - y[i][0]) * math.log(1 - self.s(w, b, x[i]))

        err *= (- 1 / m)

        return err

    def update(self, b, db, w, dw):
        for i in range(self.k):
            w[i] -= (self.alpha * dw[i])

        b -= (self.alpha * db)

        return b, w

    def train(self):
        w=[np.random.rand() for i in range(self.k)]
        b=np.random.rand()

        errTrain=self.error(w, b, self.xTrain)
        errValidation=self.error(w, b, self.xValidation)
        errTest=self.error(w, b, self.xTest)

        errorListTrain=[errTrain]
        errorListValidation=[errValidation]
        errorListTest=[errTest]

        for i in range(self.epoch):
            db, dw=self.derivate(w, b, self.xTrain)

            b, w=self.update(b, db, w, dw)

            errTrain=self.error(w, b, self.xTrain)
            errValidation=self.error(w, b, self.xValidation)
            errTest=self.error(w, b, self.xTest)

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
