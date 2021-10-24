import numpy as np
import math
import matplotlib.pyplot as plt
from functions import splitData, getAccuracy
from sklearn.metrics import confusion_matrix


class LogisticRegression:
    def __init__(self, x, y, epoch, alpha):
        self.k = len(x[0])
        self.x = x
        self.y = y
        self.epoch = epoch
        self.alpha = alpha

        self.xTrain, self.xVal, self.xTest = splitData(x, 70, 20, 10)
        self.yTrain, self.yVal, self.yTest = splitData(y, 70, 20, 10)

    def hypothesis(self, w, x):
        return np.dot(w, x)

    def s(self, w, xi):
        return 1 / (1 + math.e ** (-self.hypothesis(w, xi)))

    def derivate(self, w, x, y):
        m = len(x)

        dw = [0] * (self.k + 1)

        for i in range(self.k + 1):
            for j in range(m):
                dw[i] += y[j][0] * x[j][i] - self.s(w, x[j]) * x[j][i]

            dw[i] *= (-1 / m)

        return dw

    def error(self, w, x, y):
        err = 0
        m = len(x)

        for i in range(m):
            s = self.s(w, x[i])
            err += (y[i][0] * math.log(s)) + (1 - y[i][0]) * math.log(1 - s)

        err *= (-1 / m)

        return err

    def update(self, w, dw):
        for i in range(self.k + 1):
            w[i] -= (self.alpha * dw[i])

        return w

    def predict(self, w, x):
        return [1 if self.s(w, xi) >= 0.5 else 0 for xi in x]

    def real(self, y):
        return [yi[0] for yi in y]

    def train(self):
        w = [np.random.rand() for i in range(self.k)]
        b = np.random.rand()

        w.append(b)

        self.xTrain = [np.append(row, 1) for row in self.xTrain]
        self.xVal = [np.append(row, 1) for row in self.xVal]
        self.xTest = [np.append(row, 1) for row in self.xTest]

        errorTrain = []
        errorVal = []
        errorTest = []

        for i in range(self.epoch):
            errorTrain.append(self.error(w, self.xTrain, self.yTrain))
            errorVal.append(self.error(w, self.xVal, self.yVal))

            dw = self.derivate(w, self.xTrain, self.yTrain)
            w = self.update(w, dw)

        for i in range(self.epoch):
            errorTest.append(self.error(w, self.xTest, self.yTest))

        predTrain = self.predict(w, self.xTrain)
        predVal = self.predict(w, self.xVal)
        predTest = self.predict(w, self.xTest)

        yTrain = self.real(self.yTrain)
        yVal = self.real(self.yVal)
        yTest = self.real(self.yTest)

        trainMatrix = confusion_matrix(yTrain, predTrain)
        testMatrix = confusion_matrix(yTest, predTest)
        valMatrix = confusion_matrix(yVal, predVal)

        trainAccuracy = getAccuracy(trainMatrix)
        valAccuracy = getAccuracy(valMatrix)
        testAccuracy = getAccuracy(testMatrix)

        print("Train")
        print(trainMatrix)
        print("Accuracy:", trainAccuracy)
        print()
        print("Testing")
        print(testMatrix)
        print("Accuracy:", testAccuracy)
        print()
        print("Validation")
        print(valMatrix)
        print("Accuracy:", valAccuracy)

        # Graph
        plt.plot(errorTrain, label="Training")
        plt.plot(errorVal, label="Validation")
        plt.plot(errorTest, label="Testing")

        plt.legend()
        plt.show()
