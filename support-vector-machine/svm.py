import numpy as np
import matplotlib.pyplot as plt
from functions import splitData, getAccuracy
from sklearn.metrics import confusion_matrix


class SVM:
    def __init__(self, x, y, epoch, alpha, lagrage, C):
        self.k = len(x[0])
        self.x = x
        self.y = y
        self.epoch = epoch
        self.alpha = alpha
        self.lagrage = lagrage
        self.C = C

        rowsAmount = len(y)

        self.xTrain, self.xVal, self.xTest = splitData(x, 70, 20, 10)
        self.yTrain, self.yVal, self.yTest = splitData(y, 70, 20, 10)

    def hypothesis(self, w, b, x):
        return np.dot(w, x) + b

    def error(self, w, b, x, y):
        err = 0

        m = len(x)
        for i in range(m):
            err += max(0, 1 - y[i] * self.hypothesis(w, b, x[i]))

        err = (np.linalg.norm(w, 2)) / 2 + self.lagrage * err

        return err

    def derivate(self, w, b, x, y, column):
        dw = 0
        db = 0

        if y * self.hypothesis(w, b, x) > 1:
            dw = w[column]
        else:
            dw = w[column] - y * x[column] * self.C
            db = (-1) * y * self.C

        return dw, db

    def predict(self, w, b, x):
        return [1 if self.hypothesis(w, b, xi) >= 0 else -1 for xi in x]

    def real(self, y):
        return [yi[0] for yi in y]

    def train(self):
        w = [np.random.rand() for i in range(self.k)]
        b = np.random.rand()

        errorTrain = []
        errorVal = []
        errorTest = []

        for _ in range(self.epoch):
            errorTrain.append(self.error(w, b, self.xTrain, self.yTrain))
            errorVal.append(self.error(w, b, self.xVal, self.yVal))
            errorTest.append(self.error(w, b, self.xTest, self.yTest))

            randomRow = np.random.randint(len(self.xTrain))
            x = self.xTrain[randomRow]
            y = self.yTrain[randomRow][0]

            for i in range(self.k):
                dw, db = self.derivate(w, b, x, y, i)

                w[i] -= self.alpha * dw
                b -= self.alpha * db

        predTrain = self.predict(w, b, self.xTrain)
        predVal = self.predict(w, b, self.xVal)
        predTest = self.predict(w, b, self.xTest)

        yTrain = self.real(self.yTrain)
        yVal = self.real(self.yVal)
        yTest = self.real(self.yTest)

        trainMatrix = confusion_matrix(yTrain, predTrain)
        valMatrix = confusion_matrix(yVal, predVal)
        testMatrix = confusion_matrix(yTest, predTest)

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

        plt.plot(errorTrain, label="Training")
        plt.plot(errorVal, label="Validation")
        plt.plot(errorTest, label="Testing")

        plt.legend()
        plt.show()
