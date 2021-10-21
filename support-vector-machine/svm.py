import numpy as np
import matplotlib.pyplot as plt
from functions import percentage
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

        self.xTrain = x[:percentage(rowsAmount, 70)]
        self.xValidation = x[percentage(
            rowsAmount, 70):percentage(rowsAmount, 90)]
        self.xTest = x[percentage(rowsAmount, 90):]

        self.yTrain = y[:percentage(rowsAmount, 70)]
        self.yValidation = y[percentage(
            rowsAmount, 70):percentage(rowsAmount, 90)]
        self.yTest = y[percentage(rowsAmount, 90):]

    def hypothesis(self, w, b, x):
        return np.dot(w, x) + b

    def error(self, w, b, x):
        err = 0

        m = len(x)
        for i in range(m):
            err += max(0, 1 - self.y[i] * self.hypothesis(w, b, x[i]))

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

    def train(self):
        w = [np.random.rand() for i in range(self.k)]
        b = np.random.rand()
        errorListTrain = []
        errorListValidation = []
        errorListTest = []

        for _ in range(self.epoch):
            # errorListTrain.append(self.error(w, b, self.xTrain))
            # errorListValidation.append(self.error(w, b, self.xValidation))
            # errorListTest.append(self.error(w, b, self.xTest))

            randomRow = np.random.randint(len(self.xTrain))
            x = self.xTrain[randomRow]
            y = self.yTrain[randomRow][0]

            for i in range(self.k):
                dw, db = self.derivate(w, b, x, y, i)

                w[i] -= self.alpha * dw
                b -= self.alpha * db

        predTrain = [1 if self.hypothesis(
            w, b, xi) >= 0 else -1 for xi in self.xTrain]
        predValidation = [1 if self.hypothesis(
            w, b, xi) >= 0 else -1 for xi in self.xValidation]
        predTest = [1 if self.hypothesis(
            w, b, xi) >= 0 else -1 for xi in self.xTest]
        predValidation = [1 if self.hypothesis(
            w, b, xi) >= 0 else -1 for xi in self.xValidation]

        yTrain = [yi[0] for yi in self.yTrain]
        yValidation = [yi[0] for yi in self.yValidation]
        yTest = [yi[0] for yi in self.yTest]

        trainMatrix = confusion_matrix(yTrain, predTrain)
        testMatrix = confusion_matrix(yTest, predTest)
        validationMatrix = confusion_matrix(yValidation, predValidation)

        trainAccuracy = (trainMatrix[0][1] +
                         trainMatrix[1][0]) / len(predTrain)
        testAccuracy = (testMatrix[0][1] + testMatrix[1][0]) / len(predTest)
        validationAccuracy = (
            validationMatrix[0][1] + validationMatrix[1][0]) / len(predValidation)

        trainAccuracy = 100 - trainAccuracy
        testAccuracy = 100 - testAccuracy
        validationAccuracy = 100 - validationAccuracy

        print("Train")
        print(trainMatrix)
        print("Accuracy:", trainAccuracy)
        print()
        print("Testing")
        print(testMatrix)
        print("Accuracy:", testAccuracy)
        print()
        print("Validation")
        print(validationMatrix)
        print("Accuracy:", validationAccuracy)

        # plt.plot(errorListTrain, label="Training")
        # plt.plot(errorListValidation, label="Validation")
        # plt.plot(errorListTest, label="Testing")

        # plt.legend()
        # plt.show()
