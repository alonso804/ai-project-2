import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, x, y, epoch, alpha, C):
        self.k = len(x[0])
        self.x = x
        self.y = y
        self.epoch = epoch
        self.alpha = alpha
        self.rowsAmount = len(y)
        self.C = C

    def hypothesis(self, w, b, x):
        return np.dot(w, x) + b

    def error(self, w, b, x):
        err = 0
        for i in range(len(self.x)):
            err += max(0, 1-self.y[i]*self.hypothesis(w, b, x[i]))

        err = (np.linalg.norm(w, 1))/2 + self.C * err
        return err

    def train(self):
        w = [np.random.rand() for i in range(self.k)]
        b = np.random.rand()
        errList = []
        for i in range(self.epoch):
            print(self.error(w, b, self.x))
            errList.append(self.error(w, b, self.x))
            p = int(np.random.randint(self.rowsAmount))
            xp = self.x[p]
            yp = self.y[p]
            for j in range(len(w)):
                if yp * (np.dot(w, xp) + b) > 1:
                    w[j] = w[j] - self.alpha * w[j]
                else:
                    w[j] = w[j] - self.alpha * (w[j] - yp * xp[j]) * self.C
                    b = b - self.alpha * (-1 * yp * self.C)


        plt.plot(errList)
        plt.show()
            # Print
            # print("epoch:", i)
            # print("train:", errTrain)
            # print("validation:", errValidation)
            # print("test:", errTest)
            # print()


            # errorListTrain.append(errTrain)
            # errorListValidation.append(errValidation)
            # errorListTest.append(errTest)

        # Graph
        # plt.plot(errorListTrain, label="Training")
        # plt.plot(errorListValidation, label="Validation")
        # plt.plot(errorListTest, label="Testing")

        # plt.legend()

        #ys = [self.hypothesis(w, b, xi) for xi in self.x]
        #plt.plot(self.y, '*')
        #plt.plot(ys, '*')
        #plt.show()
        # plt.show()
