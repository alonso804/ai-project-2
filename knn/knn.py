from rtree import index
from functions import splitData, getAccuracy, getError, clearFiles
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns


class KNN:
    def __init__(self, x, y):
        self.p = index.Property()
        self.p.dimension = 7
        self.p.dat_extension = 'data'
        self.p.idx_extension = 'index'
        self.idx = index.Index('7d', properties=self.p)
        self.x = x
        self.y = y

        # self.insertAll(self.x, self.y)

    def insertAll(self, x, y):
        for i, coordinates in enumerate(x):
            point = tuple(coordinates)
            point += point
            self.idx.insert(int(y[i][0]), point)

    def clear(self):
        clearFiles()
        self.idx = index.Index('7d', properties=self.p)

    def countNeighbors(self, neighbors):
        count = {1: 0, 0: 0}

        for neighbor in neighbors:
            count[neighbor] += 1

        firstValue = count[neighbors[0]]

        equal = all(value == firstValue for value in count.values())

        return equal, count

    def knn(self, k, coordinates):
        point = coordinates
        point += coordinates

        neighbors = list(self.idx.nearest(point, k))

        equal, count = self.countNeighbors(neighbors)

        while equal == True:
            k += 1
            neighbors = list(self.idx.nearest(point, k))
            equal, count = self.countNeighbors(neighbors)

        predict = max(count, key=count.get)

        return predict

    def predict(self, k, idx):
        return [self.knn(k, tuple(self.x[i])) for i in idx]

    def real(self, idx):
        return [self.y[i] for i in idx]

    def report(self, matrix):
        print(matrix)
        print("Accuracy: ", getAccuracy(testMatrix))
        print()

    def kFoldCrossValidation(self, folds, k):
        kf = KFold(n_splits=folds)
        n = self.x.shape[0]

        errors = []

        for trainIdx, testIdx in kf.split(range(n)):
            for i in trainIdx:
                point = tuple(self.x[i])
                point += point
                self.idx.insert(int(self.y[i][0]), point)

            predTest = self.predict(k, testIdx)
            yReal = self.real(testIdx)

            testMatrix = confusion_matrix(yReal, predTest)
            # self.report(testMatrix)
            error = getError(testMatrix)
            errors.append(error)
            self.clear()

        average = sum(errors) / len(errors)
        print(f'k: {k}')
        print(f'\taverage: {average}')
        print()
        fig = plt.figure()
        plt.figure().clear()
        sns.distplot(errors)
        plt.savefig(f'./test/graphs/k_{k}_average_{round(average, 2)}.png')
