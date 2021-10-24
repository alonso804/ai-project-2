from rtree import index
from functions import splitData, getAccuracy
from sklearn.metrics import confusion_matrix


class KNN:
    def __init__(self, x, y):
        p = index.Property()
        p.dimension = 7
        p.dat_extension = 'data'
        p.idx_extension = 'index'
        self.idx = index.Index('7d', properties=p)
        self.x = x
        self.y = y

        self.xTrain, self.xVal, self.xTest = splitData(x, 70, 20, 10)
        self.yTrain, self.yVal, self.yTest = splitData(y, 70, 20, 10)

        self.insertAll(self.xTrain, self.yTrain)

    def insertAll(self, x, y):
        for i, coordinates in enumerate(x):
            point = tuple(coordinates)
            point += point
            self.idx.insert(int(y[i][0]), point)

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

    def predict(self, x):
        K_TEST = 5
        return [self.knn(K_TEST, tuple(xi)) for xi in x]

    def real(self, y):
        return [yi[0] for yi in y]

    def testing(self):
        predTest = self.predict(self.xTest)
        yTest = self.real(self.yTest)

        testMatrix = confusion_matrix(yTest, predTest)
        testAccuracy = getAccuracy(testMatrix)

        print("Testing")
        print(testMatrix)
        print("Accuracy:", testAccuracy)
