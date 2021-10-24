from rtree import index
from functions import splitData


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

    def knn(self, k, coordinates):
        point = coordinates
        point += coordinates

        neighbors = list(self.idx.nearest(point, k))

        count = {}

        for neighbor in neighbors:
            if neighbor in count:
                count[neighbor] += 1
            else:
                count[neighbor] = 1

        print(neighbors)
        print(count)
        # print(list(self.idx.nearest(point, k)))

    def testing(self):
        pass
