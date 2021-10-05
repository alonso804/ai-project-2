
import numpy as np
from regression import SVM
import csv

clasification = {
    "Iris-setosa": 1,
    "Iris-versicolor": -1
}


def passData(fileName):
    x = []
    y = []

    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x.append(
                list(map(float, [row['A'], row['B'], row['C'], row['D']])))

            y.append(float(clasification[row['CLASS']]))

    return np.array(x), np.array(y)


if __name__ == "__main__":
    x, y = passData('iris.csv')
    epoch = 10000
    alpha = 0.01
    C = 1
    svm = SVM(x, y, epoch, alpha, C)
    svm.train()

