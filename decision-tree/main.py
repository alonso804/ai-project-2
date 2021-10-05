import numpy as np
import csv

clasification = {
    "Iris-setosa": 1,
    "Iris-versicolor": -1
}


def probability(feature, total):
    return feature / total


def giniImpurity(features, total):
    g = 0
    for i in range(len(features)):
        p = probability(features[i], total)
        g += p * (1 - p)

    return g


def passData(fileName):
    root = []

    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            root.append(list(map(
                float, [row['A'], row['B'], row['C'], row['D'], clasification[row['CLASS']]])))

    return np.array(x)


if __name__ == "__main__":
    x = passData('iris.csv')
    featuresAmount = len(x[0])

    # G_iniial
    # G initial
    gInitial = 0
    total = len(x)
    for i in range(featuresAmount):
        for j in range(total):
