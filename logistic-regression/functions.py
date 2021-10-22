import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from variables import gender


def passData(fileName):
    x = []
    y = []

    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x.append(list(map(float, [row['long_hair'], row['forehead_width_cm'], row['forehead_height_cm'],
                     row['nose_wide'], row['nose_long'], row['lips_thin'], row['distance_nose_to_lip_long']])))

            y.append(float(gender[row['gender']]))

    return np.array(x), np.array(y).reshape(-1, 1)


def normalize(data):
    scaler = MinMaxScaler()
    normalizeData = scaler.fit_transform(data)

    return normalizeData


def percentage(length, fraction):
    return int(length * fraction / 100)


def shuffle(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]


def splitData(dataset, train, validation, test):
    rowsAmount = len(dataset)
    dTrain = dataset[:percentage(rowsAmount, train)]
    dValidation = dataset[percentage(rowsAmount, train):percentage(
        rowsAmount, train + validation)]
    dTest = dataset[percentage(rowsAmount, 100 - test):]

    return dTrain, dValidation, dTest


def getAccuracy(confusionMatrix):
    return (confusionMatrix[0][0] + confusionMatrix[1][1]) / (confusionMatrix[0][0] + confusionMatrix[0][1] + confusionMatrix[1][0] + confusionMatrix[1][1]) * 100
