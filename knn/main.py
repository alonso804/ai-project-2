import numpy as np
from functions import passData, normalize, shuffle, clearFiles
from knn import KNN


if __name__ == "__main__":
    clearFiles()
    x, y = passData('gender_classification.csv')

    np.random.seed(0)
    shuffleX, shuffleY = shuffle(x, y)
    epoch = 1000
    alpha = 0.1

    e1 = KNN(shuffleX, shuffleY)
    # e1.knn(4, (1, 11.8, 6.1, 1, 0, 1, 1))
    e1.testing()
