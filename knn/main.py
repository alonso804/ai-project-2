import numpy as np
from functions import passData, normalize, shuffle, clearFiles
from knn import KNN
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    clearFiles()
    x, y = passData('gender_classification.csv')

    np.random.seed(0)
    shuffleX, shuffleY = shuffle(x, y)

    e1 = KNN(shuffleX, shuffleY)

    for i in range(1, 41):
        e1.kFoldCrossValidation(10, i)
