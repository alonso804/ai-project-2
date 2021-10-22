from svm import SVM
import numpy as np
from functions import passData, normalize, shuffle

if __name__ == "__main__":
    x, y = passData('gender_classification.csv')
    x[:, [1, 2]] = normalize(x[:, [1, 2]])

    epoch = 2000
    alpha = 0.001
    C = 1

    np.random.seed(0)
    shuffleX, shuffleY = shuffle(x, y)

    svm = SVM(shuffleX, shuffleY, epoch, alpha, C)
    svm.train()
