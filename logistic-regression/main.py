import numpy as np
from functions import passData, normalize, shuffle
from regression import LogisticRegression


if __name__ == "__main__":
    x, y = passData('gender_classification.csv')

    x[:, [1, 2]] = normalize(x[:, [1, 2]])

    np.random.seed(0)
    shuffleX, shuffleY = shuffle(x, y)
    epoch = 1000
    alpha = 0.1

    e1 = LogisticRegression(shuffleX, shuffleY, epoch, alpha)
    e1.train()
