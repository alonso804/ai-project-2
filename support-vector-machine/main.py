from svm import SVM
from functions import passData

if __name__ == "__main__":
    x, y = passData('gender_classification.csv')

    epoch = 1000
    alpha = 0.001
    lagrage = 5
    C = 1

    svm = SVM(x, y, epoch, alpha, lagrage, C)
    svm.train()
