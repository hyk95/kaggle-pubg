import numpy as np


def load_data():
    x_test = np.load("../Data/Feat/All/pre_test.npy")
    x = np.load("../Data/Feat/All/pre_train.npy")
    y = np.load("../Data/Feat/All/pre_train_label.npy")
    return x, y, x_test

