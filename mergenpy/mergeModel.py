import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from mergenpy.adjustData import post_processing


def mergerModel():
    y = np.load("../Data/Feat/All/pre_train_label.npy")
    y = np.expand_dims(y, axis=-1)

    pred_train1 = np.load("lightmodel_train_pred.npy")
    pred_train2 = np.load("xgmodel_train_pred.npy")
    pred_train3 = np.load("nnmodel_train_pred.npy")
    pred_train1 = np.expand_dims(pred_train1, axis=-1)
    pred_train2 = np.expand_dims(pred_train2, axis=-1)
    pred_train3 = np.expand_dims(pred_train3, axis=-1)

    pred_test1 = np.load("lightmodel_test_pred.npy")
    pred_test2 = np.load("xgmodel_test_pred.npy")
    pred_test3 = np.load("nnmodel_test_pred.npy")
    pred_test1 = np.expand_dims(pred_test1, axis=-1)
    pred_test2 = np.expand_dims(pred_test2, axis=-1)
    pred_test3 = np.expand_dims(pred_test3, axis=-1)

    x_train = np.concatenate([pred_train1, pred_train2, pred_train3], axis=-1)
    x_test = np.concatenate([pred_test1, pred_test2, pred_test3], axis=-1)
    classifier = LinearRegression()
    classifier.fit(x_train, y)
    pred = classifier.predict(x_test)
    print(classifier.coef_)
    post_processing(pred)

