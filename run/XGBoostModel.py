import gc
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from DataReader import load_data

param = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 0.75,
    'objective': 'binary:logistic',
    "gamma": 0,
    "reg_lambda": 0,
    "reg_alpha": 1
}


def xgboost_train(X_train, y_train, X_val, y_val, x_test, param):
    model = xgb.XGBRegressor(max_depth=param["max_depth"],
                             learning_rate=param["learning_rate"],
                             n_estimators=param["n_estimators"],
                             subsample=param["subsample"],
                             colsample_bytree=param["colsample_bytree"],
                             gamma=param["gamma"],
                             min_child_weight=param["min_child_weight"],
                             silent=False,
                             objective='reg:logistic',
                             eval_metric='mae',
                             nthread=2)
    model.fit(X_train, y_train)
    pred_val_y = model.predict(X_val)
    print(np.mean(np.abs(pred_val_y - y_val)))
    del X_train, y_train
    gc.collect()
    pred_test_y = model.predict(x_test)
    del x_test
    gc.collect()
    print('=' * 60)
    return pred_val_y, pred_test_y


if __name__ == '__main__':
    train_X, train_y, test_X = load_data()
    train_meta = np.zeros(train_y.shape[0])
    test_meta = np.zeros(test_X.shape[0])
    splits = list(KFold(n_splits=5, shuffle=True, random_state=233).split(train_X, train_y))
    for idx, (train_idx, valid_idx) in enumerate(splits):
        X_train = train_X[train_idx]
        y_train = train_y[train_idx]
        X_val = train_X[valid_idx]
        y_val = train_y[valid_idx]
        pred_val_y, pred_test_y = xgboost_train(X_train, y_train, X_val, y_val, test_X, param=param)
        train_meta[valid_idx] = pred_val_y.reshape(-1)
        test_meta += pred_test_y.reshape(-1) / len(splits)
    np.save("../mergenpy/xgmodel_train_pred.npy", train_meta)
    np.save("../mergenpy/xgmodel_test_pred.npy", test_meta)
