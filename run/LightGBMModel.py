import lightgbm as lgb
import gc
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from DataReader import load_data


def model(train_data, val_data, test_X):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        'metric': {'mean_absolute_error'},  # 评估函数
        'num_leaves': 63,  # 叶子节点数
        'max_depth': -1,
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.88,  # 建树的特征选择比例
        'bagging_fraction': 0.9,  # 建树的样本采样比例
        'bagging_freq': 15,  # k 意味着每 k 次迭代执行bagging
        'device': 'gpu',
        'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    num_round = 5000
    raw_val_x = val_data.data
    bst = lgb.train(params, train_data, num_round, valid_sets=[train_data, val_data], early_stopping_rounds=200, verbose_eval=500)
    print("start eval")
    pred_val_y = bst.predict(raw_val_x)
    print(np.mean(np.abs(pred_val_y - val_data.get_label())))
    del train_data, val_data, raw_val_x
    gc.collect()
    pred_test_y = bst.predict(test_X)
    print('=' * 60)
    del bst
    gc.collect()
    return pred_val_y, pred_test_y


if __name__ == '__main__':
    train_X, train_y, test_X = load_data()
    feature = train_X.shape[1]
    train_meta = np.zeros(train_y.shape[0])
    test_meta = np.zeros(test_X.shape[0])
    splits = list(KFold(n_splits=5, shuffle=True, random_state=233).split(train_X, train_y))
    for idx, (train_idx, valid_idx) in enumerate(splits):
        train_data = lgb.Dataset(train_X[train_idx], label=train_y[train_idx])
        val_data = lgb.Dataset(train_X[valid_idx], label=train_y[valid_idx])
        pred_val_y, pred_test_y = model(train_data, val_data, test_X)
        del train_data, val_data
        gc.collect()
        train_meta[valid_idx] = pred_val_y.reshape(-1)
        test_meta += pred_test_y.reshape(-1) / len(splits)
    np.save("../mergenpy/lightmodel_train_pred.npy", train_meta)
    np.save("../mergenpy/lightmodel_test_pred.npy", test_meta)
