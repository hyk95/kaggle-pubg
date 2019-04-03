import pandas as pd
import numpy as np
import gc
from sklearn import preprocessing
from config.config import INPUT_DIR

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))
    return df


def feature_engineering(input_dir="Data/", is_train=True):
    if is_train:
        print("processing train_V2.csv")
        df = pd.read_csv(input_dir + 'train_V2.csv')
        df = df[df['maxPlace'] > 1]
    else:
        print("processing test_V2.csv")
        df = pd.read_csv(input_dir + 'test_V2.csv')
    df = reduce_mem_usage(df)
    gc.collect()
    df['totalDistance'] = (
            df['rideDistance'] +
            df["walkDistance"] +
            df["swimDistance"]
    )
    df['headshotrate'] = df['kills'] / df['headshotKills']
    df['killStreakrate'] = df['killStreaks'] / df['kills']
    df['healthitems'] = df['heals'] + df['boosts']
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df["headshotKills"] + df["roadKills"]

    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    df.fillna(0, inplace=True)

    df.loc[df.rankPoints < 0, 'rankPoints'] = 0

    target = 'winPlacePerc'
    features = df.columns.tolist()

    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchDuration")
    features.remove("matchType")
    features.remove("maxPlace")

    y = None
    if is_train:
        df_out = df.groupby(['matchId', 'groupId'])[
            ["maxPlace", "matchDuration"]].first().reset_index()
        y = df.groupby(['matchId', 'groupId'])[
            target].first().values
        features.remove(target)
    else:
        df_out = df[['matchId', 'groupId', "maxPlace", "matchDuration"]]

    df = df[features + ["matchId", "groupId"]].copy()
    gc.collect()
    print("Mean features:")
    agg = df.groupby(['matchId', 'groupId'])[
        features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(
        pct=True).reset_index()
    agg_mean = agg.reset_index().groupby(
        'matchId')[features].mean()
    agg_mean.columns = [x + "_mean_mean" for x in agg_mean.columns]
    print("Merging (mean):")
    df_out = df_out.merge(
        agg.reset_index(), how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(
        agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(
        agg_mean.reset_index(), how='left', on=['matchId'])
    df_out = reduce_mem_usage(df_out)
    print("Max features:")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(
        pct=True).reset_index()
    agg_mean = agg.groupby('matchId')[features].mean()
    agg_mean.columns = [x + "_max_mean" for x in agg_mean.columns]
    print("Merging (max):")
    df_out = df_out.merge(
        agg.reset_index(), how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=[
        "_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(
        agg_mean.reset_index(), how='left', on=['matchId'])
    df_out = reduce_mem_usage(df_out)
    print("Min features:")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(
        pct=True).reset_index()
    print("Merging (min):")
    df_out = df_out.merge(agg.reset_index(), how='left', on=[
        'matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=[
        "_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    df_out = reduce_mem_usage(df_out)

    # print("Sum features:")
    # # Make new features indicating the minimum value of the features for each group ( grouped by match )
    # agg = df.groupby(['matchId', 'groupId'])[features].agg('sum')
    # # Put the new features into a rank form ( max value will have the highest rank)
    # agg_rank = agg.groupby('matchId')[features].rank(
    #     pct=True).reset_index()
    #
    # print("Merging (sum):")
    # # Merge the new (agg and agg_rank) with df_out :
    # df_out = df_out.merge(agg.reset_index(), how='left', on=[
    #     'matchId', 'groupId'])
    # df_out = df_out.merge(agg_rank, suffixes=[
    #     "_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])
    # df_out = reduce_mem_usage(df_out)

    print("Group size:")
    agg = df.groupby(['matchId', 'groupId']).size(
    ).reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    print("Match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(
        agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    df_out = reduce_mem_usage(df_out)
    print("Match median feature")
    agg = df.groupby(['matchId'])[features].agg('median').reset_index()
    df_out = df_out.merge(
        agg, suffixes=["", "_match_median"], how='left', on=['matchId'])
    df_out = reduce_mem_usage(df_out)
    print("Match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])

    df_out[df_out == np.Inf] = np.NaN
    df_out[df_out == np.NINF] = np.NaN
    df_out.fillna(0, inplace=True)
    df_out = reduce_mem_usage(df_out)
    gc.collect()
    if is_train:
        df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
        X = np.array(df_out, dtype=np.float32)
        del df, df_out, agg, agg_rank
        return X, y
    else:
        df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
        X = np.array(df_out, dtype=np.float32)
        del df, df_out, agg, agg_rank
        return X, None


if __name__ == '__main__':
    x_train, y_train = feature_engineering(INPUT_DIR)
    x_test, _ = feature_engineering(INPUT_DIR, is_train=False)
    scaler = preprocessing.MinMaxScaler(
        feature_range=(-1, 1), copy=False).fit(x_train)
    print("x_train", x_train.shape, x_train.max(), x_train.min())
    scaler.transform(x_train).astype("float32")
    print("x_train", x_train.shape, x_train.max(), x_train.min())
    print("x_test", x_test.shape, x_test.max(), x_test.min())
    scaler.transform(x_test).astype("float32")
    print("x_test", x_test.shape, x_test.max(), x_test.min())
    np.save("x_train_v2.npy", x_train)
    np.save("y_train_v2.npy", y_train)
    np.save("x_test_v2.npy", x_test)