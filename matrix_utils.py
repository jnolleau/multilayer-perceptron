import numpy as np


def get_numeric_col_names(df) -> list:
    return df.select_dtypes(include=np.number).columns.tolist()


def standardize_feature(feature):
    return (feature - feature.mean())/feature.std()


def standardize_df(df):
    standardized_df = df.copy()
    num_features = get_numeric_col_names(df)
    for feature in num_features:
        standardized_df[feature] = standardize_feature(standardized_df[feature])
    return standardized_df

def standardize_df_test(df, mean, std):
    standardized_df = df.copy()
    num_features = get_numeric_col_names(df)
    for feature in num_features:
        standardized_df[feature] = (standardized_df[feature] - mean.loc[feature]) / std.loc[feature]
    return standardized_df

def shuffle(X, y):
    assert len(X) == len(y)
    random = np.arange(len(X))
    np.random.shuffle(random)
    return X[random], y[random]

def shuffle_cpy(X, y):
    assert len(X) == len(y)    
    random = np.random.permutation(len(X))
    return X[random], y[random]
