from tqdm import tqdm

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import config

from warnings import filterwarnings
filterwarnings('ignore')

def check_null(df):
    '''Returns % null values in each column'''
    return (100*df.isnull().sum()/len(df)).sort_values(ascending=False)

def impute_col(df, null_col, target, imputer):
    """Performs imputation on `null_col` using the `imputer`"""
    df_train = df.loc[~df[null_col].isnull()]
    df_valid = df.loc[df[null_col].isnull()]
    
    x_train = df_train.drop(["kfold", null_col, target], axis=1)
    y_train = df_train[null_col].values
    
    x_valid = df_valid.drop(["kfold", null_col, target], axis=1)
    
    imputer.fit(x_train, y_train)
    
    preds = imputer.predict(x_valid)
    
    df_valid.loc[:, null_col] = preds
    
    imputed_df = pd.concat([df_train, df_valid], axis=0)
    
    return imputed_df

def impute(df):
    """Imputes the most empty column first.
    Imputation is done by averaging the outputs of CatBoost and XGBoost."""
    null_values = check_null(df)
    null_cols = null_values[null_values>0].index.values

    cat_imputer = CatBoostRegressor(silent=True)
    lgbm_imputer = LGBMRegressor()

    for col in tqdm(null_cols):
        cat_df = impute_col(df, col, "Time_taken", cat_imputer)
        lgbm_df = impute_col(df, col, "Time_taken", lgbm_imputer)

        cat_df.loc[:, col] = np.mean([cat_df[col], lgbm_df[col]], axis=0)
        df = cat_df

    return df

if __name__ == "__main__":
    df = pd.read_csv(config.CLN_DATA_PATH)

    df = impute(df)

    df.to_csv(config.IMPUTED_DATA_PATH, index=False)