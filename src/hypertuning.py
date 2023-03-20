from functools import partial

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

# from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

import optuna

import config

def optimize(trial, data):
    param_dct = dict(
        n_estimators = trial.suggest_categorical("n_estimators", [120, 300, 500, 800, 1200]),
        max_depth = trial.suggest_categorical("max_depth", [5, 8, 15, 25, 30, None]),
        max_features = trial.suggest_categorical("max_features", ["log2", "sqrt", None]),
        min_samples_split = trial.suggest_categorical("min_samples_split", [1, 2, 5, 10, 15, 100]),
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 2, 5, 10]),
        n_jobs=-1
        )
    
    model = RandomForestRegressor(**param_dct)
    
    #mse_lst = []
    r2_lst = []
    
    for fold in range(5):
        train_data = data[data.kfold!=fold]
        val_data = data[data.kfold==fold]
        
        X_train = train_data.drop(["kfold", 'Time_taken'], axis=1)
        y_train = train_data['Time_taken'].values
        
        X_val = val_data.drop(["kfold", 'Time_taken'], axis=1)
        y_val = val_data['Time_taken'].values
        
        # cat_cols = X_train.select_dtypes(include='object').columns.values
        
        # model.fit(X_train, y_train, cat_features=cat_cols)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_val)
        
        #fold_mse = mean_squared_error(y_val, pred)
        fold_r2 = r2_score(y_val, pred)
        
        #mse_lst.append(fold_mse)
        r2_lst.append(fold_r2)
    
    return -1.0*np.mean(r2_lst)#np.mean(mse_lst)

if __name__ == "__main__":
    # df = pd.read_csv(config.CLN_DATA_PATH)
    df = pd.read_csv(config.IMPUTED_DATA_PATH)
    
    optimization_fn = partial(optimize, data=df)

    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_fn, n_trials=100)

    best_params = study.best_params
    print(best_params)
    