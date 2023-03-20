import pandas as pd

from sklearn.feature_selection import VarianceThreshold, SelectFromModel

import config
import model_dispatcher

def drop_constant_ftrs(X, y, threshold):
    var = VarianceThreshold(threshold)
    
    var.fit(X, y)
    
    features_mask = var.get_support()
    
    res = {
        "constant_ftrs": X.columns.values[~features_mask],
        "selected_ftrs": X.columns.values[features_mask]
        }
    
    return res

def from_model(X, y, model):
    estimator = model_dispatcher.model[model]
    
    model_select = SelectFromModel(estimator)
    
    model_select.fit(X, y)
    
    features_mask = model_select.get_support()
    
    res = {
        "dropped_ftrs": X.columns.values[~features_mask],
        "selected_ftrs": X.columns.values[features_mask]
        }
    
    return res

def main(df):
    X = df.drop(['Time_taken', 'kfold'], axis=1)
    y = df['Time_taken'].values
    
    print(f"Before Feature Selection, Total Features = {X.shape[1]}")
    
    threshold = 0.1
    var_ftrs = drop_constant_ftrs(X, y, threshold)
    
    print(f"Feature with variance lesser then {threshold} are: {var_ftrs['constant_ftrs']}")
    
    X = X[var_ftrs['selected_ftrs']]
    
    #Feature Selection using LGBM, CatBoost and XGBoost
    
    lgbm_ftrs = from_model(X, y, 'lgbm')
    
    print(f"Total Features dropped by LighGBM: {len(lgbm_ftrs['dropped_ftrs'])}")
    
    cat_ftrs = from_model(X, y, 'catb')
    print(cat_ftrs)
    
    print(f"Total Features dropped by CatBoost: {len(cat_ftrs['dropped_ftrs'])}")
    
    xgb_ftrs = from_model(X, y, 'xgb')
    
    print(f"Total Features dropped by XGBoost: {len(xgb_ftrs['dropped_ftrs'])}")
    
    model_selected_ftrs = list(
        set(lgbm_ftrs['selected_ftrs']).union(cat_ftrs['selected_ftrs']).union(xgb_ftrs['selected_ftrs'])
        )
    
    X_transformed = X[model_selected_ftrs]
    
    X_transformed.loc[:, 'kfold'] = df['kfold'].values
    
    new_df = pd.DataFrame(X_transformed)
    
    new_df.loc[:, 'Time_taken'] = y
    
    return new_df    

if __name__ == "__main__":
    df = pd.read_csv(config.NUM_DATA_PATH)
    #df = pd.read_csv(config.RAW_DATA_PATH)
    
    new_df = main(df)
    print(f"DataFrame shape after Feature Selection: {new_df.shape}")
    
    new_df.to_csv(config.CLN_DATA_PATH, index=False)