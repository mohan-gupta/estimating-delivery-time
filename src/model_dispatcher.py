from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


model = {
    'xgb': XGBRegressor(),
    'catb': CatBoostRegressor(iterations=100, silent=True),
    'lgbm': LGBMRegressor(),
    'lr': Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())]),
    'rf': RandomForestRegressor(n_jobs=-1),
    'gb': GradientBoostingRegressor(),
    'ada': AdaBoostRegressor(),
    
    'tuned_rf': RandomForestRegressor(n_estimators=800,
                                      max_depth=15,
                                      max_features=None,
                                      min_samples_split=10,
                                      min_samples_leaf=1,
                                      n_jobs=-1),

    'tuned_xgb':XGBRegressor(learning_rate=0.017184919475669573,
                             max_depth=9,
                             gamma=0.09876415786592894,
                             n_estimators=368,
                             subsample=0.9742579526052239,
                             colsample_bytree=0.9147576449193804),

    "tuned_lgbm": LGBMRegressor(learning_rate= 0.029553840670355856,
                                max_depth=12,
                                n_estimators=460,
                                subsample=0.8803213481350496,
                                colsample_bytree=0.9911620305938637,
                                reg_alpha=0.07538243558803955,
                                reg_lambda=0.08423075968309789,
                                n_jobs=-1),

    "tuned_catb": CatBoostRegressor(learning_rate=0.035579362761422054,
                                    depth=12,
                                    l2_leaf_reg=0.07473185066613205,
                                    iterations=577,
                                    subsample=0.8253687771043382,
                                    colsample_bylevel=0.09836426605496268,
                                    silent=True)
}