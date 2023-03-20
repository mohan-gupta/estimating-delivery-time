echo CatBoost
python train.py --data cat --model tuned_catb
echo XGBoost
python train.py --data num --model tuned_xgb