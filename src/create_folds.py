import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_folds(data):

    #Calculating number of Bins according to Sturge's Rule.
    nbins = int(1 + np.log2(len(data)))

    data['kfold'] = -1
    
    #shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    #create bins of the data
    data.loc[:, "bins"] = pd.cut(data['Time_taken'], bins=nbins, labels=False)

    kf = StratifiedKFold(n_splits=5)

    for fold, (t_, v_) in enumerate(kf.split(X=data, y=data['bins'].values)):
        data.loc[v_, "kfold"] = fold

    data = data.drop(['bins'], axis=1)

    return data

if __name__=="__main__":
    df = pd.read_csv("../dataset/train.csv")
    df_folds = create_folds(df)

    df_folds.to_csv("../dataset/data.csv", index=False)
    print("Created Folds!!")