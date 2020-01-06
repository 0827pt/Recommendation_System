import numpy as np
import pandas as pd
import sklearn

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(train.head())

cols = [i for i in train.columns if i not in ["id","target"]]
X_train , X_test = train[cols].values,test[cols].values
y=train["values"].target

# Feature Selection with RFE and lasso
estimator = Lasso(alpha=0.07437006553878982)
best_score=0

for n_features in [29] :
    selector = RFE(estimator,n_features_to_select=n_features,step=1)
