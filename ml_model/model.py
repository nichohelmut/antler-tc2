import json
import pickle

import gcsfs
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from helper import simple_hot_encoding

df = pd.read_csv('dataset/thelook2.csv')

df, _, _, _ = simple_hot_encoding(df)

Y = df['Prize']  # dependent variable
X = df[['Item', 'Brand', 'Usage', 'Gender', 'Model', 'Size', 'Sale_Until']]  # independent variable

X_num = X.drop(['Item', 'Model'], axis=1)
X_num = X_num[X_num.columns].apply(pd.to_numeric, errors='coerce')
print(X_num)
X_num2 = np.array(X_num)
y_num = np.array(Y)

kf = KFold(n_splits=3, random_state=0, shuffle=True)
kf.get_n_splits(X_num2)

for train_index, test_index in kf.split(X_num2):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_num2[train_index], X_num2[test_index]
    y_train, y_test = y_num[train_index], y_num[test_index]

model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

model.fit(X_train, y_train, verbose=True)

fs = gcsfs.GCSFileSystem(project='footy-dev-343115', token="footy-dev-343115-af83a33838e0.json")
with fs.open('gs://pricing_model_nhu/model2.pkl', 'wb') as handle:
    pickle.dump(model, handle)  # save the model

y_test_pred = model.predict(X_test)

mse = mean_squared_error(y_test_pred, y_test, squared=False)

metadata = {
    'score': model.score(X_num, y_num),
    'rooted main error': mse
}

with fs.open('gs://pricing_model_nhu/meta.json', 'w') as handle:
    json.dump(metadata, handle)  # save the model

# TODO: CHECK SCORE/RMSE
print(model.predict([[147, 1, 0, 42, 49]]))  # format of input
print(f'score: {model.score(X_num, y_num)}')
print(f"rooted main error: {mse}")
