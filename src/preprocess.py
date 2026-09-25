import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

train = pd.read_csv("data/UNSW_NB15_training-set.csv")
test = pd.read_csv("data/UNSW_NB15_testing-set.csv")

train = train.drop("id", axis=1)
test = test.drop("id", axis=1)

y_train_multi = train["attack_cat"]
y_test_multi = test["attack_cat"]
y_train_bin = train["label"]
y_test_bin = test["label"]

train_feat = train.drop(["attack_cat", "label"], axis=1)
test_feat = test.drop(["attack_cat", "label"], axis=1)

cat_cols = ["proto", "service", "state"]
train_enc = pd.get_dummies(train_feat, columns=cat_cols)
test_enc = pd.get_dummies(test_feat, columns=cat_cols)
test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(train_enc)
X_test = scaler.transform(test_enc)

os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train_bin.npy", y_train_bin.values)
np.save("data/processed/y_test_bin.npy", y_test_bin.values)
y_train_multi.to_csv("data/processed/y_train_multi.csv", index=False)
y_test_multi.to_csv("data/processed/y_test_multi.csv", index=False)
joblib.dump(scaler, "data/processed/scaler.pkl")
joblib.dump(train_enc.columns.tolist(), "data/processed/feature_columns.pkl")

print(X_train.shape, X_test.shape)