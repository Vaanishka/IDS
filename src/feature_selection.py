import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train_bin.npy")
feat_cols = joblib.load("data/processed/feature_columns.pkl")

ranker = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
ranker.fit(X_train, y_train)

importances = ranker.feature_importances_
order = np.argsort(importances)[::-1]
TOP_K = 30
top_idx = order[:TOP_K]
top_features = [feat_cols[i] for i in top_idx]

for name, val in zip(top_features, importances[top_idx]):
    print(name, val)

X_train_top = X_train[:, top_idx]
X_test_top = X_test[:, top_idx]

np.save("data/processed/X_train_top.npy", X_train_top)
np.save("data/processed/X_test_top.npy", X_test_top)
joblib.dump(top_features, "data/processed/top_features_list.pkl")

print(X_train_top.shape, X_test_top.shape)