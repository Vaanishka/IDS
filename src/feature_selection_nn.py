import numpy as np
import joblib
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train_bin.npy")
feat_cols = joblib.load("data/processed/feature_columns.pkl")

X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=50000, stratify=y_train, random_state=42)

mi_scores = mutual_info_classif(X_sub, y_sub, random_state=42)
order = np.argsort(mi_scores)[::-1]

TOP_K = 30
top_idx = order[:TOP_K]
top_features = [feat_cols[i] for i in top_idx]

for name, val in zip(top_features, mi_scores[top_idx]):
    print(name, val)

X_train_top_nn = X_train[:, top_idx]
X_test_top_nn = X_test[:, top_idx]

np.save("data/processed/X_train_top_nn.npy", X_train_top_nn)
np.save("data/processed/X_test_top_nn.npy", X_test_top_nn)
joblib.dump(top_features, "data/processed/top_features_nn_list.pkl")

print(X_train_top_nn.shape, X_test_top_nn.shape)