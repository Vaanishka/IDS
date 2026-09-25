import numpy as np
import joblib
import time
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

X_train = np.load("data/processed/X_train_top.npy")
X_test = np.load("data/processed/X_test_top.npy")
y_train = np.load("data/processed/y_train_bin.npy")
y_test = np.load("data/processed/y_test_bin.npy")

X_search, _, y_search, _ = train_test_split(X_train, y_train, train_size=30000, stratify=y_train, random_state=42)

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "scale_pos_weight": [1, 0.75]
}

base = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
search = RandomizedSearchCV(base, param_dist, n_iter=15, cv=3, scoring="f1", random_state=42, n_jobs=-1, verbose=1)

start = time.time()
search.fit(X_search, y_search)
print("best_params", search.best_params_)

clf = XGBClassifier(**search.best_params_, eval_metric="logloss", random_state=42, n_jobs=-1)

start = time.time()
clf.fit(X_train, y_train)
train_time = time.time() - start

start = time.time()
y_pred = clf.predict(X_test)
test_time = time.time() - start

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("accuracy", acc)
print("precision", prec)
print("recall", rec)
print("f1", f1)
print("train_time", train_time)
print("test_time", test_time)
print(cm)
print(classification_report(y_test, y_pred))

joblib.dump(clf, "results/xgb_model.pkl")

with open("results/xgb_metrics.txt", "w") as f:
    f.write(f"best_params,{search.best_params_}\n")
    f.write(f"accuracy,{acc}\nprecision,{prec}\nrecall,{rec}\nf1,{f1}\ntrain_time,{train_time}\ntest_time,{test_time}\n")