import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve,
roc_auc_score, average_precision_score, matthews_corrcoef, accuracy_score,
precision_score, recall_score, f1_score)
from scipy.stats import chi2

OUT_DIR = "results/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

X_test_top = np.load("data/processed/X_test_top.npy")
X_test_top_nn = np.load("data/processed/X_test_top_nn.npy")
y_test = np.load("data/processed/y_test_bin.npy")
attack_cat = pd.read_csv("data/processed/y_test_multi.csv")["attack_cat"]

X_test_top_r = X_test_top.reshape(-1, 30, 1)
X_test_top_nn_r = X_test_top_nn.reshape(-1, 30, 1)

rf = joblib.load("results/rf_model.pkl")
xgb = joblib.load("results/xgb_model.pkl")
svm = joblib.load("results/svm_model.pkl")
cnn = load_model("results/cnn_model.keras")
lstm = load_model("results/lstm_model.keras")
hybrid_extractor = load_model("results/hybrid_cnn_extractor.keras")
hybrid_knn = joblib.load("results/hybrid_knn_model.pkl")

hybrid_embeddings = hybrid_extractor.predict(X_test_top_r)

preds = {}
scores = {}

preds["RF"] = rf.predict(X_test_top)
scores["RF"] = rf.predict_proba(X_test_top)[:, 1]

preds["XGBoost"] = xgb.predict(X_test_top)
scores["XGBoost"] = xgb.predict_proba(X_test_top)[:, 1]

preds["SVM"] = svm.predict(X_test_top)
scores["SVM"] = svm.decision_function(X_test_top)

cnn_prob = cnn.predict(X_test_top_r).ravel()
preds["CNN"] = (cnn_prob >= 0.5).astype(int)
scores["CNN"] = cnn_prob

lstm_prob = lstm.predict(X_test_top_nn_r).ravel()
preds["LSTM"] = (lstm_prob >= 0.5).astype(int)
scores["LSTM"] = lstm_prob

preds["Hybrid_CNN_KNN"] = hybrid_knn.predict(hybrid_embeddings)
scores["Hybrid_CNN_KNN"] = hybrid_knn.predict_proba(hybrid_embeddings)[:, 1]

model_names = list(preds.keys())

metric_files = {
    "RF": "results/rf_metrics.txt",
    "XGBoost": "results/xgb_metrics.txt",
    "SVM": "results/svm_metrics.txt",
    "CNN": "results/cnn_metrics.txt",
    "LSTM": "results/lstm_metrics.txt",
    "Hybrid_CNN_KNN": "results/hybrid_cnn_knn_metrics.txt"
}

timings = {}
for name, path in metric_files.items():
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    timings[name] = d

rows = []
for name in model_names:
    y_pred = preds[name]
    y_score = scores[name]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    gmean = np.sqrt(sensitivity * specificity)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_score)
    auc_pr = average_precision_score(y_test, y_score)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall_DR": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "FAR": far,
        "MCC": mcc,
        "GMean": gmean,
        "AUC_ROC": auc_roc,
        "AUC_PR": auc_pr,
        "TrainTime_s": float(timings[name].get("train_time", "nan")),
        "TestTime_s": float(timings[name].get("test_time", "nan"))
    })

summary_df = pd.DataFrame(rows).set_index("Model")
print(summary_df.round(4))
summary_df.to_csv(f"{OUT_DIR}/summary_comparison_table.csv")

categories = sorted(attack_cat.unique())
detection_matrix = pd.DataFrame(index=model_names, columns=categories, dtype=float)

for name in model_names:
    y_pred = preds[name]
    for cat in categories:
        mask = attack_cat == cat
        if cat == "Normal":
            rate = (y_pred[mask.values] == 0).mean()
        else:
            rate = (y_pred[mask.values] == 1).mean()
        detection_matrix.loc[name, cat] = rate

detection_matrix.to_csv(f"{OUT_DIR}/category_detection_rates.csv")

plt.figure(figsize=(12, 6))
sns.heatmap(detection_matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
plt.title("Detection Rate by Attack Category")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_category_heatmap.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 7))
for name in model_names:
    fpr, tpr, _ = roc_curve(y_test, scores[name])
    auc = roc_auc_score(y_test, scores[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_roc_curves.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 7))
for name in model_names:
    prec, rec, _ = precision_recall_curve(y_test, scores[name])
    ap = average_precision_score(y_test, scores[name])
    plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - All Models")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_pr_curves.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
for i, name in enumerate(model_names):
    cm = confusion_matrix(y_test, preds[name])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False)
    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_confusion_grid.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
for name in model_names:
    plt.scatter(summary_df.loc[name, "TestTime_s"], summary_df.loc[name, "Accuracy"], s=100)
    plt.annotate(name, (summary_df.loc[name, "TestTime_s"], summary_df.loc[name, "Accuracy"]),
                 textcoords="offset points", xytext=(5, 5))
plt.xscale("log")
plt.xlabel("Test Time (s, log scale)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Inference Time")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_efficiency_scatter.png", dpi=150)
plt.close()

def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = chi2.sf(stat, 1)
    return stat, p

mcnemar_rows = []
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        a, b = model_names[i], model_names[j]
        stat, p = mcnemar_test(y_test, preds[a], preds[b])
        mcnemar_rows.append({"Model_A": a, "Model_B": b, "chi2_stat": stat, "p_value": p, "significant_p<0.05": p < 0.05})

mcnemar_df = pd.DataFrame(mcnemar_rows)
mcnemar_df.to_csv(f"{OUT_DIR}/mcnemar_pairwise.csv", index=False)
print(mcnemar_df)

print("all outputs saved to", OUT_DIR)