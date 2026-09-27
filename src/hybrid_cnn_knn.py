import numpy as np
import joblib
import time
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

X_train = np.load("data/processed/X_train_top.npy")
X_test = np.load("data/processed/X_test_top.npy")
y_train = np.load("data/processed/y_train_bin.npy")
y_test = np.load("data/processed/y_test_bin.npy")

X_train_fit, X_val, y_train_fit, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

X_train_fit_r = X_train_fit.reshape(-1, 30, 1)
X_val_r = X_val.reshape(-1, 30, 1)
X_train_r = X_train.reshape(-1, 30, 1)
X_test_r = X_test.reshape(-1, 30, 1)

extractor_input = layers.Input(shape=(30, 1))
x = layers.Conv1D(32, 3, activation="relu")(extractor_input)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(64, 3, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
embedding_layer = layers.Dense(64, activation="relu", name="embedding")(x)
x = layers.Dropout(0.3)(embedding_layer)
output = layers.Dense(1, activation="sigmoid")(x)

full_model = models.Model(extractor_input, output)
full_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

start = time.time()
full_model.fit(X_train_fit_r, y_train_fit, validation_data=(X_val_r, y_val), epochs=30, batch_size=256, callbacks=[es], verbose=1)
cnn_train_time = time.time() - start

extractor = models.Model(extractor_input, embedding_layer)

train_embeddings = extractor.predict(X_train_r)
test_embeddings = extractor.predict(X_test_r)

emb_sub, _, y_sub, _ = train_test_split(train_embeddings, y_train, train_size=20000, stratify=y_train, random_state=42)

param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}
search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring="f1", n_jobs=-1)

start = time.time()
search.fit(emb_sub, y_sub)
print("best_k", search.best_params_)

knn = KNeighborsClassifier(n_neighbors=search.best_params_["n_neighbors"], n_jobs=-1)
knn.fit(emb_sub, y_sub)
train_time = cnn_train_time + (time.time() - start)

start = time.time()
y_pred = knn.predict(test_embeddings)
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

extractor.save("results/hybrid_cnn_extractor.keras")
joblib.dump(knn, "results/hybrid_knn_model.pkl")

with open("results/hybrid_cnn_knn_metrics.txt", "w") as f:
    f.write(f"best_k,{search.best_params_}\n")
    f.write(f"accuracy,{acc}\nprecision,{prec}\nrecall,{rec}\nf1,{f1}\ntrain_time,{train_time}\ntest_time,{test_time}\n")