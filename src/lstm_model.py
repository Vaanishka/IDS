import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

X_train = np.load("data/processed/X_train_top_nn.npy")
X_test = np.load("data/processed/X_test_top_nn.npy")
y_train = np.load("data/processed/y_train_bin.npy")
y_test = np.load("data/processed/y_test_bin.npy")

X_train_fit, X_val, y_train_fit, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

X_train_fit = X_train_fit.reshape(-1, 30, 1)
X_val = X_val.reshape(-1, 30, 1)
X_test_r = X_test.reshape(-1, 30, 1)

model = models.Sequential([
    layers.Input(shape=(30, 1)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

start = time.time()
model.fit(X_train_fit, y_train_fit, validation_data=(X_val, y_val), epochs=30, batch_size=256, callbacks=[es], verbose=1)
train_time = time.time() - start

start = time.time()
probs = model.predict(X_test_r).ravel()
test_time = time.time() - start
y_pred = (probs >= 0.5).astype(int)

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

model.save("results/lstm_model.keras")

with open("results/lstm_metrics.txt", "w") as f:
    f.write(f"accuracy,{acc}\nprecision,{prec}\nrecall,{rec}\nf1,{f1}\ntrain_time,{train_time}\ntest_time,{test_time}\n")