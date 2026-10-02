# tuning notes

import yaml, json
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

import os
os.makedirs("models", exist_ok=True)

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")
x_val = np.load("data/processed/x_val.npy")
y_val = np.load("data/processed/y_val.npy")

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(params["dense_units"], activation="relu"),
    layers.Dropout(params["dropout_rate"]),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                     epochs=params["epochs"], batch_size=params["batch_size"])

model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
print("Training complete. Model and history saved.")