# testing stash

import yaml

from sklearn.model_selection import train_test_split
import numpy as np

import os
os.makedirs("data/processed", exist_ok=True)

with open("params.yaml") as f:
    params = yaml.safe_load(f)


x_train_full = np.load("data/raw/x_train.npy").reshape(-1, 784) / 255.0
y_train_full = np.load("data/raw/y_train.npy")
x_test = np.load("data/raw/x_test.npy").reshape(-1, 784) / 255.0
y_test = np.load("data/raw/y_test.npy")

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=params["preprocess"]["test_size"],
    random_state=params["preprocess"]["seed"]
)

np.save("data/processed/x_train.npy", x_train)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/x_val.npy", x_val)
np.save("data/processed/y_val.npy", y_val)
np.save("data/processed/x_test.npy", x_test)
np.save("data/processed/y_test.npy", y_test)

print("Processed train:", x_train.shape, "val:", x_val.shape, "test:", x_test.shape)