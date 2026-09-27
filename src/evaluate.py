import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

model = keras.models.load_model("models/model.h5")
loss, acc = model.evaluate(x_test, y_test)

y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig("models/confusion_matrix.png")

metrics = {"test_loss": float(loss), "test_accuracy": float(acc)}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Evaluation complete:", metrics)