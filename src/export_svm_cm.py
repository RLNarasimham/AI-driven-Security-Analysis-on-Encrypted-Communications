import os, joblib
import numpy as np
import matplotlib.pyplot as plt

res = joblib.load("models/baselines_full_results.joblib")
cm = np.array(res["LinearSVC"]["cm"])  # [[tn, fp], [fn, tp]]

os.makedirs("figures", exist_ok=True)
plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix: LinearSVC")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks([0,1], ["Benign (0)","Attack (1)"])
plt.yticks([0,1], ["Benign (0)","Attack (1)"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, int(cm[i,j]), ha="center", va="center")
plt.tight_layout()
out = "figures/svm_confusion_matrix.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved -> {out}")
