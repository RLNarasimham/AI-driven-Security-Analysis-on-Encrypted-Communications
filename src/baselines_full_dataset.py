# baselines_full_dataset.py
# Train DecisionTree & LinearSVC on the FULL TRAIN split (no sampling) and evaluate on FULL TEST.
# Uses your existing artifacts (train-only stats) to avoid leakage.

import os
import joblib
import numpy as np
from numpy import memmap
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_recall_fscore_support,
    average_precision_score, roc_auc_score
)

from data_loader import load_in_chunks
from preprocess import preprocess_chunk_with_globals

ART_PATH = "./models/train_preprocess.joblib"

# ---------- Load artifacts (train-only stats, scaler, file lists) ----------
ART = joblib.load(ART_PATH)
drop_cols    = ART["drop_cols"]
numeric_cols = ART["numeric_cols"]
means        = ART["means"]
scaler       = ART["scaler"]
train_files  = list(ART["train_files"])
test_files   = list(ART["test_files"])
LABEL_COL    = ART.get("label_col", "Label")

if not train_files or not test_files:
    raise RuntimeError("Artifacts missing train_files/test_files. Re-run split-aware run_full_preprocessing.py.")

N_FEATS = len(numeric_cols)
WORKDIR = "work_full"  # temp dir for memmaps
os.makedirs(WORKDIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

def binarize_labels(series):
    # BENIGN -> 0 (normal), others -> 1 (attack/anomalous)
    return (series.astype(str) != "BENIGN").astype(int).values

def gen_batches(file_list, chunksize=200_000):
    """Yield (X_scaled, y) batches using global preprocess & scaler; preserves feature names for scaler."""
    for fp in file_list:
        for ch in load_in_chunks(fp, chunksize=chunksize):
            if LABEL_COL not in ch.columns:
                continue
            ch_p = preprocess_chunk_with_globals(ch, drop_cols, means)
            mask = ch_p[LABEL_COL].notna()
            if not mask.any():
                continue

            y = binarize_labels(ch_p.loc[mask, LABEL_COL])

            X = ch_p.loc[mask].drop(columns=[LABEL_COL], errors="ignore")
            # ensure expected numeric feature set
            for col in numeric_cols:
                if col not in X.columns:
                    X[col] = np.nan
            X = X[numeric_cols].astype(float)
            for col in numeric_cols:
                if X[col].isna().any():
                    X[col] = X[col].fillna(means.get(col, 0.0))

            Xs = scaler.transform(X)  # keep DF to avoid feature-name warning; returns ndarray
            yield Xs.astype(np.float32, copy=False), y.astype(np.int8, copy=False)

# ---------- Pass 1: count full TRAIN rows ----------
def count_rows(file_list):
    total = 0
    for _, y in gen_batches(file_list):
        total += len(y)
    return total

print("Counting TRAIN rows (full pass)...")
n_train = count_rows(train_files)
if n_train == 0:
    raise RuntimeError("No labeled rows found in TRAIN files.")
print(f"Total TRAIN rows: {n_train:,}")

# ---------- Pass 2: build memmaps with FULL TRAIN data ----------
X_train_mm_path = os.path.join(WORKDIR, "train_X.dat")
y_train_mm_path = os.path.join(WORKDIR, "train_y.dat")

X_train_mm = memmap(X_train_mm_path, dtype="float32", mode="w+", shape=(n_train, N_FEATS))
y_train_mm = memmap(y_train_mm_path, dtype="int8",   mode="w+", shape=(n_train,))

write_idx = 0
for Xb, yb in gen_batches(train_files):
    n = len(yb)
    X_train_mm[write_idx:write_idx+n, :] = Xb
    y_train_mm[write_idx:write_idx+n]    = yb
    write_idx += n
del X_train_mm, y_train_mm  # flush to disk

# Reopen memmaps read-only
X_train_mm = memmap(X_train_mm_path, dtype="float32", mode="r",  shape=(n_train, N_FEATS))
y_train_mm = memmap(y_train_mm_path, dtype="int8",   mode="r",  shape=(n_train,))

# ---------- Train full-data baselines ----------
print("\nTraining DecisionTreeClassifier on FULL TRAIN...")
dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)
dt.fit(X_train_mm, y_train_mm)

print("Training LinearSVC on FULL TRAIN...")
svm = LinearSVC(class_weight="balanced", random_state=42)
svm.fit(X_train_mm, y_train_mm)

# Save models
joblib.dump(dt,  "models/dt_full.joblib")
joblib.dump(svm, "models/lsvc_full.joblib")

# ---------- Evaluate on FULL TEST (streaming) ----------
def eval_model(model, files, model_type):
    y_true_all, y_score_all, y_pred_all = [], [], []
    for Xb, yb in gen_batches(files):
        if len(yb) == 0:
            continue
        if model_type == "tree":
            # DecisionTree supports predict_proba
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(Xb)[:, 1]
            else:
                scores = model.predict(Xb)  # fallback
            preds = (scores >= 0.5).astype(int)
        else:
            # LinearSVC: use decision_function; threshold at 0
            scores = model.decision_function(Xb)
            preds  = (scores >= 0.0).astype(int)

        y_true_all.append(yb)
        y_score_all.append(scores.astype(np.float32, copy=False))
        y_pred_all.append(preds.astype(np.int8, copy=False))

    y_true  = np.concatenate(y_true_all) if y_true_all else np.zeros((0,), dtype=int)
    y_score = np.concatenate(y_score_all) if y_score_all else np.zeros((0,), dtype=float)
    y_pred  = np.concatenate(y_pred_all) if y_pred_all else np.zeros((0,), dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    pr_auc  = average_precision_score(y_true, y_score) if y_true.size else 0.0
    roc_auc = roc_auc_score(y_true, y_score)           if y_true.size else 0.0
    return cm, acc, prec, rec, f1, pr_auc, roc_auc

print("\nEvaluating DecisionTree on FULL TEST...")
dt_cm, dt_acc, dt_prec, dt_rec, dt_f1, dt_pr, dt_roc = eval_model(dt, test_files, "tree")

print("Evaluating LinearSVC on FULL TEST...")
svm_cm, svm_acc, svm_prec, svm_rec, svm_f1, svm_pr, svm_roc = eval_model(svm, test_files, "svm")

# ---------- Report ----------
def print_report(name, cm, acc, prec, rec, f1, pr_auc, roc_auc):
    print(f"\n{name}")
    print(f"Confusion Matrix [[tn, fp], [fn, tp]]:\n{cm}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n=== Full-dataset Baselines ===")
print_report("DecisionTree (full train/test)", dt_cm, dt_acc, dt_prec, dt_rec, dt_f1, dt_pr, dt_roc)
print_report("LinearSVC   (full train/test)", svm_cm, svm_acc, svm_prec, svm_rec, svm_f1, svm_pr, svm_roc)

# Save results
joblib.dump({
    "DecisionTree": {
        "cm": dt_cm, "acc": dt_acc, "prec": dt_prec, "rec": dt_rec, "f1": dt_f1, "pr_auc": dt_pr, "roc_auc": dt_roc
    },
    "LinearSVC": {
        "cm": svm_cm, "acc": svm_acc, "prec": svm_prec, "rec": svm_rec, "f1": svm_f1, "pr_auc": svm_pr, "roc_auc": svm_roc
    }
}, "models/baselines_full_results.joblib")
print("\nSaved -> models/baselines_full_results.joblib")
