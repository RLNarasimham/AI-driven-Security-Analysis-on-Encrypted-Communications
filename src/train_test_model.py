import os
import joblib
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

from data_loader import load_in_chunks
from preprocess import preprocess_chunk_with_globals


ART = joblib.load("models/train_preprocess.joblib")

drop_cols    = ART["drop_cols"]
numeric_cols = ART["numeric_cols"]
means        = ART["means"]
scaler       = ART["scaler"]
_all_train   = list(ART["train_files"])
test_files   = list(ART["test_files"])
LABEL_COL    = ART.get("label_col", "Label")


if len(_all_train) < 2:
    raise RuntimeError("Need ≥2 train files to create a validation split.")
val_files   = [_all_train[-1]]
train_files = _all_train[:-1]

def binarize_labels(series):
    # BENIGN -> 0, attack/anomalous -> 1
    return (series.astype(str) != "BENIGN").astype(int).values

def gen_batches(file_list, chunksize=200_000):
    """Yield (X_scaled, y) batches. Pass DataFrame to scaler to keep feature names consistent."""
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
            for col in numeric_cols:
                if col not in X.columns:
                    X[col] = np.nan
            X = X[numeric_cols].astype(float)


            for col in numeric_cols:
                if X[col].isna().any():
                    X[col] = X[col].fillna(means.get(col, 0.0))

            X_scaled = scaler.transform(X)
            yield X_scaled, y

def compute_class_weights(train_files):
    """Estimate class weights from TRAIN to emulate 'balanced' with partial_fit."""
    counts = np.array([0, 0], dtype=np.int64) 
    for fp in train_files:
        for ch in load_in_chunks(fp, chunksize=200_000):
            if LABEL_COL not in ch.columns:
                continue
            mask = ch[LABEL_COL].notna()
            if not mask.any():
                continue
            y = binarize_labels(ch.loc[mask, LABEL_COL])
            counts[0] += (y == 0).sum()
            counts[1] += (y == 1).sum()
    total = max(counts.sum(), 1)
    
    w0 = total / (2.0 * max(counts[0], 1))
    w1 = total / (2.0 * max(counts[1], 1))
    return np.array([w0, w1], dtype=float)

def train_sgd(train_files, epochs=2):
    """Incremental training over all TRAIN batches (multiple epochs help recall)."""
    clf = SGDClassifier(loss="log_loss", random_state=42)
    class_weights = compute_class_weights(train_files)

    first = True
    for _ in range(epochs):
        for Xs, y in gen_batches(train_files):
            if y.size == 0:
                continue
            
            sw = np.where(y == 1, class_weights[1], class_weights[0]).astype(float)
            if first:
                clf.partial_fit(Xs, y, classes=np.array([0, 1]), sample_weight=sw)
                first = False
            else:
                clf.partial_fit(Xs, y, sample_weight=sw)
    return clf

def collect_scores_labels(clf, files):
    """Concatenate predict_proba scores and labels from a file list."""
    scores, labels = [], []
    for Xs, y in gen_batches(files):
        if y.size == 0:
            continue
        
        p = clf.predict_proba(Xs)[:, 1]
        scores.append(p); labels.append(y)
    if not scores:
        return np.array([]), np.array([])
    return np.concatenate(scores), np.concatenate(labels)

def best_threshold(y_true, y_score):
    """Pick threshold that maximizes F1 on validation scores."""
    if y_true.size == 0:
        return 0.5
    ts = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

def eval_with_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum() or 1
    acc  = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    pr_auc  = average_precision_score(y_true, y_score) if y_true.size else 0.0
    roc_auc = roc_auc_score(y_true, y_score)          if y_true.size else 0.0
    return cm, acc, prec, rec, f1, pr_auc, roc_auc

if __name__ == "__main__":
    print("Training SGDClassifier on TRAIN set (streaming, 2 epochs)...")
    clf = train_sgd(train_files, epochs=2)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": clf}, "models/sgd_model.joblib")
    print("Saved -> models/sgd_model.joblib")

    # Threshold tuning on small validation slice
    print("Tuning decision threshold on validation split...")
    val_scores, val_y = collect_scores_labels(clf, val_files)
    thresh = best_threshold(val_y, val_scores)
    print(f"Chosen threshold (max F1 on val): {thresh:.2f}")

    # Final evaluation on HELD-OUT TEST with tuned threshold + extra metrics
    print("Evaluating on TEST set...")
    test_scores, test_y = collect_scores_labels(clf, test_files)
    cm, acc, prec, rec, f1, pr_auc, roc_auc = eval_with_threshold(test_y, test_scores, thresh)

    print("\nConfusion Matrix [[tn, fp], [fn, tp]]:")
    print(cm)
    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # save model + tuned threshold for reuse
    joblib.dump({"model": clf, "threshold": thresh}, "models/sgd_model_with_threshold.joblib")
    print("Saved -> models/sgd_model_with_threshold.joblib")
