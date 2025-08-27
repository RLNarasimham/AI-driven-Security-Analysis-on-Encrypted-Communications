import os
import argparse
import joblib
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_recall_fscore_support,
    accuracy_score, average_precision_score, roc_auc_score
)

from data_loader import load_in_chunks
from preprocess import preprocess_chunk_with_globals


def binarize_labels(series, benign_tag="BENIGN"):
    return (series.astype(str) != benign_tag).astype(int).values

def gen_batches(file_list, label_col, drop_cols, means, scaler, numeric_cols, chunksize):
    """Yield (X_scaled, y) batches using global preprocess + scaler with consistent feature names."""
    for fp in file_list:
        for ch in load_in_chunks(fp, chunksize=chunksize):
            if label_col not in ch.columns:
                continue
            ch_p = preprocess_chunk_with_globals(ch, drop_cols, means)
            mask = ch_p[label_col].notna()
            if not mask.any():
                continue

            y = binarize_labels(ch_p.loc[mask, label_col])

            X = ch_p.loc[mask].drop(columns=[label_col], errors="ignore")
            for col in numeric_cols:
                if col not in X.columns:
                    X[col] = np.nan
            X = X[numeric_cols].astype(float)

            # residual NaN guard
            for col in numeric_cols:
                if X[col].isna().any():
                    X[col] = X[col].fillna(means.get(col, 0.0))

            # pass DataFrame to keep feature names consistent
            X_scaled = scaler.transform(X)
            yield X_scaled, y

def collect_stratified_sample(file_list, label_col, drop_cols, means, scaler, numeric_cols,
                              chunksize=200_000, per_class_limit=50000):
    """
    Stream over files and collect up to per_class_limit samples for each class (0 and 1).
    Returns X_sample, y_sample (numpy arrays).
    """
    want0 = int(per_class_limit)
    want1 = int(per_class_limit)
    have0 = have1 = 0

    X_parts_0, X_parts_1 = [], []
    y_parts_0, y_parts_1 = [], []

    for Xb, yb in gen_batches(file_list, label_col, drop_cols, means, scaler, numeric_cols, chunksize):
        if have0 >= want0 and have1 >= want1:
            break

        # indices per class in this batch
        idx0 = np.where(yb == 0)[0]
        idx1 = np.where(yb == 1)[0]

        need0 = max(want0 - have0, 0)
        need1 = max(want1 - have1, 0)

        if need0 > 0 and idx0.size > 0:
            take0 = idx0[:min(need0, idx0.size)]
            X_parts_0.append(Xb[take0])
            y_parts_0.append(yb[take0])
            have0 += take0.size

        if need1 > 0 and idx1.size > 0:
            take1 = idx1[:min(need1, idx1.size)]
            X_parts_1.append(Xb[take1])
            y_parts_1.append(yb[take1])
            have1 += take1.size

    X0 = np.vstack(X_parts_0) if X_parts_0 else np.zeros((0, len(numeric_cols)))
    y0 = np.concatenate(y_parts_0) if y_parts_0 else np.zeros((0,), dtype=int)
    X1 = np.vstack(X_parts_1) if X_parts_1 else np.zeros((0, len(numeric_cols)))
    y1 = np.concatenate(y_parts_1) if y_parts_1 else np.zeros((0,), dtype=int)

    X = np.vstack([X0, X1]) if X0.size or X1.size else np.zeros((0, len(numeric_cols)))
    y = np.concatenate([y0, y1]) if y0.size or y1.size else np.zeros((0,), dtype=int)

    # shuffle (to mix classes)
    if y.size:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(y))
        X, y = X[idx], y[idx]

    return X, y

def eval_with_scores(y_true, y_scores, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average='binary', zero_division=0)
    pr_auc  = average_precision_score(y_true, y_scores) if y_scores.size else 0.0
    roc_auc = roc_auc_score(y_true, y_scores)           if y_scores.size else 0.0
    return cm, acc, prec, rec, f1, pr_auc, roc_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="models/train_preprocess.joblib",
                    help="Path to preprocessing artifacts saved from train-only split")
    ap.add_argument("--train_per_class", type=int, default=60000,
                    help="Max samples per class from TRAIN for fitting baselines")
    ap.add_argument("--test_per_class", type=int, default=60000,
                    help="Max samples per class from TEST for evaluation")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    ART = joblib.load(args.art)
    drop_cols    = ART["drop_cols"]
    numeric_cols = ART["numeric_cols"]
    means        = ART["means"]
    scaler       = ART["scaler"]
    train_files  = list(ART["train_files"])
    test_files   = list(ART["test_files"])
    LABEL_COL    = ART.get("label_col", "Label")

    if not train_files or not test_files:
        raise RuntimeError("Artifacts missing train_files/test_files. Re-run run_full_preprocessing.py (split-aware).")

    print("Collecting TRAIN sample for baselines...")
    Xtr, ytr = collect_stratified_sample(
        train_files, LABEL_COL, drop_cols, means, scaler, numeric_cols,
        chunksize=args.chunksize, per_class_limit=args.train_per_class
    )
    print(f"TRAIN sample: X={Xtr.shape}, y counts -> 0:{(ytr==0).sum()} 1:{(ytr==1).sum()}")

    print("Collecting TEST sample for baselines...")
    Xte, yte = collect_stratified_sample(
        test_files, LABEL_COL, drop_cols, means, scaler, numeric_cols,
        chunksize=args.chunksize, per_class_limit=args.test_per_class
    )
    print(f"TEST sample:  X={Xte.shape}, y counts -> 0:{(yte==0).sum()} 1:{(yte==1).sum()}")

    if ytr.size == 0 or yte.size == 0:
        raise RuntimeError("Sampling produced empty datasets. Increase per_class limits or check labels.")

    results = {}

    # ----- Decision Tree -----
    print("\nTraining DecisionTreeClassifier...")
    dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    dt.fit(Xtr, ytr)

    # Scores for metrics
    if hasattr(dt, "predict_proba"):
        dt_scores = dt.predict_proba(Xte)[:, 1]
    else:
        # Fallback: use predictions as scores (rare for tree)
        dt_scores = dt.predict(Xte)

    dt_pred = (dt_scores >= 0.5).astype(int)  # default 0.5 threshold
    cm, acc, prec, rec, f1, pr_auc, roc_auc = eval_with_scores(yte, dt_scores, dt_pred)
    results["DecisionTree"] = {
        "cm": cm.tolist(), "acc": acc, "prec": prec, "rec": rec,
        "f1": f1, "pr_auc": pr_auc, "roc_auc": roc_auc
    }

    # ----- Linear SVM -----
    print("Training LinearSVC...")
    svm = LinearSVC(class_weight="balanced", random_state=42)
    svm.fit(Xtr, ytr)

    # decision_function as score; threshold at 0
    svm_scores = svm.decision_function(Xte)
    svm_pred   = (svm_scores >= 0.0).astype(int)
    cm, acc, prec, rec, f1, pr_auc, roc_auc = eval_with_scores(yte, svm_scores, svm_pred)
    results["LinearSVC"] = {
        "cm": cm.tolist(), "acc": acc, "prec": prec, "rec": rec,
        "f1": f1, "pr_auc": pr_auc, "roc_auc": roc_auc
    }

    # ----- Report -----
    print("\n=== Baseline Comparison (sampled) ===")
    for name, r in results.items():
        print(f"\n{name}")
        print(f"Confusion Matrix [[tn, fp], [fn, tp]]: {r['cm']}")
        print(f"Accuracy : {r['acc']:.4f}")
        print(f"Precision: {r['prec']:.4f}")
        print(f"Recall   : {r['rec']:.4f}")
        print(f"F1-score : {r['f1']:.4f}")
        print(f"PR-AUC   : {r['pr_auc']:.4f}")
        print(f"ROC-AUC  : {r['roc_auc']:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(results, "models/baselines_results.joblib")
    print("\nSaved -> models/baselines_results.joblib")

if __name__ == "__main__":
    main()
