# objective1_analysis.py
# Satisfies Objective 1: "Understand encrypted network traffic and identify metadata patterns..."
# Non-invasive: uses existing artifacts & loaders; writes reports under ./reports

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from data_loader import load_in_chunks
from preprocess import preprocess_chunk_with_globals

ART_PATH = "models/train_preprocess.joblib"                # from run_full_preprocessing.py (split-aware)
SGD_PATH = "models/sgd_model_with_threshold.joblib"        # from train_test_model.py
DT_PATH  = "models/dt_full.joblib"                         # from baselines_full_dataset.py (optional but recommended)
LSVC_PATH = "models/lsvc_full.joblib"                      # from baselines_full_dataset.py (optional)

OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------- Load artifacts/models ----------------------
art = joblib.load(ART_PATH)
drop_cols    = art["drop_cols"]
numeric_cols = art["numeric_cols"]
means        = art["means"]
scaler       = art["scaler"]
train_files  = list(art["train_files"])
LABEL_COL    = art.get("label_col", "Label")

# SGD (mandatory for this analysis)
sgd_bundle = joblib.load(SGD_PATH)
sgd_model = sgd_bundle["model"]

# Decision Tree (if available)
dt_model = joblib.load(DT_PATH) if os.path.exists(DT_PATH) else None

# LinearSVC (if available)
lsvc_model = joblib.load(LSVC_PATH) if os.path.exists(LSVC_PATH) else None

# ---------------------- Streaming batches ----------------------
def binarize_labels(series):
    # BENIGN -> 0 (normal), others -> 1 (attack/anomalous)
    return (series.astype(str) != "BENIGN").astype(int).values

def gen_batches(file_list, chunksize=200_000):
    """Yield (X_scaled_df, y_ndarray) with numeric_cols in fixed order, using global preprocess & scaler."""
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
            # residual NaN guard via global means
            for col in numeric_cols:
                if X[col].isna().any():
                    X[col] = X[col].fillna(means.get(col, 0.0))

            # Pass DataFrame to preserve feature names for scaler
            X_scaled = scaler.transform(X)  # ndarray, but names validated
            yield X_scaled, y

# ---------------------- Streaming descriptive statistics ----------------------
# We compute class-wise means/variances for each feature and overall std for point-biserial correlation.
D = len(numeric_cols)
n0 = 0
n1 = 0
mean0 = np.zeros(D, dtype=np.float64)
mean1 = np.zeros(D, dtype=np.float64)
M2_0  = np.zeros(D, dtype=np.float64)  # Welford accumulators
M2_1  = np.zeros(D, dtype=np.float64)

Nall = 0
mean_all = np.zeros(D, dtype=np.float64)
M2_all   = np.zeros(D, dtype=np.float64)

def welford_update(count, mean, M2, x):
    """Vectorized Welford update for a batch x with shape (batch, D)."""
    # Process in one go
    if x.shape[0] == 0:
        return count, mean, M2
    new_count = count + x.shape[0]
    delta = x - mean  # broadcasts
    mean += delta.sum(axis=0) / max(new_count, 1)
    delta2 = x - mean
    M2 += (delta * delta2).sum(axis=0)
    return new_count, mean, M2

print("Scanning TRAIN set to compute class-wise statistics (streaming)...")
for Xb, yb in gen_batches(train_files):
    # overall
    Nall, mean_all, M2_all = welford_update(Nall, mean_all, M2_all, Xb)

    # class 0
    mask0 = (yb == 0)
    if mask0.any():
        n0, mean0, M2_0 = welford_update(n0, mean0, M2_0, Xb[mask0])

    # class 1
    mask1 = (yb == 1)
    if mask1.any():
        n1, mean1, M2_1 = welford_update(n1, mean1, M2_1, Xb[mask1])

if Nall == 0 or n0 == 0 or n1 == 0:
    raise RuntimeError("Stats pass found no data for one of the classes. Check label column and splits.")

var0 = M2_0 / max(n0 - 1, 1)
var1 = M2_1 / max(n1 - 1, 1)
std0 = np.sqrt(np.maximum(var0, 0.0))
std1 = np.sqrt(np.maximum(var1, 0.0))

var_all = M2_all / max(Nall - 1, 1)
std_all = np.sqrt(np.maximum(var_all, 0.0))

# Effect size & point-biserial correlation
diff = mean1 - mean0
# Cohen's d with pooled std
s_pooled = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / max((n0 + n1 - 2), 1))
cohen_d = np.divide(diff, np.where(s_pooled == 0, 1.0, s_pooled))

p = n1 / (n0 + n1)
r_pb = diff * np.sqrt(p * (1 - p)) / np.where(std_all == 0, 1.0, std_all)

# ---------------------- Model-based feature importances ----------------------
rows = []
# SGD absolute weights
sgd_coef = np.abs(sgd_model.coef_[0])
sgd_coef_norm = sgd_coef / max(sgd_coef.sum(), 1e-12)

# Decision Tree importances (if available)
if dt_model is not None and hasattr(dt_model, "feature_importances_"):
    dt_imp = dt_model.feature_importances_
else:
    dt_imp = np.zeros(D, dtype=float)

# LinearSVC absolute weights (if available)
if lsvc_model is not None and hasattr(lsvc_model, "coef_"):
    lsvc_coef = np.abs(lsvc_model.coef_[0])
    lsvc_coef_norm = lsvc_coef / max(lsvc_coef.sum(), 1e-12)
else:
    lsvc_coef = np.zeros(D, dtype=float)
    lsvc_coef_norm = lsvc_coef

for i, feat in enumerate(numeric_cols):
    rows.append({
        "feature": feat,
        "mean_benign": float(mean0[i]),
        "mean_attack": float(mean1[i]),
        "diff_attack_minus_benign": float(diff[i]),
        "std_benign": float(std0[i]),
        "std_attack": float(std1[i]),
        "std_overall": float(std_all[i]),
        "cohen_d": float(cohen_d[i]),
        "point_biserial_r": float(r_pb[i]),
        "sgd_abs_weight": float(sgd_coef[i]),
        "sgd_abs_weight_norm": float(sgd_coef_norm[i]),
        "dt_importance": float(dt_imp[i]),
        "lsvc_abs_weight": float(lsvc_coef[i]),
        "lsvc_abs_weight_norm": float(lsvc_coef_norm[i]),
    })

df = pd.DataFrame(rows)

# Rank by each signal
df["rank_sgd"]  = df["sgd_abs_weight_norm"].rank(ascending=False, method="dense")
df["rank_dt"]   = df["dt_importance"].rank(ascending=False, method="dense")
df["rank_lsvc"] = df["lsvc_abs_weight_norm"].rank(ascending=False, method="dense")
df["rank_rpb"]  = df["point_biserial_r"].abs().rank(ascending=False, method="dense")
df["rank_d"]    = df["cohen_d"].abs().rank(ascending=False, method="dense")

# Composite rank (median of available ranks)
df["composite_rank"] = pd.concat(
    [df["rank_sgd"], df["rank_dt"], df["rank_lsvc"], df["rank_rpb"], df["rank_d"]],
    axis=1
).median(axis=1)

df_sorted = df.sort_values(["composite_rank", "rank_sgd"], ascending=[True, True])

# ---------------------- Write detailed CSVs ----------------------
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_all = os.path.join(OUT_DIR, f"obj1_feature_analysis_all_{stamp}.csv")
csv_top = os.path.join(OUT_DIR, f"obj1_feature_analysis_top15_{stamp}.csv")

df.to_csv(csv_all, index=False)
df_sorted.head(15).to_csv(csv_top, index=False)

print(f"Saved full feature analysis -> {csv_all}")
print(f"Saved top-15 summary -> {csv_top}")

# ---------------------- Write a short Markdown summary for the paper ----------------------
md_path = os.path.join(OUT_DIR, f"obj1_summary_{stamp}.md")
top15 = df_sorted.head(15)[
    ["feature", "point_biserial_r", "cohen_d", "sgd_abs_weight_norm", "dt_importance", "lsvc_abs_weight_norm",
     "mean_benign", "mean_attack", "diff_attack_minus_benign"]
]

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Objective 1 — Metadata Patterns Indicative of Suspicious Behaviour\n\n")
    f.write("This analysis uses **flow-level metadata only** (no decryption). We report the most informative features by "
            "combining linear weights (SGD), tree importances (DecisionTree), margin weights (LinearSVC), and "
            "distributional statistics (point-biserial correlation and Cohen’s d). All preprocessing was learned on the **train split only**.\n\n")
    f.write("## Top 15 Features (by composite rank)\n\n")
    f.write("| Feature | r_pb | d | SGD(w)_norm | DT_importance | L-SVC(w)_norm | mean(BENIGN) | mean(ATTACK) | Δ(ATTACK−BENIGN) |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in top15.iterrows():
        f.write(f"| {r['feature']} | {r['point_biserial_r']:.4f} | {r['cohen_d']:.3f} | "
                f"{r['sgd_abs_weight_norm']:.4f} | {r['dt_importance']:.4f} | "
                f"{r['lsvc_abs_weight_norm']:.4f} | {r['mean_benign']:.4g} | {r['mean_attack']:.4g} | "
                f"{r['diff_attack_minus_benign']:.4g} |\n")
    f.write("\n**Interpretation note:** Positive Δ and r_pb indicate the feature tends to be *larger* for attacks than benign flows; "
            "negative values indicate the opposite. Large |d| suggests strong separation between classes.\n")
    f.write("\n*Generated:* " + stamp + "\n")

print(f"Saved paper-ready summary -> {md_path}")
print("Objective 1 satisfied: you now have quantified patterns and top features derived from encrypted-traffic metadata.")
