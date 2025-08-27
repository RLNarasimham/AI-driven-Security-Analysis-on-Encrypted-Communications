# # run_full_preprocessing.py
# from data_loader import list_csvs, load_in_chunks
# from preprocess import compute_global_dropcols_and_means, preprocess_chunk_with_globals
# from scaling import fit_scaler_on_entire_dataset
# import joblib, os

# ROOT = r"../data/CICIDS2017_CSVs"

# files = list_csvs(ROOT)
# if not files:
#     raise FileNotFoundError(f"No CSVs found under {ROOT}")

# # 1) Compute GLOBAL decisions/statistics from the ENTIRE dataset
# drop_cols, numeric_cols, means = compute_global_dropcols_and_means(
#     load_in_chunks, files, drop_thresh=0.3
# )

# # 2) Fit StandardScaler on the ENTIRE dataset using the preprocessed numeric columns
# def preprocessed_numeric_loader(path, chunksize=200_000):
#     for ch in load_in_chunks(path, chunksize=chunksize):
#         ch_p = preprocess_chunk_with_globals(ch, drop_cols, means)
#         yield ch_p[numeric_cols]

# scaler, numeric_cols_used = fit_scaler_on_entire_dataset(
#     preprocessed_numeric_loader, files, numeric_cols=numeric_cols
# )

# # 3) Save artifacts for reuse by your training/inference code
# os.makedirs("models", exist_ok=True)
# joblib.dump(
#     {
#         "drop_cols": drop_cols,
#         "numeric_cols": numeric_cols_used,
#         "means": means,
#         "scaler": scaler,
#     },
#     "models/full_dataset_preprocess.joblib",
# )
# print("Saved -> models/full_dataset_preprocess.joblib")

# run_full_preprocessing.py  (TRAIN-ONLY STATS)
from data_loader import list_csvs, load_in_chunks
from preprocess import compute_global_dropcols_and_means, preprocess_chunk_with_globals
from scaling import fit_scaler_on_entire_dataset
import joblib, os, re

# <<< EDIT THIS PATH >>>
ROOT = r"../data/CICIDS2017_CSVs"

files = list_csvs(ROOT)
if not files:
    raise FileNotFoundError(f"No CSVs found under {ROOT}")

# Example time-based split: use 'Friday' files as TEST, rest TRAIN (adjust if needed)
train_files = [f for f in files if "Friday" not in os.path.basename(f)]
test_files  = [f for f in files if "Friday" in  os.path.basename(f)]
if not train_files or not test_files:
    raise RuntimeError("Split failed: adjust the day filter to match your filenames.")

# 1) GLOBAL stats from TRAIN ONLY
drop_cols, numeric_cols, means = compute_global_dropcols_and_means(
    load_in_chunks, train_files, drop_thresh=0.3
)

# 2) Fit StandardScaler on TRAIN ONLY
def preprocessed_numeric_loader(path, chunksize=200_000):
    for ch in load_in_chunks(path, chunksize=chunksize):
        ch_p = preprocess_chunk_with_globals(ch, drop_cols, means)
        yield ch_p[numeric_cols]

scaler, numeric_cols_used = fit_scaler_on_entire_dataset(
    preprocessed_numeric_loader, train_files, numeric_cols=numeric_cols
)

# 3) Save artifacts (+ file lists for reproducibility)
os.makedirs("models", exist_ok=True)
joblib.dump(
    {
        "drop_cols":    drop_cols,
        "numeric_cols": numeric_cols_used,
        "means":        means,
        "scaler":       scaler,
        "train_files":  train_files,
        "test_files":   test_files,
        "label_col":    "Label",
    },
    "models/train_preprocess.joblib",
)
print("Saved -> models/train_preprocess.joblib")
