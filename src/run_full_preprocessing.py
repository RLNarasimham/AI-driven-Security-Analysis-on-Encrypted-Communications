from data_loader import list_csvs, load_in_chunks
from preprocess import compute_global_dropcols_and_means, preprocess_chunk_with_globals
from scaling import fit_scaler_on_entire_dataset
import joblib, os, re

ROOT = r"../data/CICIDS2017_CSVs"

files = list_csvs(ROOT)
if not files:
    raise FileNotFoundError(f"No CSVs found under {ROOT}")

train_files = [f for f in files if "Friday" not in os.path.basename(f)]
test_files  = [f for f in files if "Friday" in  os.path.basename(f)]
if not train_files or not test_files:
    raise RuntimeError("Split failed: adjust the day filter to match your filenames.")

drop_cols, numeric_cols, means = compute_global_dropcols_and_means(
    load_in_chunks, train_files, drop_thresh=0.3
)

def preprocessed_numeric_loader(path, chunksize=200_000):
    for ch in load_in_chunks(path, chunksize=chunksize):
        ch_p = preprocess_chunk_with_globals(ch, drop_cols, means)
        yield ch_p[numeric_cols]

scaler, numeric_cols_used = fit_scaler_on_entire_dataset(
    preprocessed_numeric_loader, train_files, numeric_cols=numeric_cols
)

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
