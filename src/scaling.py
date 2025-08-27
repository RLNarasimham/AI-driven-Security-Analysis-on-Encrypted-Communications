import os
import joblib
import pandas as pd
import numpy as np

try:
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise ImportError(
        "scikit-learn is required for scaling functionality. "
        "Please install it with 'pip install scikit-learn'."
    ) from e


def fit_scaler_on_entire_dataset(load_in_chunks_fun, train_file_paths, numeric_cols=None):
    """
    Fits a StandardScaler on the entire numeric data from CSV chunks.
    """
    scaler = StandardScaler()
    inferred_numeric = numeric_cols

    for file_path in train_file_paths:
        for chunk in load_in_chunks_fun(file_path, chunksize=200_000):
            chunk.columns = chunk.columns.str.strip()
            if inferred_numeric is None:
                inferred_numeric = chunk.select_dtypes(include=[np.number]).columns.tolist()
                if not inferred_numeric:
                    raise ValueError(f"No numeric columns found in first chunk of {file_path}")

            chunk_clean = chunk[inferred_numeric].replace([np.inf, -np.inf], np.nan).dropna()
            if not chunk_clean.empty:
                scaler.partial_fit(chunk_clean)

    return scaler, inferred_numeric


def scale_chunk(chunk, scaler, numeric_cols, output_dir, file_name, save_scaled=False):
    """
    Scales a single chunk of data and optionally saves it.
    """
    chunk.columns = chunk.columns.str.strip()
    chunk_scaled = chunk.copy()
    chunk_scaled[numeric_cols] = scaler.transform(chunk[numeric_cols])

    if save_scaled:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        scaled_file_path = os.path.join(output_dir, file_name)
        chunk_scaled.to_csv(scaled_file_path, index=False)

    return chunk_scaled


def scale_and_save_scaler(load_in_chunks_fun, train_file_paths, test_file_paths,
                           output_dir='scaled_data', scaler_path='scaler.gz',
                           numeric_cols=None):
    """
    Fits a scaler on the entire training data, scales both train and test sets,
    saves the scaled data, and the scaler object.
    """
    print("Fitting scaler on the entire training dataset...")
    scaler, numeric_cols = fit_scaler_on_entire_dataset(load_in_chunks_fun, train_file_paths, numeric_cols)

    print(f"Scaler fitted. Numeric columns used for scaling: {numeric_cols}")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Scaling and saving training data
    for file_path in train_file_paths:
        file_name = os.path.basename(file_path)
        print(f"Scaling training file: {file_name}")
        for i, chunk in enumerate(load_in_chunks_fun(file_path, chunksize=200_000)):
            scale_chunk(chunk, scaler, numeric_cols, output_dir, f"train_scaled_{i}_{file_name}", save_scaled=True)

    # Scaling and saving test data
    for file_path in test_file_paths:
        file_name = os.path.basename(file_path)
        print(f"Scaling test file: {file_name}")
        for i, chunk in enumerate(load_in_chunks_fun(file_path, chunksize=200_000)):
            scale_chunk(chunk, scaler, numeric_cols, output_dir, f"test_scaled_{i}_{file_name}", save_scaled=True)

    print(f"Scaled data saved to '{output_dir}' directory.")
    return scaler, numeric_cols