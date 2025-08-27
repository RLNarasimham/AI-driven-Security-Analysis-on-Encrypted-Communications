# import os
# import joblib
# import pandas as pd
# import numpy as np

# try:
#     from sklearn.preprocessing import StandardScaler
# except ImportError as e:
#     raise ImportError(
#         "scikit-learn is required for scaling functionality. "
#         "Please install it with 'pip install scikit-learn'."
#     ) from e


# def sample_for_scaler(load_in_chunks_fun, train_file_paths, numeric_cols=None,
#                       sample_per_chunk=5000, max_chunks=3, random_state=42):
#     """
#     Samples numeric data from CSV chunks to fit a scaler.
#     """
#     samples = []
#     inferred_numeric = numeric_cols

#     for file_path in train_file_paths:
#         chunk_cnt = 0
#         for chunk in load_in_chunks_fun(file_path, chunksize=200_000):
#             chunk.columns = chunk.columns.str.strip()
#             if inferred_numeric is None:
#                 inferred_numeric = chunk.select_dtypes(include=[np.number]).columns.tolist()
#                 if not inferred_numeric:
#                     raise ValueError(f"No numeric columns found in first chunk of {file_path}")

#             chunk_clean = chunk[inferred_numeric] \
#                 .replace([np.inf, -np.inf], np.nan) \
#                 .dropna()
#             if chunk_clean.empty:
#                 chunk_cnt += 1
#                 if chunk_cnt >= max_chunks:
#                     break
#                 continue

#             n = min(sample_per_chunk, len(chunk_clean))
#             sample = chunk_clean.sample(n=n, random_state=random_state)
#             samples.append(sample)

#             chunk_cnt += 1
#             if chunk_cnt >= max_chunks:
#                 break

#     if not samples:
#         raise ValueError(
#             "No valid samples found for scaler fitting. "
#             "Check that data has numeric columns and non-null values"
#         )

#     df_sample = pd.concat(samples, ignore_index=True)
#     return df_sample, inferred_numeric


# def fit_and_save_scaler(df_sample, numeric_cols, scaler_path,
#                         with_mean=True, with_std=True):
#     """
#     Fits a StandardScaler on sampled data and saves it.
#     """
#     scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
#     scaler.fit(df_sample[numeric_cols].values)

#     os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
#     joblib.dump({'scaler': scaler, 'numeric_cols': numeric_cols}, scaler_path)
#     return scaler


# def load_scaler(scaler_path):
#     """
#     Loads a saved scaler and its numeric columns list.
#     """
#     obj = joblib.load(scaler_path)
#     scaler = obj.get('scaler')
#     numeric_cols = obj.get('numeric_cols')
#     if scaler is None or numeric_cols is None:
#         raise ValueError(f"Scaler file {scaler_path} missing required keys.")
#     return scaler, numeric_cols


# def transform_chunk_with_scaler(chunk, scaler, numeric_cols):
#     """
#     Applies the scaler to numeric columns in the chunk, ensuring float dtype to avoid future errors.
#     """
#     # Clean column names
#     chunk.columns = chunk.columns.str.strip()

#     # Verify all numeric columns exist
#     missing = [col for col in numeric_cols if col not in chunk.columns]
#     if missing:
#         raise KeyError(f"Missing columns for scaling: {missing}")

#     # Copy to avoid modifying original
#     chunk_copy = chunk.copy()

#     # Cast numeric columns to float to allow assigning floats
#     chunk_copy[numeric_cols] = chunk_copy[numeric_cols].astype(float)

#     # Replace inf with NaN and drop rows with NaN
#     data = chunk_copy[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
#     if data.empty:
#         return chunk_copy

#     # Scale and rebuild DataFrame
#     scaled_vals = scaler.transform(data.values)
#     df_scaled = pd.DataFrame(scaled_vals, columns=numeric_cols, index=data.index)

#     # Assign scaled values back
#     chunk_copy.loc[data.index, numeric_cols] = df_scaled
#     return chunk_copy

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

    # Scale and save training data
    for file_path in train_file_paths:
        file_name = os.path.basename(file_path)
        print(f"Scaling training file: {file_name}")
        for i, chunk in enumerate(load_in_chunks_fun(file_path, chunksize=200_000)):
            scale_chunk(chunk, scaler, numeric_cols, output_dir, f"train_scaled_{i}_{file_name}", save_scaled=True)

    # Scale and save test data
    for file_path in test_file_paths:
        file_name = os.path.basename(file_path)
        print(f"Scaling test file: {file_name}")
        for i, chunk in enumerate(load_in_chunks_fun(file_path, chunksize=200_000)):
            scale_chunk(chunk, scaler, numeric_cols, output_dir, f"test_scaled_{i}_{file_name}", save_scaled=True)

    print(f"Scaled data saved to '{output_dir}' directory.")
    return scaler, numeric_cols