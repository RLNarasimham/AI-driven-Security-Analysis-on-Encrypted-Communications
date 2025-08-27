from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def compute_global_dropcols_and_means(load_in_chunks_fun, file_paths, drop_thresh=0.3):
    """
    One full pass over ALL files:
      - decide columns to drop by global null fraction
      - compute GLOBAL means for numeric columns
    Returns: (drop_cols, numeric_cols, means_dict)
    """
    total = None
    nulls = None
    sums = None
    counts = None
    numeric_union = set()

    for fp in file_paths:
        for ch in load_in_chunks_fun(fp, chunksize=200_000):
            c = ch.copy()
            c.columns = c.columns.str.strip()

            # global null stats
            col_nulls = c.isnull().sum()
            col_tot = pd.Series(len(c), index=c.columns)
            if nulls is None:
                nulls = col_nulls
                total = col_tot
            else:
                nulls = nulls.add(col_nulls, fill_value=0)
                total = total.add(col_tot, fill_value=0)

            # global mean stats for numeric columns
            num_cols = c.select_dtypes(include=[np.number]).columns.tolist()
            numeric_union.update(num_cols)
            num = c[num_cols].replace([np.inf, -np.inf], np.nan)

            if sums is None:
                sums = pd.Series(0.0, index=sorted(numeric_union))
                counts = pd.Series(0, index=sorted(numeric_union), dtype='int64')

            new_idx = sorted(numeric_union)
            sums   = sums.reindex(new_idx,   fill_value=0.0)
            counts = counts.reindex(new_idx, fill_value=0)

            num = num.reindex(columns=new_idx)
            sums   = sums.add(num.sum(skipna=True),            fill_value=0.0)
            counts = counts.add(num.notna().sum(),             fill_value=0)

    null_frac   = (nulls / total).fillna(0)
    drop_cols   = null_frac[null_frac > drop_thresh].index.tolist()
    numeric_cols = [c for c in sorted(numeric_union) if c not in drop_cols]

    means = (sums / counts).to_dict()
    means = {k: v for k, v in means.items() if k in numeric_cols}
    return drop_cols, numeric_cols, means


def preprocess_chunk_with_globals(chunk, drop_cols, means):
    """
    Apply GLOBAL decisions per chunk:
      - drop globally-selected columns
      - fill numeric NaNs with GLOBAL means
    """
    c = chunk.copy()
    c.columns = c.columns.str.strip()

    
    c = c.drop(columns=[col for col in drop_cols if col in c.columns], errors='ignore')

    for col in means.keys():
        if col not in c.columns:
            c[col] = np.nan

    num_cols = [col for col in c.select_dtypes(include=[np.number]).columns if col in means]
    c[num_cols] = c[num_cols].replace([np.inf, -np.inf], np.nan)
    for col in num_cols:
        c[col] = c[col].fillna(means[col])

    return c

def preprocess_chunk(chunk,drop_thresh=0.3,impute_strategy="median"):
    null_frac=chunk.isnull().mean()
    cols_to_drop=null_frac[null_frac>drop_thresh].index.tolist()
    chunk=chunk.drop(columns=cols_to_drop)
    numeric_cols=chunk.select_dtypes(include=[np.number]).columns.tolist()
    imputer=SimpleImputer(strategy=impute_strategy)
    chunk[numeric_cols]=imputer.fit_transform(chunk[numeric_cols])
    return chunk