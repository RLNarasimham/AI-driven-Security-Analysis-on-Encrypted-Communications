import pandas as pd
import os, glob

def list_csvs(root_dir):
    """
    Recursively list all CSV files under a directory.
    """
    pattern = os.path.join(root_dir, '**', '*.csv')
    return sorted(glob.glob(pattern, recursive=True))

def load_in_chunks(path,chunksize=200_000):
    for chunk in pd.read_csv(path,chunksize=chunksize):
        chunk.columns=chunk.columns.str.strip()
        yield chunk