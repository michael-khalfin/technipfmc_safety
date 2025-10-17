import os
import pandas as pd
from dataVisualizer import DataVisualizer  
from dataModifier import DataModifier
from functools import reduce
from data_clean import DataLoader

# General Path Variables 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH, CONSOLIDATED_PREFIX = "data/", "DIM_CONSOLIDATED_"
VIS_DIR = os.path.join(BASE_DIR, "visualization")

# Cache file (fast reload)
CACHE_FILE = os.path.join(DATA_PATH, "combined.csv")

# These are Files that have the "NAME" Column that match to the same incident acrros files 
INDEX = "NAME"
FILE_SUFFIXES = ["INJURY_ILLNESS", "ACCIDENTS", "HAZARD_OBSERVATIONS", "PROPERTY_DAMAGE_INCIDENT", "NEAR_MISSES"]

def coerce_binary_like(s: pd.Series) -> pd.Series:
    """Try to coerce object/string binary columns into pandas nullable boolean"""
    mapping = {
        "Y": True, "N": False, "YES": True, "NO": False,
        "True": True, "False": False, "true": True, "false": False,
        "1": True, "0": False, 1: True, 0: False
    }
    s2 = s.map(lambda x: mapping.get(x, x))
    if pd.Series(s2).dropna().isin([True, False]).all():
        return s2.astype("boolean")
    return s


def prep_df(df: pd.DataFrame, suffix: str):
    # Normalize join key
    s = (df[INDEX]
         .astype("string")
         .str.normalize("NFKC")
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.lower())

    # Move NAME to index and drop empty keys
    df = df.drop(columns=[INDEX]).copy()
    df.index = s
    df = df[~df.index.isna()]

    # Report duplicates after normalization
    if not df.index.is_unique:
        counts = df.index.to_series().value_counts()
        print(f"[WARN] {suffix}: {counts[counts > 1].head(10)}")
        df = df[~df.index.duplicated(keep="last")]

    # Normalize column dtypes
    for c in df.columns:
        if df[c].dtype == "bool":
            df[c] = df[c].astype("boolean")  
        elif df[c].dtype == "object" and df[c].dropna().nunique() <= 2:
            df[c] = coerce_binary_like(df[c])

    # Tag columns by source
    df.columns = [f"{c}__{suffix}" for c in df.columns]
    return df


def load_data():
    dfs = []
    for sfx in FILE_SUFFIXES:
        f = os.path.join(DATA_PATH, f"{CONSOLIDATED_PREFIX}{sfx}.csv")
        print(f"Reading {f}")
        try:
            raw = pd.read_csv(f, low_memory=False)
            if INDEX not in raw.columns:
                print(f"[SKIP] {f}: missing {INDEX}")
                continue
            df = prep_df(raw, suffix=sfx)
            dfs.append(df)
        except Exception as e:
            print(f"[SKIP] {f} due to error: {e}")

    if not dfs:
        raise RuntimeError("No valid CSVs loaded — nothing to concatenate.")

    # Concatenate horizontally on NAME (index)
    merged = pd.concat(dfs, axis=1, join="outer")

    # Re-coerce types after merge to preserve booleans
    merged = merged.convert_dtypes(dtype_backend="numpy_nullable")

    # Bring NAME back as a column
    merged = merged.reset_index().rename(columns={"index": INDEX})

    # Save cache
    merged.to_csv(CACHE_FILE, index=False)
    return merged

def describe_data(df: pd.DataFrame):
    print("\nData Info")
    print(df.info())

    print("\nData Describe")
    print(df.describe(include='all'))

    print("\nData Head")
    print(df.head())

    # Print column names and their types explicitly
    print("\nData Columns")
    for col in df.columns:
        print(f"{col}, Type: {df[col].dtype}")

if __name__ == "__main__":
    # Load Data and Describe it 
    loader = DataLoader(data_dir= DATA_PATH, verbose=True)
    combined_df = loader.load_all_data_v1(include_actions= True)
    describe_data(combined_df)

    # Change Data
    modifier = DataModifier(df= combined_df)
    modifier.set_dropped_names(["url_link", "url"])
    cleaned_df = modifier.clean()
    cleaned_df.rename(columns={"DESCRIPTION": "text"}, inplace=True)
    cleaned_df.to_csv(os.path.join("data", "cleaned_data.csv"), index=False)

    # Generate Visualizer
    viz = DataVisualizer(df=cleaned_df, vis_dir=VIS_DIR, ignored_features= ["description", "date", "email", "title", "incident_no"])

    # General Visualization of Data Types and Missing Values
    viz.visualizeDataTypes()
    viz.visualizeMissingValues()
    viz.visualizeCorrelationHeatmap()
    viz.visualizeVariances(0.5)
    viz.visualizeCardinality()

    # viz.visualizeTextNgrams(column_name='TITLE__ACCIDENTS', ngram_range=(1, 1))
    # viz.visualizeTextNgrams(column_name='TITLE__ACCIDENTS', ngram_range=(2, 2))
    # viz.visualizeTextNgrams(column_name='TITLE__NEAR_MISSES', ngram_range=(1, 1))
    # viz.visualizeTextNgrams(column_name='TITLE__NEAR_MISSES', ngram_range=(2, 2))