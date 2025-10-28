"""
Data feature analysis module for safety incident datasets.

This module provides functions to analyze data features, find common columns
across multiple CSV files, and generate dataset summaries.
"""

import os 
import pandas as pd

DATA_PATH = "data/"
CONSOLIDATED_PREFIX ="DIM_CONSOLIDATED_"

# get the common columns among all the features
# based on feature names, there are no shared features among all the consolidated files
def get_common_columns():
    """
    Find common columns across all CSV files in the data directory.
    
    Returns:
        tuple: (common_cols, file_columns) where:
            - common_cols: Set of columns present in all files
            - file_columns: Dictionary mapping file paths to their column sets
    """
    common_cols = None
    file_columns = {}

    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".csv"):
            file_name = os.path.join(DATA_PATH, file_name)
            try:
                df = pd.read_csv(file_name, low_memory = False, nrows=1)
                cols = set(df.columns.str.strip())
                file_columns[file_name] = cols
                
                if common_cols is None:
                    common_cols = cols
                else:
                    common_cols = common_cols.intersection(cols)
            except Exception as e:
                print(f"[Error] Could not read {file_path}: {e}")
    return common_cols, file_columns

# to list all the features for each excel file -> wanted to manually check the previous code
def list_all_csv_columns():
    """
    List all columns for each CSV file in the data directory.
    
    Prints column names for each file to console for manual inspection.
    """
    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".csv"):
            file_name = os.path.join(DATA_PATH, file_name)
            try:
                df = pd.read_csv(file_name, low_memory = False)
                print(f"\nColumns in {file_name}")
                for col in df.columns:
                    print(f" - {col}")
            except Exception as e:
                print(f"[Error] Could not read {file_name}: {e}")


# create table for summary of dataset including rows, columns, and data volumn
def dataset_summary():
    """
    Create a summary table of all datasets in the data directory.
    
    Returns:
        pd.DataFrame: Summary table with columns: Dataset, Rows, Columns, Volume_MB
    """
    summary = []
    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".csv"):
            file_path = os.path.join(DATA_PATH, file_name)
            try:
                df = pd.read_csv(file_path, low_memory=False)
                rows, cols = df.shape
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                summary.append({
                    "Dataset": file_name,
                    "Rows": rows,
                    "Columns": cols,
                    "Volume_MB": round(memory_mb, 2)
                })
            except Exception as e:
                print(f"[Error] Could not read {file_path}: {e}")
    return pd.DataFrame(summary)

# find the various row number features in each excel
def no_columns_summary():
    """
    Find columns containing 'NO' in their names across all CSV files.
    
    Returns:
        pd.DataFrame: Summary of files and their 'NO' columns
    """
    summary = []
    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".csv"):
            file_path = os.path.join(DATA_PATH, file_name)
            try:
                df = pd.read_csv(file_path, low_memory=False, nrows=1)  # read header only
                cols_with_no = [col for col in df.columns if 'NO' in col.upper()]
                summary.append({
                    "File": file_name,
                    "Num_Columns_with_NO": len(cols_with_no),
                    "Columns_with_NO": ", ".join(cols_with_no) if cols_with_no else ""
                })
            except Exception as e:
                print(f"[Error] Could not read {file_path}: {e}")
    return pd.DataFrame(summary)

if __name__ == "__main__":
    """
    #list_all_csv_columns()
    common_cols, file_columns = get_common_columns()
    print(common_cols)
    for col in sorted(common_cols):
        print(f" - {col}")
    
    print("\n Columns by File:")
    for file, cols in file_columns.items():
        print(f"\n {file}:")
        for col in sorted(cols):
            print(f" - {col}")

    summary_df = dataset_summary()
    print("\nDataset Summary:")
    print(summary_df.to_string(index=False))
    """

    #df1 = pd.read_csv("data/cleaned_data.csv")

    #for col in df1.columns:
        #unique_values = df1[col].dropna().unique()
        #print(f"\nColumn: {col}")
        #print(f"Unique Values (up to 10): {unique_values[:10]}")

    print(f"\n dataFeatures.py Program Running")
    print(f"\n Looking at DIM_CONSOLIDATED_HAZARD_OBSERVATIONS_translated.csv")

    df2 = pd.read_csv("data/DIM_CONSOLIDATED_HAZARD_OBSERVATIONS_translated.csv", low_memory = False)

    print(df2.columns)

    print(df2["SIF_PREVENTION"].value_counts(dropna=False))
    print(df2["STOPPED_WORK"].value_counts(dropna=False))

    print(df2.head())

    print(f"\n Looking at DIM_CONSOLIDATED_HAZARD_OBSERVATIONS")

    df3 = pd.read_csv("data/DIM_CONSOLIDATED_HAZARD_OBSERVATIONS.csv", low_memory = False)

    print(df3["SIF_PREVENTION"].value_counts(dropna=False))
    print(df3["STOPPED_WORK"].value_counts(dropna=False))

    print(df3.head())


    


