import glob
import os
import pandas as pd
from dataVisualizer import DataVisualizer  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "data/"
CONSOLIDATED_PREFIX = "DIM_CONSOLIDATED_*"
VIS_DIR = os.path.join(BASE_DIR, "visualization")

def load_data():
    files = glob.glob(os.path.join(DATA_PATH, CONSOLIDATED_PREFIX))
    if not files:
        print("No files matched the pattern. Check DATA_PATH and CONSOLIDATED_PREFIX.")
    dfs = []
    #count =0 
    for f in files:
        #if count < 3:
            try:
                df = pd.read_csv(f) 
                dfs.append(df)
                #count +=1 
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")
    if not dfs:
        raise RuntimeError("No valid CSVs loaded — nothing to concatenate.")
    return pd.concat(dfs, ignore_index=True)

def describe_data(df: pd.DataFrame):
    print("\nData Info")
    print(df.info())

    print("\nData Describe")
    print(df.describe(include='all'))

    print("\nData Head")
    print(df.head())

if __name__ == "__main__":
    combined_df = load_data()
    describe_data(combined_df)

    # Visualize the Data 
    viz = DataVisualizer(df=combined_df, vis_dir=VIS_DIR)
    viz.visualizeDataTypes()
    viz.visualizeMissingValues()
