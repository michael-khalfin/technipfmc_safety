import os
import pandas as pd
import argparse
from dataVisualizer import DataVisualizer  
from dataModifier import DataModifier
from data_clean import DataLoader

# General Path Variables 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH, CONSOLIDATED_PREFIX = "data/", "DIM_CONSOLIDATED_"
VIS_DIR = os.path.join(BASE_DIR, "visualization")

# Cache file (fast reload)
CACHE_FILE = os.path.join(DATA_PATH, "combined.csv")

def describe_data(df: pd.DataFrame):
    print("\nData Info")
    print(df.info())

    print("\nData Describe")
    print(df.describe(include='all'))

    print("\nData Head")
    print(df.head())

    print("\nData Columns")
    for col in df.columns:
        print(f"{col}, Type: {df[col].dtype}")

if __name__ == "__main__":
    # Command Line Args
    parser = argparse.ArgumentParser(description="Incident Data processing")
    parser.add_argument(
        "-D", "--descriptions",
        action="store_true",
        help="Load Data Set With Only Descriptions"
    )
    args = parser.parse_args()


    # Load Data and Describe it 
    loader = DataLoader(data_dir= DATA_PATH, verbose=True)
    combined_df = loader.load_all_data(include_actions= False)
    describe_data(combined_df)

    # Modify Data
    modifier = DataModifier(df= combined_df)
    if args.descriptions: modifier = DataModifier(df= combined_df, onlyDescriptions = True)

    modifier.set_dropped_names(["url_link", "url"])
    cleaned_df = modifier.clean()
    cleaned_df.to_csv(os.path.join("data", "cleaned_data.csv"), index=False)

    # Generate Visualizer
    viz = DataVisualizer(df=cleaned_df, vis_dir=VIS_DIR, ignored_features= ["description", "date", "email", "title", "incident_no"])

    # General Visualization of Data Types and Missing Values
    if not args.descriptions:
        viz.visualizeDataTypes()
        viz.visualizeMissingValues()
        viz.visualizeCorrelationHeatmap()
        viz.visualizeVariances(0.5)
        viz.visualizeCardinality()

    # viz.visualizeTextNgrams(column_name='TITLE__ACCIDENTS', ngram_range=(1, 1))
    # viz.visualizeTextNgrams(column_name='TITLE__ACCIDENTS', ngram_range=(2, 2))
    # viz.visualizeTextNgrams(column_name='TITLE__NEAR_MISSES', ngram_range=(1, 1))
    # viz.visualizeTextNgrams(column_name='TITLE__NEAR_MISSES', ngram_range=(2, 2))