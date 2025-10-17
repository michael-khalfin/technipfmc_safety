import pandas as pd 
from typing import Dict

TARGET_FILE = 'data/combined.csv'
CANDIDATE_FILE = "data/cleaned_data.csv"
DESCRIPTION_COL = "DESCRIPTION"
# DESCRIPTION_COL = "text"

target_df = pd.read_csv(TARGET_FILE, low_memory= False)
candidate_df = pd.read_csv(CANDIDATE_FILE, low_memory= False)


def translateColumnsToSentences(df: pd.DataFrame, colToSentenceDict: Dict[str, str]):
    if DESCRIPTION_COL not in df.columns:
        df[DESCRIPTION_COL] = ""


    for col, placeholder in colToSentenceDict.items():
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping...")
            continue

        df[DESCRIPTION_COL] += df[col].apply(
            lambda val: f" {placeholder.format(val)}" if pd.notna(val) else ""
        )
    
        df.drop(columns=[col], inplace=True)

    return df





# colToSentenceDict = {
#     "AGE": "The individual is {} years old.",
#     "CITY": "They live in {}.",
#     "STATUS": "Their current status is {}."
# }

# target_df = translateColumnsToSentences(target_df, colToSentenceDict)
# candidate_df = translateColumnsToSentences(candidate_df, colToSentenceDict)


# target_df.to_csv("data/combined_translated.csv", index=False)
# candidate_df.to_csv("data/cleaned_translated.csv", index=False)

# print(target_df.head())