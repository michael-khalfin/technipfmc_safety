import pandas as pd 
from typing import Dict

TARGET_FILE = "data/cleaned_data.csv"
DESCRIPTION_COL = "text"
RECORD_NO_COL = "RECORD_NO_LOSS_POTENTIAL"
target_df = pd.read_csv(TARGET_FILE, low_memory= False)


def translateColumnsToSentences(df: pd.DataFrame, colToSentenceDict: Dict[str, str]):
    df = df.copy()
    if DESCRIPTION_COL not in df.columns:
        print(f"{DESCRIPTION_COL} not in DataFrame!")
        df[DESCRIPTION_COL] = ""


    for col, placeholder in colToSentenceDict.items():
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping...")
            continue

        df[DESCRIPTION_COL] += df[col].apply(
            lambda val: f" {placeholder.format(val)}" if pd.notna(val) else ""
        )
    
        df.drop(columns=[col], inplace=True)
    

    # Drop All Other Rows other than data col
    df = df[[RECORD_NO_COL, DESCRIPTION_COL]]

    return df


# CHANGE HERE WITH YOUR MAPPING
colToSentenceDict = {
    "TYPE": "This is a type of {}.",
}

# Just TO Check (COMMENT THIS OUT )
mask = (target_df["TYPE"] == "Hazard Observation")
rows_of_interest_idx = target_df[mask].index
print(target_df.loc[rows_of_interest_idx, DESCRIPTION_COL])


# THIS IS THE MAIN FUNCTION THAT WILL MAKE IT WORK 
embedded_df = translateColumnsToSentences(target_df, colToSentenceDict)
embedded_df.to_csv("data/cleaned_description_translated.csv", index=False) # Change name if you want 
print(embedded_df.head())


# COMMENT THIS OUT 
idx = rows_of_interest_idx[0]  
print(embedded_df.loc[idx, DESCRIPTION_COL])

