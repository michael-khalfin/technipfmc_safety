import pandas as pd 
from typing import Dict

final_removed_features = {
    "INCIDENT_TYPE", "DATE_OF_INCIDENT", "ORGANIZATION", "REPORTED_BY", 
    "PERSON_RESPONSIBLE", "PERSON_RESPONSIBLE_NAME", "SOURCE_FILE", 
    "RECORD_NO_HAZARD_OBSERVATIONS", "RECORD_NO_NEAR_MISSES"
}

df = pd.read_csv("data/cleaned_data.csv", low_memory = False)

existing_removed_cols = [col for col in final_removed_features if col in df.columns]

cleaned_df = df.drop(columns = existing_removed_cols)

cleaned_df.to_csv("data/cleaned_data_without_unnecessary_features.csv", index = False)

TARGET_FILE = "data/cleaned_data_without_unnecessary_features.csv"
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
        
        if col == 'WORK_RELATED':
            df[DESCRIPTION_COL] += df[col].apply(
                lambda val: " This case is work related." if val is True
                else " This case is not work related." if val is False
                else ""
            )

            continue
        
        if col == 'TFMC_OWNED':
            df[DESCRIPTION_COL] += df[col].apply(
                lambda val: " The incident involves a TFMC owned operation." if val is True 
                else " This incident involves a non-TFMC owned operation." if val is False
                else ""
            )

            continue
        
        if col == 'SIF_PREVENTION':
            df[DESCRIPTION_COL] += df[col].apply(
                lambda val: " This case is a Significant Incident Failure Potential Case." if val is True
                else " This case is not a Significant Incident Failure Potential Case." if val is False
                else ""
            )

            continue
        
        if col == 'STOPPED_WORK':
            df[DESCRIPTION_COL] += df[col].apply(
                lambda val: " This case stopped work. " if val is True
                else " This case did not stop work. " if val is False
                else ""
            )

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
    "TYPE": "This case is a type of {}.",
    "IMPACT_TYPE": "The type of impact specified is {}.",
    "SEVERITY_DESCRIPTION": "The description of the severity selected in the Risk Matrix is {}.",
    "SEVERITY_VALUE": "The severity value for this case is {}.",
    "LIKELIHOOD_TYPE": "The description of the type of likelihood selected in the Risk Matrix is {}.",
    "LIKELIHOOD_DESCRIPTION": "This case is ({}) to occur .",
    "LIKELIHOOD_VALUE": "The value of the likelihood is {}.",
    "CRITICALITY": "The criticality of this case, which is the risk weight of the loss potential, is {}.",
    "RISK_COLOR": "The risk color for this case is {}.",
    "MITIGATED_RISK": "The value of the mitigated risk is {}.",
    "MITIGATED_RISK_COLOR": "The mitigated risk color is {}.",
    "SYSTEM_OF_RECORD": "The system of record used for this case is {}.",
    "DATE_TIME_OF_INCIDENT": "The date and the time of the case is {}.",
    "TITLE": "The title of the case is {}.",
    "GBU": "The General Business Unit involved in the case is {}.",
    "BU": "The specific Business Unit involved in the case is {}.",
    "WORKPLACE": "This case occurred at the workplace: {}.",
    "PROJECT": "The name of the project is {}.",
    "CLIENT": "The client for this case is {}.",
    "CASE_CATEGORIZATION": "The case categorization is {}.",
    "WORK_PROCESS": "The work process involved is {}.",
    "LIFE_SAVING_RULES": "The life saving rules used in this case are {}.",
    "DATE_REPORTED": "The case was reported on {}.",
    "STATUS": "The status of the case is {}.",
    "LOSS_POTENTIAL_SEVERITY": "The loss potential severity of the case was {}.",
    "DATE_OF_APPROVAL": "The date when the incident has been approved as completed is {}.",
    "DATE_OF_CLOSURE": " The date when the incident was reported as closed is {}.",
    "DUE_DATE": " The deadline for the incident to be closed is {}.",
    "OPERATING_CENTER": "The operating center corresponding to the selected incident is {}.",
    "LOCATION_CODE": "The location code of the incident is {}.",
    "PERSON_RESPONSIBLE_STATUS": "The status of the person responsible is {}."
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

