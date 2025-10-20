
import pandas as pd 
import re

# Map Col to Sentence
DESCRIPTION_COL = "text"
RECORD_NO_COL = "RECORD_NO_LOSS_POTENTIAL"
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


# Define Regex Rules for Dates, etc
rx = re.compile(r"(DATE|TIME)", re.IGNORECASE)

class DataModifier:

    def __init__(self, df:pd.DataFrame = None, onlyDescriptions: bool = False):
        self.df = df
        self.onlyDescriptions = onlyDescriptions
        self.date_list = [c for c in df.columns if rx.search(c)]

        # Values That Are to be set by user
        self.to_drop = set()
        if self.onlyDescriptions:
            self.to_drop = {
                "INCIDENT_TYPE", "DATE_OF_INCIDENT", "ORGANIZATION", "REPORTED_BY", 
                "PERSON_RESPONSIBLE", "PERSON_RESPONSIBLE_NAME", "SOURCE_FILE", 
                "RECORD_NO_HAZARD_OBSERVATIONS", "RECORD_NO_NEAR_MISSES"
            }


        # Just General Things We Do Before Modiying the Code 
        self.df.rename(columns={"DESCRIPTION": "text"}, inplace=True)
    
    def set_dropped_names(self, names_to_drop):
        drop = []
        names_to_drop = set(names_to_drop)
        for c in self.df.columns:
            low = c.lower()
            if any(name in low for name in names_to_drop):
                drop.append(c)
            if "email" in low:
                drop.append(c)
            if low == "name":
                drop.append(c)
        self.to_drop.update(set(drop))

    def translateColumnsToSentences(self):
        if DESCRIPTION_COL not in self.df.columns:
            print(f"{DESCRIPTION_COL} not in DataFrame!")
            self.df[DESCRIPTION_COL] = ""

        for col, placeholder in colToSentenceDict.items():
            if col not in self.df.columns:
                print(f"Warning: Column '{col}' not found, skipping...")
                continue
            
            if col == 'WORK_RELATED':
                self.df[DESCRIPTION_COL] += self.df[col].apply(
                    lambda val: " This case is work related." if val is True
                    else " This case is not work related." if val is False
                    else ""
                )

                continue
            
            if col == 'TFMC_OWNED':
                self.df[DESCRIPTION_COL] += self.df[col].apply(
                    lambda val: " The incident involves a TFMC owned operation." if val is True 
                    else " This incident involves a non-TFMC owned operation." if val is False
                    else ""
                )

                continue
            
            if col == 'SIF_PREVENTION':
                self.df[DESCRIPTION_COL] += self.df[col].apply(
                    lambda val: " This case is a Significant Incident Failure Potential Case." if val is True
                    else " This case is not a Significant Incident Failure Potential Case." if val is False
                    else ""
                )

                continue
            
            if col == 'STOPPED_WORK':
                self.df[DESCRIPTION_COL] += self.df[col].apply(
                    lambda val: " This case stopped work. " if val is True
                    else " This case did not stop work. " if val is False
                    else ""
                )

                continue 

            self.df[DESCRIPTION_COL] += self.df[col].apply(
                lambda val: f" {placeholder.format(val)}" if pd.notna(val) else ""
            )
        
            self.df.drop(columns=[col], inplace=True)
        

        # Drop All Other Rows other than data col
        self.df = self.df[[RECORD_NO_COL, DESCRIPTION_COL]]

        return self.df


    # Main Function that would hot-one encode, drop_cols, etc 
    def clean(self):
        if self.onlyDescriptions:
            # mask = (self.df["TYPE"] == "Hazard Observation")
            # rows_of_interest_idx = self.df[mask].index
            # print(self.df.loc[rows_of_interest_idx, DESCRIPTION_COL])
            description_only_df = self.translateColumnsToSentences()
            description_only_df.to_csv("data/cleaned_description_translated.csv", index=False)
            # idx = rows_of_interest_idx[0]  
            # print(description_only_df.loc[idx, DESCRIPTION_COL])

        if self.to_drop:
            print("Dropped columns:")
            for col in self.to_drop:
                print(f" - {col}")
            self.df = self.df.drop(columns=self.to_drop, errors="ignore")
        return self.df
    





    