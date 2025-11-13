
"""
Data modification module for safety incident data processing.

This module provides functionality to transform safety incident data,
convert columns to descriptive sentences, and clean datasets for analysis.
When `onlyDescriptions=True`, the output `text` column is now a structured
block that combines the narrative with bullet-pointed metadata so downstream
pipelines (GraphRAG, KG builders, etc.) no longer require an extra
post-processing step.
"""

import csv
import pandas as pd
import re

# Column constants
DESCRIPTION_COL = "text"
RECORD_NO_COL = "RECORD_NO_LOSS_POTENTIAL"
FACT_LIMIT = 18

# Mapping from consolidated column names to canonical fact labels
COLUMN_FACT_LABELS = {
    "TYPE": "INCIDENT_TYPE",
    "IMPACT_TYPE": "IMPACT_TYPE",
    "SEVERITY_DESCRIPTION": "SEVERITY_DESC",
    "SEVERITY_VALUE": "SEVERITY_SCORE",
    "LIKELIHOOD_TYPE": "LIKELIHOOD_DESC",
    "LIKELIHOOD_DESCRIPTION": "LIKELIHOOD_RANGE",
    "LIKELIHOOD_VALUE": "LIKELIHOOD_SCORE",
    "CRITICALITY": "CRITICALITY",
    "RISK_COLOR": "RISK_COLOR",
    "MITIGATED_RISK": "MITIGATED_RISK",
    "MITIGATED_RISK_COLOR": "MITIGATED_RISK_COLOR",
    "SYSTEM_OF_RECORD": "SOURCE_SYSTEM",
    "DATE_TIME_OF_INCIDENT": "EVENT_DATETIME",
    "TITLE": "TITLE",
    "GBU": "GENERAL_BUSINESS_UNIT",
    "BU": "SPECIFIC_BUSINESS_UNIT",
    "WORKPLACE": "WORKPLACE",
    "PROJECT": "PROJECT",
    "CLIENT": "CLIENT",
    "CASE_CATEGORIZATION": "CASE_CATEGORIZATION",
    "WORK_PROCESS": "WORK_PROCESS",
    "LIFE_SAVING_RULES": "LIFE_SAVING_RULES",
    "DATE_REPORTED": "REPORTED_DATE",
    "STATUS": "STATUS",
    "LOSS_POTENTIAL_SEVERITY": "LOSS_POTENTIAL",
    "DATE_OF_APPROVAL": "APPROVED_DATE",
    "DATE_OF_CLOSURE": "CLOSED_DATE",
    "DUE_DATE": "DEADLINE",
    "OPERATING_CENTER": "OPERATING_CENTER",
    "LOCATION_CODE": "LOCATION_CODE",
    "PERSON_RESPONSIBLE_STATUS": "PERSON_RESPONSIBLE_STATUS",
}

# Boolean columns need bespoke phrasing
BOOLEAN_FACT_LABELS = {
    "WORK_RELATED": ("WORK_RELATIONSHIP", "Work related case", "Not work related case"),
    "TFMC_OWNED": ("TFMC_OWNERSHIP", "TFMC owned operation", "Non-TFMC owned operation"),
    "SIF_PREVENTION": ("SIF_PREVENTION", "Significant Incident Failure potential", "Not a Significant Incident Failure potential"),
    "STOPPED_WORK": ("WORK_INTERRUPTION", "Work stopped due to event", "Work continued during event"),
}

# Define Regex Rules for Dates, etc
rx = re.compile(r"(DATE|TIME)", re.IGNORECASE)

class DataModifier:
    """
    Data modification class for safety incident data processing.
    
    This class provides methods to transform safety incident data by:
    - Converting structured columns to descriptive sentences
    - Cleaning and filtering data
    - Preparing data for analysis
    """

    def __init__(self, df:pd.DataFrame = None, onlyDescriptions: bool = False):
        """
        Initialize DataModifier with DataFrame and processing options.
        
        Args:
            df (pd.DataFrame): Input DataFrame to modify
            onlyDescriptions (bool): Whether to process only description-related columns
        """
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
        """
        Add column names to the drop list based on name patterns.
        
        Args:
            names_to_drop (list): List of name patterns to match against column names
        """
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
        """
        Convert structured columns to an interpretable text block.
        
        Each record becomes:
            INCIDENT_ID: <id>
            NARRATIVE:
            <original description>
            
            FACTS:
            - LABEL: value
        
        Returns:
            pd.DataFrame: Two-column DataFrame (record id + structured text)
        """
        if DESCRIPTION_COL not in self.df.columns:
            print(f"{DESCRIPTION_COL} not in DataFrame!")
            self.df[DESCRIPTION_COL] = ""
        else:
            self.df[DESCRIPTION_COL] = self.df[DESCRIPTION_COL].fillna("").astype(str)

        def _format_value(val):
            if pd.isna(val):
                return None
            if isinstance(val, str):
                stripped = val.strip()
            else:
                stripped = str(val).strip()
            return stripped if stripped else None

        def _collapse_newlines(value: str) -> str:
            value = value.replace("\r\n", "\n").replace("\r", "\n")
            return value.replace("\n", "\\n")

        def _build_structured_text(row: pd.Series) -> str:
            record_id = _format_value(row.get(RECORD_NO_COL)) or "UNKNOWN"
            narrative = row.get(DESCRIPTION_COL, "")
            narrative = narrative.strip() if isinstance(narrative, str) else str(narrative).strip()
            if not narrative:
                narrative = "Narrative not provided."

            facts = []
            for col, label in COLUMN_FACT_LABELS.items():
                value = row.get(col)
                formatted = _format_value(value)
                if formatted:
                    facts.append(f"- {label}: {formatted}")

            for col, (label, true_desc, false_desc) in BOOLEAN_FACT_LABELS.items():
                if col not in row.index or pd.isna(row[col]):
                    continue
                val = row[col]
                if val is True:
                    facts.append(f"- {label}: {true_desc}")
                elif val is False:
                    facts.append(f"- {label}: {false_desc}")

            if len(facts) > FACT_LIMIT:
                facts = facts[:FACT_LIMIT]

            lines = [f"INCIDENT_ID: {record_id}", "NARRATIVE:", narrative]
            if facts:
                lines.extend(["", "FACTS:"])
                lines.extend(facts)

            return _collapse_newlines("\n".join(lines).strip())

        self.df[DESCRIPTION_COL] = self.df.apply(_build_structured_text, axis=1)
        self.df = self.df[[RECORD_NO_COL, DESCRIPTION_COL]]
        return self.df


    # Main Function that would hot-one encode, drop_cols, etc 
    def clean(self):
        """
        Main cleaning function that processes the DataFrame.
        
        This method handles:
        - Column translation to sentences (if onlyDescriptions is True)
        - Dropping specified columns
        - Saving processed data to CSV
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if self.onlyDescriptions:
            # mask = (self.df["TYPE"] == "Hazard Observation")
            # rows_of_interest_idx = self.df[mask].index
            # print(self.df.loc[rows_of_interest_idx, DESCRIPTION_COL])
            description_only_df = self.translateColumnsToSentences()
            description_only_df.to_csv(
                "data/cleaned_description_translated.csv",
                index=False,
                quoting=csv.QUOTE_ALL,
                lineterminator="\n",
            )
            # idx = rows_of_interest_idx[0]  
            # print(description_only_df.loc[idx, DESCRIPTION_COL])

        if self.to_drop:
            print("Dropped columns:")
            for col in self.to_drop:
                print(f" - {col}")
            self.df = self.df.drop(columns=self.to_drop, errors="ignore")
        return self.df
    





    
