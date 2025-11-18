"""
Data loading module for safety incident data integration.

This module provides comprehensive data loading capabilities to merge
and integrate multiple safety data files from different sources.
"""

import pandas as pd
from typing import List, Optional 
from .column_analyzer import ColumnAnalyzer
from .coaleser import Coalescer

class DataLoader:
    """
    Merges and Loads All Safety Data Files (horizontal coalescing and stacking).
    
    This class provides methods to:
    - Load and merge multiple safety data files
    - Handle different data formats and structures
    - Integrate action plans and leading indicators
    - Clean and prepare data for analysis
    """

    def __init__(self, data_dir: str = 'data/', verbose: bool = True):
        """
        Initialize DataLoader with data directory and processing options.
        
        Args:
            data_dir (str): Directory containing safety data files
            verbose (bool): Whether to print detailed processing information
        """
        self.verbose = verbose
        self.equalizer = ColumnAnalyzer()
        self.coalescer = Coalescer(self.equalizer)
        self.df = None

        self.data_dir = data_dir
        self.prefix = "DIM_CONSOLIDATED_"
    
    def _file_path(self, name:str) -> str:
        return f"{self.data_dir}{self.prefix}{name}.csv"
    

    def _coarce_boolean(self, df: pd.DataFrame):
        def _coerce_binary_like(s: pd.Series) -> pd.Series:
            mapping = {
                "Y": True, "N": False, "YES": True, "NO": False,
                "True": True, "False": False, "true": True, "false": False,
                "1": True, "0": False, 1: True, 0: False
            }
            s2 = s.map(lambda x: mapping.get(x, x))
            if pd.Series(s2).dropna().isin([True, False]).all():
                return s2.astype("boolean")
            return s
        
        for c in df.columns:
            if df[c].dtype == "bool":
                df[c] = df[c].astype("boolean")  
            elif df[c].dtype == "object" and df[c].dropna().nunique() <= 2:
                df[c] = _coerce_binary_like(df[c])
        return df  

    def load_all_data(self, include_actions = False) -> pd.DataFrame:
        master_key = 'RECORD_NO_MASTER'
        BASE_FILE = "LOSS_POTENTIAL"
        BASE_FILE_RECORD = "RECORD_NO_LOSS_POTENTIAL"
        incidents = pd.read_csv(self._file_path(BASE_FILE), low_memory = False)
        incidents[BASE_FILE_RECORD] = incidents[BASE_FILE_RECORD].astype("string").str.replace(r"\.0$", "", regex=True)
        incidents = self.coalescer.create_mutated_key(
            incidents, BASE_FILE_RECORD, self.coalescer.SYS_RECORD_FIELD, drop_original=False
        )
        main_key = f"{BASE_FILE_RECORD}_{self.coalescer.MUTATED}"

        if self.verbose: 
            print(f"\n{'='*70}")
            print(f"STACKING MUTUALLY EXCLUSIVE INCIDENT FILES")
            print(f"{'='*70}")
        
        # Only stack files with 0% overlap
        incident_files = ["ACCIDENTS_translated", "HAZARD_OBSERVATIONS_translated", "NEAR_MISSES_translated"]
        for file_name in incident_files:
            df = pd.read_csv(self._file_path(file_name), low_memory = False)
            df[BASE_FILE_RECORD] = df[BASE_FILE_RECORD].astype("string").str.replace(r"\.0$", "", regex=True)

            df["SOURCE_FILE"] = file_name
            if self.verbose: print(f"\t{file_name}: {len(df):,} rows, {len(df.columns)} cols")

            analysis = self.equalizer.analyze_columns(incidents, df, BASE_FILE_RECORD, BASE_FILE_RECORD, verbose=self.verbose)
            incidents = self.coalescer.merge_and_coalescese(incidents, df, main_key, BASE_FILE_RECORD, file_name, analysis["safe"],
                                                        system_col= self.coalescer.SYS_RECORD_FIELD, verbose = self.verbose)
        
        if self.verbose:print(f"\n  Stacked: {len(incidents):,} rows, {len(incidents.columns)} cols")


        # incidents = self.attach_leading_indicators(incidents, master_key)
        # incidents = self.attach_remainder_files(incidents, master_key)

        # Incorporate Action Files 
        if include_actions:
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Incorporating Action FILES")
                print(f"{'='*70}")
            # incidents = self.attach_action_files(incidents, master_key)


        description_cols = [
                col for col in incidents.columns
                if any(kw in col.lower() for kw in ["description", "descrp", "desc"])
            ]
        print(f"Description Cols: {description_cols}")

        # Coerce Types N Drop Rows That Don't Contain Description
        incidents = self._coarce_boolean(incidents)
        incidents = incidents[incidents["DESCRIPTION"].notna()]
        
        
        # Final summary
        if self.verbose:
            print(f"\n{'='*70}")
            print("FINAL HYBRID RESULT")
            print(f"{'='*70}")
            print(f"  Total rows: {len(incidents):,}")
            print(f"  Total columns: {len(incidents.columns)}")
            print(f"  Memory: {incidents.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            
            suffixed = [c for c in incidents.columns if '__' in c]
            print(f"  Suffixed columns: {len(suffixed)}")

            description_cols = [
                col for col in incidents.columns
                if any(kw in col.lower() for kw in ["description", "descrp", "desc"])
            ]
            print(f"Description Cols: {description_cols}")



        return incidents
       
        

