import pandas as pd
from typing import List, Optional 
from .column_analyzer import ColumnAnalyzer
from .coaleser import Coalescer


class DataLoader:
    """
    Merges and Loads All Safety Data Files (horizontal coalescing and stacking)
    """

    def __init__(self, data_dir: str = 'data/', verbose: bool = True):
        self.verbose = verbose
        self.equalizer = ColumnAnalyzer()
        self.coalescer = Coalescer(self.equalizer)
        self.df = None

        self.data_dir = data_dir
        self.prefix = "DIM_CONSOLIDATED_"
    
    def _file_path(self, name:str) -> str:
        return f"{self.data_dir}{self.prefix}{name}.csv"
    

    def load_base(self) -> pd.DataFrame:
        BASE = "LOSS_POTENTIAL"
        record_col = "RECORD_NO_LOSS_POTENTIAL"
        file_path = self._file_path(BASE)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"LOADING BASE: {BASE}")
            print(f"{'='*70}")
        
        df = pd.read_csv(file_path, low_memory= False)
        df = self.coalescer.create_mutated_key(df, record_col, self.coalescer.SYS_RECORD_FIELD, drop_original= False)
        if self.verbose:
            print(f"  Rows: {len(df):,}, Cols: {len(df.columns)}")
            print(f"  Key: {record_col}_{self.coalescer.MUTATED}")
        return df 
        


    def load_all_data_v1(self, include_actions = False) -> pd.DataFrame:

        # Stack Loss Potential Files (are mutually exclusive)
        if self.verbose: 
            print(f"\n{'='*70}")
            print(f"STACKING LOSS FILES (Mutually Exclusive)")
            print(f"{'='*70}")
        loss_files = ["LOSS_POTENTIAL", "ACCIDENTS", "HAZARD_OBSERVATIONS", "NEAR_MISSES"]
        loss_dfs = []
        for file_name in loss_files:
            df = pd.read_csv(self._file_path(file_name), low_memory = False)
            df["SOURCE_FILE"] = file_name
            if self.verbose: print(f"\t{file_name}: {len(df):,} rows")
            loss_dfs.append(df)
        
        incidents = pd.concat(loss_dfs, axis= 0, ignore_index= True, sort = False)
        if self.verbose:print(f"\n  Stacked: {len(incidents):,} rows, {len(incidents.columns)} cols")

        return incidents
       
        

