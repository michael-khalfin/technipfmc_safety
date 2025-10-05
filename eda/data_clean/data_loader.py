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

        

       
        

