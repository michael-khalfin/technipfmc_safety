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
    

    def aggregate_actions_to_record_no(self, combined: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates Multiple Actions To Its Record NO
        """
        record_no = "RECORD_NO" 
        if record_no in combined.columns:
            avg_actions = len(combined) / combined[record_no].nunique()

            if avg_actions > 1.5:
                if self.verbose: print(f" Aggregating {len(combined)} actions to {combined[record_no].nunique()} records")


                # EXPIREMENTAL: take first value per RECORD_NO
                agg_dict = {col: 'first' for col in combined.columns if col != 'RECORD_NO'}
                action_counts = combined.groupby('RECORD_NO').size().reset_index(name='ACTION_COUNT')
                combined = combined.groupby('RECORD_NO', as_index=False).agg(agg_dict)
                combined = combined.merge(action_counts, on='RECORD_NO', how='left')
                
                if self.verbose:
                    print(f" Result: {len(combined):,} rows, {len(combined.columns)} cols")
        
        return combined
        

    def attach_action_files(self, main_df: pd.DataFrame, master_key: str) -> pd.DataFrame:
        """
        Attach Action Plan and CAPA Action Plan
        Strategy:
        1. Merge ACTION_PLAN with CAPA_ACTION_PLAN on ACTION_NO
        2. Aggrete Actions to one row per RECORD_NO
        3. Merge into main on main_record
        """
        action_df = pd.read_csv(self._file_path("ACTION_PLAN"), low_memory = False)
        capa_df = pd.read_csv(self._file_path("CAPA_ACTION_PLAN"), low_memory =False)

        if self.verbose:
            print(f"  ACTION_PLAN: {len(action_df):,} rows, {len(action_df.columns)} cols")
            print(f"  CAPA_ACTION_PLAN: {len(capa_df):,} rows, {len(capa_df.columns)} cols")
        
        # Save system field for later
        sub_system_record = 'SYS_FOR_RECORD'
        if self.coalescer.SYS_RECORD_FIELD in action_df.columns:
            action_df[sub_system_record] = action_df[self.coalescer.SYS_RECORD_FIELD].copy()

        # Merge on ACTION_NO 
        action_df_mut = self.coalescer.create_mutated_key(
            action_df, "ACTION_NO", self.coalescer.SYS_RECORD_FIELD, drop_original=True
        )
        capa_df_mut = self.coalescer.create_mutated_key(
            capa_df, "ACTION_NO", self.coalescer.SYS_RECORD_FIELD, drop_original=True
        )

        combined = pd.merge(
            action_df_mut, capa_df_mut,
            on=f"ACTION_NO_{self.coalescer.MUTATED}",
            how="left",
            suffixes=('', '__CAPA')
        )

        # Coalesce CAPA Columns
        capa_cols = [c for c in combined.columns if c.endswith('__CAPA')]
        base_cols = [c.replace('__CAPA', '') for c in capa_cols if c.replace('__CAPA', '') in combined.columns]
        combined = self.coalescer.coalesce_columns(combined, base_cols, '_CAPA', verbose=True)

        # Restore system field and drop ACTION_NO_MUTATED
        if sub_system_record in combined.columns:
            combined[self.coalescer.SYS_RECORD_FIELD] = combined[sub_system_record]
            combined = combined.drop(columns=[sub_system_record], errors='ignore')
        combined = combined.drop(columns=[f"ACTION_NO_{self.coalescer.MUTATED}"], errors='ignore')

        # Aggregate to One row Per Record_NO
        combined = self.aggregate_actions_to_record_no(combined)

        ### Step 2: Aggregate to Main DF
        main_df = self.coalescer.create_mutated_key(main_df, master_key, 
                    self.coalescer.SYS_RECORD_FIELD, 
                    drop_original=False)
        main_key =  f"{master_key}_{self.coalescer.MUTATED}"


        # Merge Combined actions (both capa and action plans)
        namespace = "COMBINED_ACTIONS"
        combined_mut = self.coalescer.create_mutated_key(combined, "RECORD_NO", self.coalescer.SYS_RECORD_FIELD, drop_original=False)
        combined_key = f"RECORD_NO_{self.coalescer.MUTATED}"
        
        # Analyze columns
        analysis = self.equalizer.analyze_columns(main_df, combined_mut, main_key, combined_key, verbose=self.verbose)
        
        # Merge and coalesce
        main_df = self.coalescer.merge_and_coalescese(
            main_df, combined, main_key, "RECORD_NO", namespace, 
            analysis["safe"], verbose=self.verbose)

        return main_df


    def load_all_data_v1(self, include_actions = False) -> pd.DataFrame:
        master_key = 'RECORD_NO_MASTER'
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


        # Incorporate Action Files 
        if include_actions:
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Incorporating Action FILES")
                print(f"{'='*70}")
            incidents = self.attach_action_files(incidents, master_key)

        return incidents
       
        

