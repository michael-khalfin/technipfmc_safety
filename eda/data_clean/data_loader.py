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

    def aggregate_remainder_to_record(self, main_df: pd.DataFrame, main_key: str):
        num_rows = len(main_df)
        num_unique = main_df[main_key].nunique()
        avg_per_key = num_rows / num_unique if num_unique > 0 else 1
        
        if avg_per_key > 1.1:
            if self.verbose:
                print(f"  Aggregating {num_rows:,} rows to {num_unique:,} unique keys")
            
            agg_dict = {col: 'first' for col in main_df.columns if col != main_key}
            main_df = main_df.groupby(main_key, as_index=False).agg(agg_dict)
        
        if self.verbose:
            print(f"  Result: {len(main_df):,} rows, {len(main_df.columns)} cols")
        
        return main_df
        

        

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
        main_key =  f"{master_key}_{self.coalescer.MUTATED}"
        if main_key not in main_df.columns:
            main_df = self.coalescer.create_mutated_key(main_df, master_key, self.coalescer.SYS_RECORD_FIELD, drop_original=False)

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
    
    def attach_remainder_files(self, main_df: pd.DataFrame, master_key:str):
        """
        Attach INJURY_ILLNESS and PROPERTY_DAMAGE_INCIDENT (additional details).
        
        
        Strategy:
        1. Load INJURY_ILLNESS as base (provides injury details for ~5.5% of incidents)
        2. Merge PROPERTY_DAMAGE_INCIDENT into it (~1.8% of incidents)
        3. Aggregate if needed
        4. Merge entire remainder into main
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("MERGING REMAINDER (INJURY_ILLNESS, PROPERTY_DAMAGE) - Additional Details")
            print(f"{'='*70}")

        # Load Injury Df and Mutate Key
        injury_df = pd.read_csv(self._file_path("INJURY_ILLNESS"), low_memory= False)
        if self.verbose:  print(f"  INJURY_ILLNESS: {len(injury_df):,} rows, {len(injury_df.columns)} cols")
        injury_df = self.coalescer.create_mutated_key(injury_df, "MASTER_RECORD_NO", self.coalescer.SYS_RECORD_FIELD, drop_original=False)
        injury_key = f"MASTER_RECORD_NO_{self.coalescer.MUTATED}"

        # Load Property Damage df
        property_df = pd.read_csv(self._file_path("PROPERTY_DAMAGE_INCIDENT"), low_memory=False)
        if self.verbose: print(f"  PROPERTY_DAMAGE_INCIDENT: {len(property_df):,} rows, {len(property_df.columns)} cols")
        property_df = self.coalescer.create_mutated_key(property_df, "MASTER_RECORD_NO", self.coalescer.SYS_RECORD_FIELD, drop_original=True)
        property_key = f"MASTER_RECORD_NO_{self.coalescer.MUTATED}"

        # Merge Property Damage into Injury (both use MASTER_RECORD_NO)
        property_df = self.coalescer.namespace_columns(property_df, "PROPERTY_DAMAGE_INCIDENT", {property_key})
        if injury_key == property_key:
            injury_df = pd.merge(injury_df, property_df, on=injury_key, how='left', suffixes=('', '__PROPERTY'))
        else:
            injury_df = pd.merge(injury_df, property_df, left_on=injury_key, right_on=property_key, how='left')
            injury_df = injury_df.drop(columns=[property_key], errors='ignore')

        injury_df = self.aggregate_remainder_to_record(injury_df, injury_key)


        #preapre to merge on main 
        main_key = f"RECORD_NO_MASTER_{self.coalescer.MUTATED}"
        if main_key not in main_df.columns:
            main_df = self.coalescer.create_mutated_key(main_df, master_key, self.coalescer.SYS_RECORD_FIELD, drop_original=False)

        if master_key in injury_df.columns:
            injury_df = self.coalescer.create_mutated_key(injury_df, master_key, self.coalescer.SYS_RECORD_FIELD, drop_original=False)
            remainder_key = f"RECORD_NO_MASTER_{self.coalescer.MUTATED}"
        else:
            remainder_key = injury_key

        # Analyze and Coalesce
        namespace = "REMAINDER"
        analysis = self.equalizer.analyze_columns(main_df, injury_df, main_key, remainder_key, verbose=self.verbose)
        

        # Merge
        # injury_df_ns = self.coalescer.namespace_columns(injury_df, namespace, {remainder_key})
        # main_df = pd.merge(main_df, injury_df_ns, left_on=main_key, right_on=remainder_key, how='left')
        # main_df = self.coalescer.coalesce_columns(main_df, analysis["safe"], namespace, self.verbose)
        main_df = self.coalescer.merge_and_coalescese(main_df, injury_df, main_key, 
                                                      remainder_key, namespace, analysis["safe"],
                                                      self.coalescer.SYS_RECORD_FIELD, self.verbose)

        # Drop duplicate key
        if remainder_key in main_df.columns and remainder_key != main_key:
            main_df = main_df.drop(columns=[remainder_key], errors='ignore')
        
        if self.verbose: print(f"  After merging remainder: {len(main_df):,} rows, {len(main_df.columns)} cols")
        return main_df


    # def attach_remainder_files(self, main_df: pd.DataFrame, master_key:str) -> pd.DataFrame:
    #     if self.verbose:
    #         print(f"\n{'='*70}")
    #         print("ATTACHING REMAINDER FILES")
    #         print(f"{'='*70}")
        
    #     # List of remainder files to merge: (file_name, join_key_in_file)
    #     remainder_files = [
    #         ("LEADING_INDICATORS", "RECORD_NO"),
    #         ("INJURY_ILLNESS", "MASTER_RECORD_NO"),
    #         ("PROPERTY_DAMAGE_INCIDENT", "MASTER_RECORD_NO")
    #     ]
        
    #     # Create mutated key if it doesn't exist
    #     main_key = f"{master_key}_{self.coalescer.MUTATED}"
    #     if main_key not in main_df.columns:
    #         main_df = self.coalescer.create_mutated_key(main_df, master_key, self.coalescer.SYS_RECORD_FIELD, drop_original=False)
        
    #     # Merge each remainder file using merge_file_with_analysis
    #     for file_name, file_key in remainder_files:
    #         try:
    #             if self.verbose:
    #                 print(f"\n{'='*65}")
    #                 print(f"Merging: {file_name}")
    #                 print(f"{'='*65}")
                
    #             main_df = self.coalescer.merge_file(
    #                 main_df=main_df,
    #                 file_path=self._file_path(file_name),
    #                 main_key=main_key,
    #                 sub_record_col=file_key,
    #                 is_base=False,
    #                 system_col=self.coalescer.SYS_RECORD_FIELD,
    #                 verbose=self.verbose
    #             )
                
    #         except FileNotFoundError:
    #             if self.verbose:
    #                 print(f"  [SKIP] {file_name}: File not found")
    #         except Exception as e:
    #             if self.verbose:
    #                 print(f"  [ERROR] {file_name}: {e}")
    #             import traceback
    #             traceback.print_exc()
        
    #     return main_df
      
    def load_all_data_v1(self, include_actions = False) -> pd.DataFrame:
        master_key = 'RECORD_NO_MASTER'

        # Stack Incident Files (are mutually exclusive - different incidents)
        if self.verbose: 
            print(f"\n{'='*70}")
            print(f"STACKING INCIDENT FILES (Mutually Exclusive)")
            print(f"{'='*70}")
        
        # These files contain different incident types with 0% key overlap
        incident_files = ["LOSS_POTENTIAL", "ACCIDENTS", "HAZARD_OBSERVATIONS", "NEAR_MISSES", "LEADING_INDICATORS"]
        incident_dfs = []
        for file_name in incident_files:
            df = pd.read_csv(self._file_path(file_name), low_memory = False)
            df["SOURCE_FILE"] = file_name
            if self.verbose: print(f"\t{file_name}: {len(df):,} rows, {len(df.columns)} cols")
            incident_dfs.append(df)
        
        incidents = pd.concat(incident_dfs, axis= 0, ignore_index= True, sort = False)
        if self.verbose:print(f"\n  Stacked: {len(incidents):,} rows, {len(incidents.columns)} cols")


        # Attach Remaining Files
        incidents = self.attach_remainder_files(incidents, master_key)


        # Incorporate Action Files 
        if include_actions:
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Incorporating Action FILES")
                print(f"{'='*70}")
            incidents = self.attach_action_files(incidents, master_key)

        
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
       
        

