import pandas as pd 
import os 
from typing import Dict, List, Optional, Set
from .column_analyzer import ColumnAnalyzer


class Coalescer: 
    """
    Handles Merging of Df's and Coalescing of Files into Columns
    """

    SYS_RECORD_FIELD = "SYSTEM_OF_RECORD"
    MUTATED = "MUTATED"
    UNKNOWN = "unknown"
    CONFLICT = "CONFLICT"


    def __init__(self, analyzer: Optional[ColumnAnalyzer] = None):
        self.analyzer = analyzer or ColumnAnalyzer()
        self.merge_history = [] # For Debugging in future cases 

    def normalize_series(s: pd.Series):
        s = s.astype("string")
        s = s.str.strip()
        s = s.str.replace(r"\.0+$", "", regex=True)
        s = s.str.lower()
        return s
    
    
    def create_mutated_key(self, df: pd.DataFrame, 
                           record_col: str, 
                           system_col: Optional[str] = None, 
                           drop_original: bool = False) -> pd.DataFrame:
        """
        Creates a unique key by combining Record ID with System Name
        """
        
        system_col = system_col or self.SYS_RECORD_FIELD

        if record_col not in df.columns:
            raise ValueError(f"Column '{record_col}' not foud in dataframe")
        
        if system_col not in df.columns:
            raise ValueError(f"Column '{system_col}' not foud in dataframe")
        
        df = df.copy()

        # Handle Na's as Unknowns
        if pd.api.types.is_categorical_dtype(df[system_col]):
            if self.UNKNOWN not in df[system_col].cat.categories:
                df[system_col] = df[system_col].cat.add_categories([self.UNKNOWN])
            df[system_col] = df[system_col].fillna(self.UNKNOWN)
        else:
            df[system_col] = df[system_col].fillna(self.UNKNOWN).astype('category')

        # covnert to string
        df[record_col] = df[record_col].astype('string')
        df[system_col].astype('string')


        # Create Mutated Key
        mutated_col_name = f"{record_col}_{self.MUTATED}"
        df[mutated_col_name] = df[record_col] + "_" + df[system_col]


        # Drop Columns if Set 
        if drop_original:
            cols_to_drop = [record_col]
            if system_col != self.SYS_RECORD_FIELD:
                cols_to_drop.append(system_col)
            
            df = df.drop(columns = cols_to_drop, errors="ignore")
        
        return df
    

    def namespace_columns(self, df: pd.DataFrame, namespace: str, 
                        exclude_cols: Optional[Set[str]] = set()) -> pd.DataFrame:
        """
        Adds Namespace Suffix to all columns excpet for excluded_cols
        """
        df = df.copy()
        rename_map = {col : f"{col}_{namespace}" for col in df.columns if col not in exclude_cols}
        return df.rename(columns= rename_map)


    def coalesce_column(self, df: pd.DataFrame, dest_col: str, src_col: str,
                        track_conflicts: bool = True, drop_source: bool = True) -> pd.DataFrame:
        """
        Coalesce Source Column Into Dest Column
        Fills Nan's in Dest From Source
        """
        if dest_col not in df.columns or src_col not in df.columns:
            print(f"[Coalesce Issue] Source or Dest. Column DNE")
            return df 

        df = df.copy()
        a_norm, b_norm = self.normalize_series(df[dest_col]), self.normalize_series(df[src_col])

        # Find Conflicts
        both_present = df[dest_col].notna() & df[src_col].notna()
        conflict_mask = both_present & (a_norm != b_norm)

        if track_conflicts and conflict_mask.any():
            conflict_col = f"{dest_col}_{self.CONFLICT}"
            if conflict_col not in df.columns:
                df[conflict_col] = pd.Series(False,  index=df.index, dtype="boolean")
            else:
                df[conflict_col] = df[conflict_col].astype("boolean")
            df.loc[conflict_mask, conflict_col] = True
        
        # Coalesce
        df[dest_col] = df[dest_col].where(df[dest_col].notno(), df[src_col])
        if drop_source: df = df.drop(columns= [src_col], errors="ignore")
        return df 
    
    def merge_and_coalescese(self, main_df: pd.DataFrame, sub_df: pd.DataFrame,
                             main_key: str, sub_record_col:str, namespace:str, 
                             safe_cols: List[str], system_col: Optional[str] = None,
                             verbose: bool = True) -> pd.DataFrame:
        """
        Merge sub_df into main_df and coalesce safe columns
        """
        system_col = system_col or self.SYS_RECORD_FIELD

        # Create Mutated Key 
        sub_df = self.create_mutated_key(sub_df, sub_record_col, system_col, drop_original= True)
        sub_key = f"{sub_record_col}_{self.MUTATED}"

        # Merge
        sub_df = self.namespace_columns(sub_df, namespace, exclude_cols=[sub_key])
        merged = pd.merge(main_df, sub_df, how="left", left_on= main_key, right_on=sub_key)
        if sub_key in merged.columns and sub_key != main_key:
            merged = merged.drop(columns=[sub_key], errors="ignore")
        

        #Coalece Safe Column (save metrics)
        coalesced_count = 0
        filled_count = 0
        if verbose:
            print(f"\n Coalescing {len(safe_cols)} safe columns:")
        
        for col in safe_cols:
            src = f"{col}_{namespace}"
            if src in merged.columns and col in merged.columns:
                fills = (merged[col].isna() & merged[src].notna()).sum()
                conflicts = (merged[col].notna() & merged[src].notna() &
                            (merged[col].astype(str) != merged[src].astype(str))).sum()

                if fills > 0 or conflicts > 0:
                    if verbose:
                        status = f"filled = {fills}"
                        if conflicts > 0:
                            status += f" , conflicts = {conflicts}"
                        print(f"\t[COALESCE] {col} <- {src} : {status}")
                    filled_count += fills
                
                merged = self.coalesce_column(merged, col, src, track_conflicts=True, drop_source= True)
                coalesced_count += 1 
            
        if verbose: print(f"\tSummary: Coalesced {coalesced_count} columns, filled {filled_count} values")
        return merged


    def merge_base_file(self, main_df: pd.DataFrame, sub_df: pd.DataFrame,
                        main_key:str, sub_record_col:str, namespace:str, system_col: Optional[str] = None) -> pd.DataFrame:
        """
        Merge File Into Main
        """
        system_col = system_col or self.SYS_RECORD_FIELD
        sub_df = self.create_mutated_key(sub_df, sub_record_col, system_col, drop_original=True)
        sub_key = f"{sub_record_col}_{self.MUTATED}"

        # Id Cols that should be namespace fur future coalescing 
        id_cols = {col for col in sub_df.columns if self.analyzer.is_id_column(col)}
        rename_map = {col: f"{col}__{namespace}" for col in id_cols if col != sub_key}
        sub_df = sub_df.rename(columns=rename_map)

        merged = pd.merge(main_df, sub_df, how='left', left_on=main_key, right_on=sub_key)

        # Drop Duplicate Join Key
        if sub_key in merged.columns and sub_key != main_key:
            merged = merged.drop(columns=[sub_key], errors="ignore")
        
        return merged


    def merge_file(self, main_df: pd.DataFrame, file_path:str, main_key:str,
                   sub_record_col:str, is_base: bool = False, system_col: Optional[str] = None,
                    verbose: bool =True) -> pd.DataFrame:

        '''
        Load File, analyze columns, and merge it into main_df
        '''

        system_col = system_col or self.SYS_RECORD_FIELD
        namespace = os.path.splitext(os.path.basename(file_path))[0].replace("DIM_CONSOLIDATED_", "")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing: {namespace} {'(BASE)' if is_base else ''}")
            print(f"{'='*70}")
        
        sub_df = pd.read_csv(file_path, low_memory = False)
        if verbose: print(f"Rows: {len(sub_df):,}, Cols: {len(sub_df.columns)}")

        if is_base: result = self.merge_base_file(main_df, sub_df, main_key, sub_record_col, namespace, system_col)
        else:
            sub_df_temp = self.create_mutated_key(
                sub_df.copy(), sub_record_col, system_col, drop_original=False
            )
            sub_key_temp = f"{sub_record_col}_{self.MUTATED}"
            
            # Analyze columns
            analysis = self.analyzer.analyze_columns(
                main_df, sub_df_temp, main_key, sub_key_temp, verbose=verbose
            )

            # Get safe, review and avoid columns
            safe_cols = analysis["safe"]
            review_cols = analysis["review"]
            avoid_cols = analysis["avoid"]
            if verbose and safe_cols: print(f"\tSafe: {', '.join(safe_cols[:5])}{' ...' if len(safe_cols) > 5 else ''}")
            if verbose and review_cols: print(f"\Review: {', '.join(review_cols[:5])}{' ...' if len(review_cols) > 5 else ''}")
            if verbose and avoid_cols: print(f"\Avoid: {', '.join(avoid_cols[:5])}{' ...' if len(avoid_cols) > 5 else ''}")

            result = self.merge_and_coalescese(main_df, sub_df, main_key, sub_record_col, safe_cols, system_col, verbose )

        if verbose: print(f"\tResult: {len(result):,} rows, {len(result.columns)} cols")
        self.merge_history.append({
            "file": file_path,
            "namespace": namespace,
            "is_base": is_base,
            "rows": len(result),
            "cols": len(result.columns)
        })

        return result

    






