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

    def normalize_series(self, s: pd.Series):
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

        # Convert to string and normalize
        df[record_col] = df[record_col].astype('string').str.replace(r'\.0$', '', regex=True)
        df[system_col] = df[system_col].astype('string')

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
        df[dest_col] = df[dest_col].where(df[dest_col].notna(), df[src_col])
        if drop_source: df = df.drop(columns= [src_col], errors="ignore")
        return df 

    def coalesce_columns(self, df: pd.DataFrame, columns: List[str], namespace:str, verbose: bool = True) -> pd.DataFrame:
        coalesced_count = 0
        filled_count = 0
        if verbose: print(f"\n Coalescing {len(columns)} safe columns:")

        for col in columns:
            src = f"{col}_{namespace}"
            if src in df.columns and col in df.columns:
                fills = (df[col].isna() & df[src].notna()).sum()
                conflicts = (df[col].notna() & df[src].notna() &
                            (df[col].astype(str) != df[src].astype(str))).sum()
                
                status = f"filled = {fills}, conflicts = {conflicts}"
                filled_count += fills
        
                df = self.coalesce_column(df, col, src, track_conflicts=True, drop_source= True)
                if verbose: print(f"\t[COALESCE] {col} <- {src} : {status}")
                coalesced_count += 1 
            else:
                print(f"\t[DNE] Dest={src} & Source={col}")
                df = df.rename(columns = {col:src})
            
        if verbose: print(f" Summary: Coalesced {coalesced_count} columns, filled {filled_count} values")
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
    
        #Coalece Safe Column
        merged = self.coalesce_columns(merged, safe_cols, namespace, verbose)

        # Remove Suffix If Base DNE
        suffix = f"_{namespace}"
        rename_map = {}
        for col in merged.columns:
            if suffix in col:
                base_col = col.replace(suffix, "")
                if base_col not in merged.columns:
                    rename_map[col] = base_col
        merged = merged.rename(columns=rename_map)
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

        system_col = self.SYS_RECORD_FIELD
        namespace = os.path.splitext(os.path.basename(file_path))[0].replace("DIM_CONSOLIDATED_", "")
        
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
            if verbose and review_cols: print(f"\tReview: {', '.join(review_cols[:5])}{' ...' if len(review_cols) > 5 else ''}")
            if verbose and avoid_cols: print(f"\nAvoid: {', '.join(avoid_cols[:5])}{' ...' if len(avoid_cols) > 5 else ''}")


            if not safe_cols:
                if verbose:
                    print("\tNo SAFE columns → skipping merge for this file.")
                return main_df

            needed_cols = [sub_record_col, system_col] + safe_cols
            needed_cols = [c for c in needed_cols if c in sub_df.columns]
            sub_df_reduced = sub_df[needed_cols].copy()

            result = self.merge_and_coalescese(
                main_df=main_df,
                sub_df=sub_df_reduced,   # << use reduced df
                main_key=main_key,
                sub_record_col=sub_record_col,
                namespace=namespace,
                safe_cols=safe_cols,
                system_col=system_col,
                verbose=verbose,
            )


        if verbose: print(f"\tResult: {len(result):,} rows, {len(result.columns)} cols")
        self.merge_history.append({
            "file": file_path,
            "namespace": namespace,
            "is_base": is_base,
            "rows": len(result),
            "cols": len(result.columns)
        })

        return result

    






