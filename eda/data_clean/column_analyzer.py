import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional


class ColumnAnalyzer:
    """
    Analyzes columns between dataframes to determine if they can be safely merged.
    """
    def __init__(
        self,
        min_coverage: float = 0.95,
        min_equal_non_null: float = 0.95,
        max_conflict_rate: float = 0.02,
        max_distinct_auto: int = 64,
        review_band: Tuple[float, float] = (0.85, 0.95),
        allow_high_card_if_exact: bool = True,
        exact_equal_thresh: float = 0.999,
        exact_conflict_ceiling: float = 1e-6,
    ):
        self.min_coverage = min_coverage
        self.min_equal_non_null = min_equal_non_null
        self.max_conflict_rate = max_conflict_rate
        self.max_distinct_auto = max_distinct_auto
        self.review_band = review_band
        self.allow_high_card_if_exact = allow_high_card_if_exact
        self.exact_equal_thresh = exact_equal_thresh
        self.exact_conflict_ceiling = exact_conflict_ceiling
        

        self.id_like_tokens = ("_id", "id_", "_no", "no_", "_key", "key_", "record_no", "action_no", "master_record")
        self.date_like_tokens = ("date", "time", "approval", "closure", "due", "reported", "incident")
    

    # Helper Funcs 
    def normalize_series(self, s: pd.Series) -> pd.Series:
        s = s.astype("string")
        s = s.str.strip()
        s = s.str.replace(r"\.0+$", "", regex=True)
        s = s.str.lower()
        return s
    
    def is_id_column(self, col_name: str) -> bool:
        return any(tok in col_name.lower() for tok in self.id_like_tokens)
    
    def is_date_column(self, col_name: str) -> bool:
        return any(tok in col_name.lower() for tok in self.date_like_tokens)
    
    def is_small_cardinality(self, distinct_left: int, distinct_right: int, col_name: str) -> bool:
        return (
            (distinct_left <= self.max_distinct_auto and distinct_right <= self.max_distinct_auto)
            or self.is_date_column(col_name)
        )
    

    # Main Funcs
    def compare_columns_by_key(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        key_left: str,
        key_right: str,
        col_name: str,
        how: str = "inner"
    ) -> Dict:
        """
        Compare a column between two dataframes after joining on keys.
        
        Returns metrics about equality, conflicts, and fill potential.
        """
        if col_name not in left_df.columns:
            raise ValueError(f"Column {col_name} not in left_df")
        if col_name not in right_df.columns:
            raise ValueError(f"Column {col_name} not in left_df")
        
        # Deduplicate right on key
        right_dedup = right_df[[key_right, col_name]].drop_duplicates(subset=[key_right], keep='first')
        left_subset = left_df[[key_left, col_name]]
        
        # Merge to align
        aligned = pd.merge(
            left_subset, right_dedup,
            how=how,
            left_on=key_left,
            right_on=key_right,
            suffixes=("_L", "_R")
        )
        
        # Get the column names after merge
        left_col = f"{col_name}_L" if f"{col_name}_L" in aligned.columns else col_name
        right_col = f"{col_name}_R" if f"{col_name}_R" in aligned.columns else col_name
        
        a_raw = aligned[left_col]
        b_raw = aligned[right_col]
        a = self.normalize_series(a_raw)
        b = self.normalize_series(b_raw)
        
        matched_rows = len(aligned)
        both_nan = a_raw.isna() & b_raw.isna()
        both_present = a_raw.notna() & b_raw.notna()
        exact_equal = both_present & (a == b)
        value_conflict = both_present & (a != b)
        one_nan_left = a_raw.isna() & b_raw.notna()
        one_nan_right = a_raw.notna() & b_raw.isna()
        
        cnt_both_nan = int(both_nan.sum())
        cnt_exact = int(exact_equal.sum())
        cnt_conflict = int(value_conflict.sum())
        cnt_lnan = int(one_nan_left.sum())
        cnt_rnan = int(one_nan_right.sum())
        cnt_both_pres = int(both_present.sum())
        
        def rate(x, d):
            return (x / d) if d else 1.0
        
        return {
            "matched_rows": matched_rows,
            "both_present": cnt_both_pres,
            "exact_equal": cnt_exact,
            "both_nan": cnt_both_nan,
            "one_nan_left": cnt_lnan,
            "one_nan_right": cnt_rnan,
            "value_conflict": cnt_conflict,
            "equal_including_nan": rate(cnt_exact + cnt_both_nan, matched_rows),
            "equal_non_null": rate(cnt_exact, cnt_both_pres) if cnt_both_pres > 0 else None,
            "conflict_rate": rate(cnt_conflict, matched_rows),
            "fill_rate": rate(cnt_lnan, matched_rows),
        }
    
    def compare_column_values(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        col_name: str
    ) -> Dict:
        """
        Compare value distributions between two dataframes (without key alignment).
        Uses multiset comparison.
        """
        def _value_multiset(s: pd.Series) -> Counter:
            s = self.normalize_series(s)
            s = s.dropna()
            return Counter(s.tolist())
        
        L = _value_multiset(left_df[col_name])
        R = _value_multiset(right_df[col_name])
        nL = sum(L.values())
        nR = sum(R.values())
        
        keys = set(L) | set(R)
        inter = sum(min(L[k], R[k]) for k in keys)
        union = sum(max(L[k], R[k]) for k in keys) if keys else 0
        jaccard = (inter / union) if union else 1.0
        coverage_L_in_R = (inter / nL) if nL else 1.0
        coverage_R_in_L = (inter / nR) if nR else 1.0
        
        return {
            "left_non_null": nL,
            "right_non_null": nR,
            "distinct_left": len(L),
            "distinct_right": len(R),
            "distinct_overlap": len(set(L) & set(R)),
            "jaccard": jaccard,
            "coverage_left_in_right": coverage_L_in_R,
            "coverage_right_in_left": coverage_R_in_L,
        }
    
    def analyze_columns(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        key_left: str,
        key_right: str,
        cols: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, List[str]]:
        """
        Analyze common columns and categorize them as SAFE, REVIEW, or AVOID.
        """
        if cols is None:
            cols = sorted((set(left_df.columns) & set(right_df.columns)) - {key_left, key_right})
        
        safe, review, avoid = [], [], []
        details = {}
        
        for col in cols:
            # Skip ID columns
            if self.is_id_column(col):
                avoid.append(col)
                details[col] = {"decision": "avoid", "reason": "ID column"}
                continue
            
            # Get value-level comparison
            val_metrics = self.compare_column_values(left_df, right_df, col)
            
            # Get key-aligned comparison
            key_metrics = self.compare_columns_by_key(
                left_df, right_df, key_left, key_right, col
            )
            
            if "error" in key_metrics:
                avoid.append(col)
                details[col] = {"decision": "avoid", "reason": key_metrics["error"]}
                continue
            
            # Extract key metrics
            eq_nonnull = key_metrics["equal_non_null"] #maybe hold these as actaul values?
            conflict = key_metrics["conflict_rate"]
            fill_rate = key_metrics["fill_rate"]
            both_pres = key_metrics["both_present"]
            
            covL = val_metrics["coverage_left_in_right"]
            covR = val_metrics["coverage_right_in_left"]
            
            distinct_left = val_metrics["distinct_left"]
            distinct_right = val_metrics["distinct_right"]
            small_card = self.is_small_cardinality(distinct_left, distinct_right, col)
            
            # Decision logic
            decision = "avoid"
            reason = ""
            
            # High equality, exact match (even high cardinality)
            if (self.allow_high_card_if_exact and eq_nonnull is not None and 
                eq_nonnull >= self.exact_equal_thresh and conflict <= self.exact_conflict_ceiling):
                decision = "safe"
                reason = "exact_match"
            
            #  Standard safe merge
            elif (small_card and (covL >= self.min_coverage or covR >= self.min_coverage) and
                  eq_nonnull is not None and eq_nonnull >= self.min_equal_non_null and 
                  conflict <= self.max_conflict_rate):
                decision = "safe"
                reason = "high_equality_low_conflict"
            
            # Fill-only (no overlapping values) - UNION scenario
            elif both_pres == 0:
                # Check if either column has substantial data
                left_filled = val_metrics["left_non_null"]
                right_filled = val_metrics["right_non_null"]
                
                if left_filled > 0 or right_filled > 0:
                    decision = "safe"
                    reason = "union_fill_only"
                else:
                    decision = "avoid"
                    reason = "both_empty"
            
            # Review band
            elif ((covL >= self.review_band[0] or covR >= self.review_band[0]) and 
                  eq_nonnull is not None and eq_nonnull >= 0.80):
                decision = "review"
                reason = "moderate_match"
            
            else:
                decision = "avoid"
                reason = "low_match_or_high_conflict"
            
            # Store decision
            if decision == "safe":
                safe.append(col)
            elif decision == "review":
                review.append(col)
            else:
                avoid.append(col)
            
            details[col] = {
                "decision": decision,
                "reason": reason,
                "metrics": {
                    "equal_non_null": eq_nonnull,
                    "conflict_rate": conflict,
                    "fill_rate": fill_rate,
                    "coverage_L": covL,
                    "coverage_R": covR,
                    "distinct_L": distinct_left,
                    "distinct_R": distinct_right,
                }
            }
        
        if verbose:
            print(f"\nEqualizer Analysis:")
            print(f"\tSAFE: {len(safe)} columns")
            print(f"\tREVIEW: {len(review)} columns")
            print(f"\tAVOID: {len(avoid)} columns")
        
        return {
            "safe": sorted(safe),
            "review": sorted(review),
            "avoid": sorted(avoid),
            "details": details
        }
