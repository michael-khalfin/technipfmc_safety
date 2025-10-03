import clean_data.utils as utils
import pandas as pd
import os
from typing import Optional, List, Tuple

SYS_RECORD_FIELD = "SYSTEM_OF_RECORD"
MUTATED = "MUTATED"
NO = "NO"
UNKNOWN = "unknown"


def merge_df_to_main(main_df, main_record, sub_df, overwrite_record = None, overwrite_system_field = None, to_drop = None, file_path = None):

    # Overwrite Values if needed
    if overwrite_record is None:
        sub_record = utils.find_col(sub_df, NO) 
    else:
        sub_record = overwrite_record
        print(f"Detected Record Overwrite {overwrite_record}!")

    
    if overwrite_system_field is None:
        sub_system_field = SYS_RECORD_FIELD
    else:
        sub_system_field = overwrite_system_field
        print(f"Detected System Overwrite {overwrite_record}!")

    # Attach System to Record No 
    sub_df = utils.attach_system_to_record(sub_df, sub_record, sub_system_field)
    sub_record_mutated = sub_record + f"_{MUTATED}"

    # Drop System And Old record field 
    drop_cols = set(to_drop or [])
    drop_cols.update([sub_record])  # keep mutated join key only
    sub_df = sub_df.drop(columns=[c for c in drop_cols if c in sub_df.columns], errors="ignore")

    # Rename to avoid duplicates 
    if file_path:
        ns = os.path.splitext(os.path.basename(file_path))[0].replace("DIM_CONSOLIDATED_", "")
        sub_df = utils.namespace_subcolumns(sub_df, keep_cols=[sub_record_mutated], ns=ns)

    merged_df = pd.merge(
        main_df, sub_df, how='left',
        left_on=main_record, right_on=sub_record_mutated
    )

    
    utils.check_duplicates(merged_df)

    return merged_df

def merge_on_column(df1, df2, df1_col, df2_col, how='left', suffixes=('', '__R')):
    # Work on copies to avoid mutating callers
    df1 = df1.copy()
    df2_renamed = df2.rename(columns={df1_col: df2_col}).copy()

    if df2_col in df2_renamed.columns:
        df2_renamed = df2_renamed.drop_duplicates(subset=[df2_col])


    def _normalize_join_key(s: pd.Series) -> pd.Series:
        s = s.astype('string')                 # preserves <NA>
        s = s.str.strip()
        s = s.str.replace(r'\.0+$', '', regex=True)  # "123.0" -> "123"
        return s

    # Ensure the join column exists on both sides (df1 must already have df2_col)
    if df2_col not in df1.columns:
        raise KeyError(f"Left DataFrame is missing join column '{df2_col}'")

    df1[df2_col] = _normalize_join_key(df1[df2_col])
    df2_renamed[df2_col] = _normalize_join_key(df2_renamed[df2_col])

    merged = pd.merge(df1, df2_renamed, on=df2_col, how=how, suffixes=suffixes)

    # If df1_col also exists post-merge (because of renaming), drop the duplicate
    merged = merged.drop(columns=[c for c in [df1_col] if c in merged.columns], errors='ignore')
    return merged

def merge_file_to_main(main_df, main_record, file_path, overwrite_record = None, overwrite_system_field = None, to_drop = None):
    print(f"Processing {file_path}")

    sub_df = pd.read_csv(file_path, low_memory = False)
    return merge_df_to_main(main_df, main_record, sub_df, overwrite_record, overwrite_system_field, to_drop, file_path)
  


def coalesce_into(
    df: pd.DataFrame,
    dest_col: str,
    src_col: str,
    create_conflict_flag: bool = True,
    conflict_flag_col: Optional[str] = None
) -> pd.DataFrame:
    if dest_col not in df.columns or src_col not in df.columns:
        return df

    # normalized comparison for conflict detection
    a = utils._normalize_series(df[dest_col])
    b = utils._normalize_series(df[src_col])

    # A “conflict” row = both present AND not equal after normalization
    both_present = ~df[dest_col].isna() & ~df[src_col].isna()
    conflict_mask = both_present & (a != b)

    if create_conflict_flag:
        flag_name = conflict_flag_col or f"{dest_col}__CONFLICT"
        if flag_name not in df.columns:
            df[flag_name] = False
        df.loc[conflict_mask, flag_name] = True

    # Coalesce: prefer existing dest, otherwise pull from src
    df[dest_col] = df[dest_col].where(~df[dest_col].isna(), df[src_col])
    return df



def consolidated_matching_cols(
    main_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    to_drop: Optional[List[str]] = None,
    to_consolidate: Optional[List[Tuple[str, str]]] = None,
    equality_tol_all: float = 0.995,
    equality_tol_both: float = 0.99,
) -> pd.DataFrame:

    to_drop = set(to_drop or [])
    to_consolidate = list(to_consolidate or [])

    # --- detect the join key on main (must be a single *_MUTATED column) ---
    mutated_cols = [c for c in main_df.columns if c.endswith(f"_{MUTATED}")]
    if len(mutated_cols) != 1:
        raise ValueError(f"Could not uniquely infer main join key; found {mutated_cols}. "
                         f"Pass through your merge path instead.")
    main_join = mutated_cols[0]

    # Determine the raw record + system columns on sub_df so merge_df_to_main can attach the mutated key.
    # If your sub_df lacks the expected columns, your overwrite logic in merge_df_to_main will kick in.
    # We do not do namespacing here; we manage overlaps manually.
    common_cols = set(main_df.columns).intersection(set(sub_df.columns))

    # Never treat the join key or system field as a “content overlap”
    protected = {main_join, SYS_RECORD_FIELD}
    overlap = [c for c in common_cols if c not in protected]

    # 1) Decide drops vs. coalesces automatically (unless the user passes explicit instructions)
    auto_drop = []
    auto_coalesce = []   # (main_col, sub_col_subname)
    sub_renames = {}

    for col in overlap:
        # Compare main vs sub version of the same-named column
        # (We do comparison on a temporary left-join by index to align rows)
        temp = pd.DataFrame({
            f"__main_{col}": main_df[col],
            f"__sub_{col}": sub_df[col]
        }, index=main_df.index if len(main_df) == len(sub_df) else None).copy()

        # If lengths differ, we just compare available values by aligning lengths safely
        a = main_df[col]
        b = sub_df[col].reindex(main_df.index) if len(sub_df) != len(main_df) else sub_df[col]

        temp = pd.DataFrame({f"__main_{col}": a, f"__sub_{col}": b})
        # Use the equivalence check on the aligned temp
        tmp = temp.rename(columns={f"__main_{col}": "A", f"__sub_{col}": "B"})
        try:
            same = utils.columns_essentially_equal(tmp, "A", "B",
                                             tol_all=equality_tol_all,
                                             tol_both=equality_tol_both)
        except Exception:
            same = False

        if same:
            auto_drop.append(col)  # sub column is redundant
        else:
            # keep the sub col but with a "__SUB" suffix to avoid collisions
            sub_name = f"{col}__SUB"
            sub_renames[col] = sub_name
            # If they look similar (but not perfect), we will coalesce after merge
            r = utils.columns_equivalence_report(tmp, "A", "B", show_conflicts=0)
            if r["equal_rate_including_nans"] >= 0.9 and r["equal_rate_where_both_present"] >= 0.9:
                auto_coalesce.append((col, sub_name))

    # Respect explicit instructions
    to_drop.update(auto_drop)
    to_consolidate = list(set(to_consolidate + auto_coalesce))

    # 2) Prepare sub_df: drop duplicates, rename overlaps that we keep
    sub_df_prep = sub_df.drop(columns=[c for c in to_drop if c in sub_df.columns], errors="ignore").copy()
    if sub_renames:
        sub_df_prep = sub_df_prep.rename(columns=sub_renames)

    # 3) Merge using your standardized path (which will attach the MUTATED join on the right)
    merged = merge_df_to_main(
        main_df=main_df,
        main_record=main_join,
        sub_df=sub_df_prep,
        overwrite_record=None,              # your internal inference handles it
        overwrite_system_field=None,
        to_drop=[],                         # already dropped
        file_path=None                      # skip file-based namespacing here
    )

    # 4) Coalesce post-merge for chosen pairs
    for main_col, sub_col in to_consolidate:
        if (main_col in merged.columns) and (sub_col in merged.columns):
            merged = coalesce_into(merged, main_col, sub_col, create_conflict_flag=True)
            # After coalescing, you can drop the sub copy if you wish:
            merged = merged.drop(columns=[sub_col], errors="ignore")

    return merged




