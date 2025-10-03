
import pandas as pd
import os 


MAIN_FILE = "data/DIM_CONSOLIDATED_LOSS_POTENTIAL.csv"
SYS_RECORD_FIELD = "SYSTEM_OF_RECORD"
MUTATED, NO = "MUTATED", "NO"
UNKNOWN = "unknown"


#Helper to Find Desired Column Based On keyword
def find_col(df, keyword):
    columns, keyword = df.columns,  keyword.lower()
    main_record = None

    for col in columns:
        col_parts = col.lower().split("_")
        if any(p == keyword for p in col_parts) or col.lower() == keyword:
            if main_record is None: 
                main_record = col
                print(f"Main Column: {main_record}")
            else:
                print(f"Found Another Potential Column : {col}")

    if main_record is None: raise ValueError(f"Could not find column matching keyword '{keyword}'")
    return main_record




def attach_system_to_record(df, record_col, system_col):
    # Use pandas StringDtype / category instead of Python object strings
    df[system_col] = df[system_col].fillna(UNKNOWN).astype('category')
    df[record_col] = df[record_col].astype('string')
    
    df[record_col + f"_{MUTATED}"] = df[record_col] + "_" + df[system_col].astype('string')
    df = df.drop(columns=[record_col])
    return df


def namespace_subcolumns(sub_df, keep_cols,  ns):
    keep = set(keep_cols)
    newcols = {
        c: (c if c in keep else f"{c}__{ns}")
        for c in sub_df.columns
    }
    return sub_df.rename(columns=newcols)

def count_common_columns(df1, df2):
    common_cols = set(df1.columns).intersection(set(df2.columns))
    print(f"Common columns: {common_cols}")
    return len(common_cols)


def merge_file_to_main(main_df, main_record, file_path, overwrite_record = None, overwrite_system_field = None, to_drop = None):
    print(f"Processing {file_path}")

    sub_df = pd.read_csv(file_path, low_memory = False)

    # Overwrite Values if needed
    if overwrite_record is None:
        sub_record = find_col(sub_df, NO) 
    else:
        sub_record = overwrite_record
        print(f"Detected Record Overwrite {overwrite_record}!")

    
    if overwrite_system_field is None:
        sub_system_field = SYS_RECORD_FIELD
    else:
        sub_system_field = overwrite_system_field
        print(f"Detected System Overwrite {overwrite_record}!")

    # Attach System to Record No 
    sub_df = attach_system_to_record(sub_df, sub_record, sub_system_field)
    sub_record_mutated = sub_record + f"_{MUTATED}"

    # Build source namespace from filename, e.g. "ACCIDENTS"
    ns = os.path.splitext(os.path.basename(file_path))[0].replace("DIM_CONSOLIDATED_", "")

    # Drop System And Old record field 
    drop_cols = set(to_drop or [])
    drop_cols.update([sub_record])  # keep mutated join key only
    sub_df = sub_df.drop(columns=[c for c in drop_cols if c in sub_df.columns], errors="ignore")

    # Rename to avoid duplicates 
    sub_df = namespace_subcolumns(sub_df, keep_cols=[sub_record_mutated], ns=ns)

    merged_df = pd.merge(
        main_df, sub_df, how='left',
        left_on=main_record, right_on=sub_record_mutated
    )

    

    check_duplicates(merged_df)

    return merged_df


def print_df(target_df, record, name):
    rows, cols = target_df.shape
    memory_mb = target_df.memory_usage(deep=True).sum() / (1024 * 1024)
    summary = []
    summary.append({
        "Rows": rows,
        "Columns": cols,
        "Volume_MB": round(memory_mb, 2),
        f"{record}" : target_df[record].notna().sum(),
    })


    # print(sub_df[[sub_record, sub_system_field, sub_record + "_MUTATED"]].head())

    summary_df = pd.DataFrame(summary)
    print(f"\nDataset Summary For {name}:")
    print(summary_df.to_string(index=False))
    print('\n')


def check_duplicates(main_df, remove= False):
    dups = main_df.columns[main_df.columns.duplicated(keep=False)]
    if len(dups) > 0:
        print("[WARN] Duplicated columns found:")
        for col in dups.unique():
            positions = [i for i, c in enumerate(main_df.columns) if c == col]
            print(f"  {col}: positions {positions}")

def count_system_aware_matches(
    df,
    master_rec="RECORD_NO_MASTER",
    master_sys="SYSTEM_OF_RECORD",
    show_conflicts=0,          # set >0 to print conflicting rows
    return_summary=False       # set True to get a DataFrame summary back
):
    total = len(df)
    summary_rows = []

    for rec_col in df.columns:
        if not rec_col.startswith("RECORD_NO__"):
            continue

        ns = rec_col.split("__")[-1]
        sys_col = f"SYSTEM_OF_RECORD__{ns}"
        if sys_col not in df.columns:
            print(f"\n[CHECKING] {rec_col} vs {master_rec} (SKIPPED: missing {sys_col})")
            continue

        print(f"\n[CHECKING] {rec_col} (+ {sys_col})  vs  {master_rec} (+ {master_sys})")

        rec1, rec2 = df[rec_col], df[master_rec]
        sys1, sys2 = df[sys_col], df[master_sys]

        # Raw equality allowing NaN==NaN for the "raw" view
        rec_eq_raw = rec1.fillna("___NA___").eq(rec2.fillna("___NA___"))
        sys_eq_raw = sys1.fillna("___NA___").eq(sys2.fillna("___NA___"))
        raw_matches_mask = rec_eq_raw & sys_eq_raw
        raw_match_count = int(raw_matches_mask.sum())

        # Matches caused by both fields being NaN on both sides
        nan_match_mask = rec1.isna() & rec2.isna() & sys1.isna() & sys2.isna()
        nan_match_count = int(nan_match_mask.sum())

        # Both fields present on both sides
        rec_both_present = (~rec1.isna()) & (~rec2.isna())
        sys_both_present = (~sys1.isna()) & (~sys2.isna())
        both_present = rec_both_present & sys_both_present

        # True matches: equal on both fields and both present
        true_match_mask = both_present & rec1.eq(rec2) & sys1.eq(sys2)
        true_match_count = int(true_match_mask.sum())

        # Non-matches are complement of raw matches
        non_match_mask = ~raw_matches_mask
        non_match_count = int(non_match_mask.sum())

        # Of the non-matches, split into (a) one-sided NaN vs (b) value differences
        one_sided_nan_mask = non_match_mask & (
            (rec1.isna() ^ rec2.isna()) | (sys1.isna() ^ sys2.isna())
        )
        one_sided_nan_count = int(one_sided_nan_mask.sum())

        # Value differences: both fields present on both sides, but at least one differs
        value_diff_conflict_mask = both_present & ~(rec1.eq(rec2) & sys1.eq(sys2))
        value_diff_nonmatch_count = int(value_diff_conflict_mask.sum())

        # Print with percentages
        def pct(x): return f"{(x/total):.2%}"
        print(f"  Total rows:                         {total}")
        print(f"  Raw matches (incl. NaN==NaN):       {raw_match_count} ({pct(raw_match_count)})")
        print(f"  Matches due to both NaNs:           {nan_match_count} ({pct(nan_match_count)})")
        print(f"  True matches (rec+sys, non-NaN):    {true_match_count} ({pct(true_match_count)})")
        print(f"  True non-matches:                   {non_match_count} ({pct(non_match_count)})")
        print(f"  Non-matches w/ one-sided NaN:       {one_sided_nan_count} ({pct(one_sided_nan_count)})")
        print(f"  Non-matches due to value diff:      {value_diff_nonmatch_count} ({pct(value_diff_nonmatch_count)})")

        # Optionally show a few true conflict rows
        if show_conflicts and value_diff_nonmatch_count > 0:
            cols_to_show = [rec_col, master_rec, sys_col, master_sys]
            print("\n  Example conflicts (both present, values differ):")
            print(df.loc[value_diff_conflict_mask, cols_to_show].head(show_conflicts).to_string(index=False))

        # Collect a summary row for optional return
        summary_rows.append({
            "file_ns": ns,
            "record_col": rec_col,
            "system_col": sys_col,
            "total": total,
            "raw_matches": raw_match_count,
            "raw_matches_pct": raw_match_count / total,
            "nan_matches": nan_match_count,
            "nan_matches_pct": nan_match_count / total,
            "true_matches": true_match_count,
            "true_matches_pct": true_match_count / total,
            "true_nonmatches": non_match_count,
            "true_nonmatches_pct": non_match_count / total,
            "one_sided_nan_nonmatches": one_sided_nan_count,
            "one_sided_nan_nonmatches_pct": one_sided_nan_count / total,
            "value_diff_nonmatches": value_diff_nonmatch_count,
            "value_diff_nonmatches_pct": value_diff_nonmatch_count / total,
        })

    if return_summary:
        import pandas as pd
        return pd.DataFrame(summary_rows)


def merge_on_column(df1, df2, df1_col, df2_col, how='left', suffixes=('', '__R')):
    """
    Merge df2 into df1 by aligning df2[df1_col] to df1[df2_col].
    Ensures both join columns are comparable by normalizing to pandas StringDtype,
    removing benign '.0' tails that come from float upcasts, and stripping spaces.
    """
    # Work on copies to avoid mutating callers
    df1 = df1.copy()
    df2_renamed = df2.rename(columns={df1_col: df2_col}).copy()

    # Guard: avoid one->many explosions from duplicate keys on the right
    if df2_col in df2_renamed.columns:
        df2_renamed = df2_renamed.drop_duplicates(subset=[df2_col])

    # --- Normalize join keys on BOTH sides to the same string form ---
    # Convert to pandas StringDtype, strip whitespace, and trim trailing ".0"
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

    # Optional: quick debug if you ever need it
    # print("[DEBUG] dtypes:", df1[df2_col].dtype, df2_renamed[df2_col].dtype)

    merged = pd.merge(df1, df2_renamed, on=df2_col, how=how, suffixes=suffixes)

    # If df1_col also exists post-merge (because of renaming), drop the duplicate
    merged = merged.drop(columns=[c for c in [df1_col] if c in merged.columns], errors='ignore')
    return merged




def load_data_general(main_file, main_record, files, file_records):
    main_df = pd.read_csv(main_file, low_memory = False)
    main_record_mutated = f"{main_record}_{MUTATED}"
    main_df = attach_system_to_record(main_df, main_record, SYS_RECORD_FIELD)

    for (path, record) in zip(files, file_records):
        main_df = merge_file_to_main(main_df, main_record_mutated, path, record )
        print_df(main_df, main_record_mutated, main_file)
    
    return main_df



def load_all_data():
    loss_files = ["data/DIM_CONSOLIDATED_ACCIDENTS.csv", "data/DIM_CONSOLIDATED_HAZARD_OBSERVATIONS.csv", "data/DIM_CONSOLIDATED_NEAR_MISSES.csv" ]
    loss_main_record = "RECORD_NO_LOSS_POTENTIAL"
    loss_records = [loss_main_record, loss_main_record, loss_main_record]
    loss_df = load_data_general("data/DIM_CONSOLIDATED_LOSS_POTENTIAL.csv", loss_main_record, loss_files, loss_records)
    #count_system_aware_matches(loss_df, show_conflicts=5)


    # Concatonate All Action Plans together 
    action_files = ["data/DIM_CONSOLIDATED_CAPA_ACTION_PLAN.csv"]
    action_main_record = "ACTION_NO"
    action_records = [action_main_record]
    action_df = load_data_general("data/DIM_CONSOLIDATED_ACTION_PLAN.csv", action_main_record, action_files, action_records)
    #count_common_columns(loss_df, action_df)
    

    # Merge Action To Loss Df (give action columns a clear suffix)
    loss_df = merge_on_column(
        loss_df, action_df,
        df1_col="RECORD_NO", df2_col="RECORD_NO_MASTER",
        how="left", suffixes=('', '__ACTION')
    )


    #Concactonate Remainder of files On Main 
    rem_files = ["data/DIM_CONSOLIDATED_LEADING_INDICATORS.csv", "data/DIM_CONSOLIDATED_PROPERTY_DAMAGE_INCIDENT.csv"]
    rem_main_record = "MASTER_RECORD_NO"
    rem_records = ["RECORD_NO", rem_main_record]
    rem_df = load_data_general("data/DIM_CONSOLIDATED_INJURY_ILLNESS.csv", rem_main_record, rem_files, rem_records)


    #print_df(loss_df, "RECORD_NO_MASTER", "BEFORE")
    loss_df = merge_on_column(
        loss_df, rem_df,
        df1_col="MASTER_RECORD_NO_MUTATED", df2_col="RECORD_NO_MASTER",
        how="left", suffixes=('', '__REM')
    )

    print_df(loss_df, "RECORD_NO_MASTER", "AFTER")


        
    

    return loss_df





if __name__== "__main__":

    load_all_data()
    # Conclusion, we can connect most files through loss potential, but we need a way to consolodite those files that don't have this field
    # Plan of Action: Create three df's and consolidate them
    #   1. df that consolidates all files that have references to loss_potential
    #   2. df that consolidates all action planns (action_id can be tagged in capa file)
    #   3. df that consolidtes master record (this should consolidate all files + the 2 previous dfs)



    # To Consolidate Record_NO in Files, we must check matches and discrepencies 