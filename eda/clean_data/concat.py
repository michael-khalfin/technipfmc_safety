
import pandas as pd
import os 
import clean_data.utils as utils 
import clean_data.coalesce as coalesce
import clean_data.equal as equal
import clean_data.prettyprint as prettyprint


SYS_RECORD_FIELD = "SYSTEM_OF_RECORD"
MUTATED, NO = "MUTATED", "NO"
UNKNOWN = "unknown"


#Helper to Find Desired Column Based On keyword
def get_common_columns(df1, df2):
    common_cols = set(df1.columns).intersection(set(df2.columns))
    if len(common_cols) == 0:
        print("No matching Cols")
    else:
        print(f"Common columns: {common_cols}")
    return common_cols  


def subdf_to_maindf(main_df, main_record, sub_df, sub_record, file_path, safe, drop_coalesced=True):
    """
    Merge a sub dataframe to main and coalesce common columns.
    
    Args:
        drop_coalesced: If True, drop the source columns after coalescing.
                       If False, keep them for reference.
    """
    ns = os.path.splitext(os.path.basename(file_path))[0].replace("DIM_CONSOLIDATED_", "")

    merged = coalesce.merge_df_to_main(
        main_df, main_record, sub_df,
        overwrite_record=sub_record,
        to_drop=[],
        file_path=file_path,              
    )

    for col in safe:
        src = f"{col}__{ns}"
        if src in merged.columns and col in merged.columns:
            fills = (merged[col].isna() & merged[src].notna()).sum()
            print(f"[COALESCE] {col} <= {src}  filled={fills}")
            merged = coalesce.coalesce_into(merged, dest_col=col, src_col=src)
            if drop_coalesced:
                merged.drop(columns=[src], inplace=True, errors="ignore")
        else:
            print(f"[SKIP] {col}: dest? {col in merged.columns}  src? {src in merged.columns}")

    # optional: drop right join key to keep table tidy (but never drop the main key)
    right_key = f"{sub_record}_{MUTATED}"
    if (right_key in merged.columns) and (right_key != main_record):
        merged.drop(columns=[right_key], inplace=True, errors="ignore")
    return merged



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


    summary_df = pd.DataFrame(summary)
    print(f"\nDataset Summary For {name}:")
    print(summary_df.to_string(index=False))
    print('\n')



def aggregate_conflicts(df, flag_col="HAS_CONFLICT", drop_conflict_cols=True):
    conflict_cols = [c for c in df.columns if c.endswith("__CONFLICT")]
    if conflict_cols:
        df[flag_col] = df[conflict_cols].any(axis=1)
        if drop_conflict_cols:
            df = df.drop(columns=conflict_cols)
    else:
        if flag_col not in df.columns:
            df[flag_col] = False
    num_conflict_rows = (df["HAS_CONFLICT"] != False).sum()
    print(f"Number of conflict rows: {num_conflict_rows}")
    return df


def load_data_general(main_file, main_record, files, file_records):
    main_df = pd.read_csv(main_file, low_memory = False)
    main_record_mutated = f"{main_record}_{MUTATED}"
    main_df = utils.attach_system_to_record(main_df, main_record, SYS_RECORD_FIELD)

    for (path, record) in zip(files, file_records):
        main_df = coalesce.merge_file_to_main(main_df, main_record_mutated, path, record )
        print_df(main_df, main_record_mutated, main_file)
    
    return main_df



def load_all_data_v1():
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
    loss_df = coalesce.merge_on_column(
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
    loss_df = coalesce.merge_on_column(
        loss_df, rem_df,
        df1_col="MASTER_RECORD_NO_MUTATED", df2_col="RECORD_NO_MASTER",
        how="left", suffixes=('', '__REM')
    )

    print_df(loss_df, "RECORD_NO_MASTER", "AFTER")

    return loss_df



def _attach_loss_potential_files(main_df, loss_column_match_record):
    #Attach Consolidated Accidents First
    sub_file = "data/DIM_CONSOLIDATED_ACCIDENTS.csv"
    main_record = f"{loss_column_match_record}_{MUTATED}"
    ns = os.path.splitext(os.path.basename(sub_file))[0].replace("DIM_CONSOLIDATED_", "")
    print(f"Processing File : {ns}")
    acc_df = pd.read_csv(sub_file, low_memory = False)
    matching_cols = get_common_columns(main_df, acc_df)
    if len(matching_cols) == 0:
        main_df = coalesce.merge_df_to_main(main_df, main_record, acc_df, overwrite_record=loss_column_match_record)
    else:
        main_df = coalesce.consolidated_matching_cols(main_df, acc_df)


    # Attach Hazard File 
    sub_file = "data/DIM_CONSOLIDATED_HAZARD_OBSERVATIONS.csv"
    ns = os.path.splitext(os.path.basename(sub_file))[0].replace("DIM_CONSOLIDATED_", "")
    print(f"Processing File : {ns}")
    hazard_df = pd.read_csv(sub_file, low_memory= False)
    res = equal.propose_coalesce_with_reports(
        left_df=main_df,
        right_df=hazard_df,
        key_left=main_record,                  
        # let the function synthesize the right key with these:
        key_right=None,                        # same name as left
        right_record_col=loss_column_match_record,  # 'RECORD_NO_LOSS_POTENTIAL'
        right_system_col=SYS_RECORD_FIELD           # 'SYSTEM_OF_RECORD'
    )
    prettyprint.pretty_print_coalesce_report(res)
    safe = res["safe"]
    print("SAFE:", safe)
    print("REVIEW:", res["review"])
    print("AVOID:", res["avoid"])
    main_df = subdf_to_maindf(main_df, main_record, hazard_df, loss_column_match_record, sub_file, safe)


    # Attach Near Misses
    sub_file = "data/DIM_CONSOLIDATED_NEAR_MISSES.csv" 
    ns = os.path.splitext(os.path.basename(sub_file))[0].replace("DIM_CONSOLIDATED_", "")
    print(f"Processing File : {ns}")
    misses_df = pd.read_csv(sub_file, low_memory= False)
    res = equal.propose_coalesce_with_reports(
        left_df=main_df,
        right_df=misses_df,
        key_left=main_record,                  # 'RECORD_NO_LOSS_POTENTIAL_MUTATED' (exists on left)
        # let the function synthesize the right key with these:
        key_right=None,                        # same name as left
        right_record_col=loss_column_match_record,  # 'RECORD_NO_LOSS_POTENTIAL'
        right_system_col=SYS_RECORD_FIELD           # 'SYSTEM_OF_RECORD'
    )
    # prettyprint.pretty_print_coalesce_report(res)
    safe = res["safe"]
    print("SAFE:", safe)
    print("REVIEW:", res["review"])
    print("AVOID:", res["avoid"])
    main_df = subdf_to_maindf(main_df, main_record, misses_df, loss_column_match_record, sub_file, safe)
    


    return main_df


def _aggregate_actions_per_record(action_df, record_col="RECORD_NO", keep_all_columns=False):
    print(f"\n[AGGREGATING ACTIONS] Reducing from {len(action_df):,} to {action_df[record_col].nunique():,} rows")
    
    if keep_all_columns:
        # Keep ALL columns - for each one, take first value
        print(f"  [MODE] Keeping all {len(action_df.columns)} columns (using 'first' aggregation)")
        
        agg_dict = {}
        
        for col in action_df.columns:
            if col != record_col:
                agg_dict[col] = 'first'
        
        aggregated = action_df.groupby(record_col, as_index=False).agg(agg_dict)
        
        # Add action count as a new column
        action_counts = action_df.groupby(record_col).size().reset_index(name='ACTION_COUNT')
        aggregated = aggregated.merge(action_counts, on=record_col, how='left')
    else:
        # Original minimal aggregation - only keep key columns
        agg_dict = {}
        
        # Count column
        count_col = None
        for col in ['ACTION_NO', 'STATUS', record_col]:
            if col in action_df.columns and col != record_col:
                count_col = col
                break
        
        if count_col:
            agg_dict[count_col] = 'count'
        
        # Important columns to keep
        important_cols = {
            'STATUS': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'SYSTEM_OF_RECORD': 'first',
            'INCIDENT_TYPE': 'first',
            'PERSON_RESPONSIBLE_NAME': 'first',
            'PERSON_RESPONSIBLE_EMAIL': 'first',
            'ACTION_TYPE': 'first',
            'ACTION_REQUIRED': 'first',
            'TARGET_DATE': 'first',
            'COMPLETION_DATE': 'first',
            'LOCATION_CODE': 'first',
        }
        
        for col, agg_func in important_cols.items():
            if col in action_df.columns:
                agg_dict[col] = agg_func
        
        aggregated = action_df.groupby(record_col, as_index=False).agg(agg_dict)
        
        # Rename count column to ACTION_COUNT
        if count_col:
            aggregated.rename(columns={count_col: 'ACTION_COUNT'}, inplace=True)
    
    print(f"[AGGREGATED] Rows: {len(aggregated):,}, Cols: {len(aggregated.columns)}")
    return aggregated


def _attach_action_files(main_df, main_record, aggregate=True, keep_all_action_columns=True):
    master_column_match_record = f"{main_record}_{MUTATED}"
    main_df = utils.attach_system_to_record(main_df, main_record, SYS_RECORD_FIELD, drop_sys=False)
    
    
    action_file = "data/DIM_CONSOLIDATED_ACTION_PLAN.csv"
    capa_file = "data/DIM_CONSOLIDATED_CAPA_ACTION_PLAN.csv"
    
    print(f"\n Loading ACTION_PLAN...")
    action_df = pd.read_csv(action_file, low_memory=False)
    print(f"  Rows: {len(action_df):,}, Cols: {len(action_df.columns)}")
    print(f"  Unique RECORD_NO: {action_df['RECORD_NO'].nunique():,}")
    print(f"  Unique ACTION_NO: {action_df['ACTION_NO'].nunique():,}")
    
    print(f"\n Loading CAPA_ACTION_PLAN...")
    capa_df = pd.read_csv(capa_file, low_memory=False)
    print(f"  Rows: {len(capa_df):,}, Cols: {len(capa_df.columns)}")
    print(f"  Unique ACTION_NO: {capa_df['ACTION_NO'].nunique():,}")
    

    print(f"\n Merging CAPA into ACTION on ACTION_NO...")
    action_df['SYS_FOR_RECORD'] = action_df[SYS_RECORD_FIELD].copy()
    
    # Create mutated keys for the ACTION_NO merge
    action_df = utils.attach_system_to_record(action_df.copy(), "ACTION_NO", SYS_RECORD_FIELD, drop_sys=True)
    capa_df = utils.attach_system_to_record(capa_df.copy(), "ACTION_NO", SYS_RECORD_FIELD, drop_sys=True)
    
    # Merge on ACTION_NO_MUTATED
    combined_actions = pd.merge(
        action_df, capa_df,
        on="ACTION_NO_MUTATED",
        how="left",
        suffixes=('', '__CAPA')
    )
    print(f"  Combined: {len(combined_actions):,} rows, {len(combined_actions.columns)} cols")
    
    # Drop ACTION_NO_MUTATED and restore SYSTEM_OF_RECORD for RECORD_NO
    combined_actions.drop(columns=["ACTION_NO_MUTATED"], inplace=True, errors="ignore")
    combined_actions[SYS_RECORD_FIELD] = combined_actions['SYS_FOR_RECORD']
    combined_actions.drop(columns=['SYS_FOR_RECORD'], inplace=True, errors="ignore")
    
    # Coalesce CAPA columns with ACTION columns
    print(f"\n  Coalescing CAPA columns with ACTION columns...")
    capa_cols = [c for c in combined_actions.columns if c.endswith('__CAPA')]
    coalesced_count = 0
    
    for capa_col in capa_cols:
        # Get the base column name
        base_col = capa_col.replace('__CAPA', '')
        
        if base_col in combined_actions.columns:
            # Count how many values will be filled
            fills = (combined_actions[base_col].isna() & combined_actions[capa_col].notna()).sum()
            if fills > 0:
                print(f"    [COALESCE] {base_col} <- {capa_col}: filled {fills} values")
                combined_actions = coalesce.coalesce_into(combined_actions, dest_col=base_col, src_col=capa_col)
                coalesced_count += 1
            # Drop the CAPA column after coalescing
            combined_actions.drop(columns=[capa_col], inplace=True, errors='ignore')
        # If base column doesn't exist, keep the CAPA column but rename it
        else:
            combined_actions.rename(columns={capa_col: base_col}, inplace=True)
            print(f"    [RENAME] {capa_col} -> {base_col} (base didn't exist)")
    
    print(f"  Coalesced {coalesced_count} columns, dropped {len(capa_cols) - coalesced_count} CAPA-only columns")
    print(f"  After coalescing: {len(combined_actions.columns)} columns")
    
    # Aggregate by RECORD_NO if needed
    avg_actions = len(combined_actions) / combined_actions['RECORD_NO'].nunique()
    print(f"  Actions per RECORD_NO (avg): {avg_actions:.2f}")
    print(f"  Combined actions has {len(combined_actions.columns)} columns")
    
    if aggregate and avg_actions > 1.5:
        print(f"\n[4/4] Aggregating to 1 row per RECORD_NO")
        if keep_all_action_columns:
            print(f"  [INFO] Keeping ALL {len(combined_actions.columns)} columns (taking first value per record)")
        else:
            print(f"  [WARNING] Only keeping ~10 important columns. Set keep_all_action_columns=True to keep all.")
        combined_actions = _aggregate_actions_per_record(combined_actions, record_col="RECORD_NO", keep_all_columns=keep_all_action_columns)
    else:
        print(f"\n[4/4] Keeping all action rows (will multiply main_df rows by ~{avg_actions:.1f}x)")
        print(f"  [INFO] All {len(combined_actions.columns)} action columns will be preserved")
    
    cons_action_df = combined_actions

    # Create the mutated key in action dataframe for comparison
    cons_action_df_check = utils.attach_system_to_record(cons_action_df.copy(), "RECORD_NO", SYS_RECORD_FIELD, drop_sys=False)
    action_mutated_key = f"RECORD_NO_{MUTATED}"
    
    main_keys = set(main_df[master_column_match_record].dropna().unique())
    action_keys = set(cons_action_df_check[action_mutated_key].dropna().unique())
    
    overlap = main_keys & action_keys
    
    print(f"\nMerging COMBINED ACTIONS TO MAIN_DF")
    
    res = equal.propose_coalesce_with_reports(
        left_df=main_df,
        right_df=cons_action_df,
        key_left=master_column_match_record,
        key_right=None,
        right_record_col="RECORD_NO",
        right_system_col=SYS_RECORD_FIELD
    )
    safe = res["safe"]
    print("\nSAFE:", safe)
    print("REVIEW:", res["review"])
    print("AVOID:", res["avoid"])
    
    # Use a generic name for the combined actions
    sub_file = "data/DIM_CONSOLIDATED_COMBINED_ACTIONS.csv"
    main_df = subdf_to_maindf(main_df, master_column_match_record, cons_action_df, "RECORD_NO", sub_file, safe)
    
    return main_df


def _attach_remainder_files(main_df, main_record):
    """
    Attach remainder files (INJURY_ILLNESS, LEADING_INDICATORS, PROPERTY_DAMAGE_INCIDENT)
    These files use MASTER_RECORD_NO or RECORD_NO to connect
    """
    master_column_match_record = f"{main_record}_{MUTATED}"
    
    # Ensure we have the mutated version of RECORD_NO_MASTER
    if master_column_match_record not in main_df.columns:
        if main_record in main_df.columns:
            print(f"Creating {master_column_match_record}")
            main_df = utils.attach_system_to_record(main_df, main_record, SYS_RECORD_FIELD, drop_sys=False)
        else:
            raise ValueError(f"{main_record} not found in main_df")
    
    print(f"\n{'='*70}")
    print("COMBINING REMAINDER FILES (INJURY/ILLNESS, LEADING INDICATORS, PROPERTY DAMAGE)")
    print(f"{'='*70}")
    
    # Step 1: Load INJURY_ILLNESS as the base (has MASTER_RECORD_NO)
    injury_file = "data/DIM_CONSOLIDATED_INJURY_ILLNESS.csv"
    print(f"\n[1/4] Loading INJURY_ILLNESS as base...")
    injury_df = pd.read_csv(injury_file, low_memory=False)
    print(f"  Rows: {len(injury_df):,}, Cols: {len(injury_df.columns)}")
    print(f"  Key column: MASTER_RECORD_NO")
    
    # Create mutated key for INJURY_ILLNESS
    injury_df = utils.attach_system_to_record(injury_df, "MASTER_RECORD_NO", SYS_RECORD_FIELD, drop_sys=False)
    injury_record_mutated = "MASTER_RECORD_NO_MUTATED"
    
    # Step 2: Load and merge LEADING_INDICATORS (has RECORD_NO)
    leading_file = "data/DIM_CONSOLIDATED_LEADING_INDICATORS.csv"
    print(f"\n[2/4] Loading LEADING_INDICATORS...")
    leading_df = pd.read_csv(leading_file, low_memory=False)
    print(f"  Rows: {len(leading_df):,}, Cols: {len(leading_df.columns)}")
    print(f"  Key column: RECORD_NO")
    
    # Merge LEADING_INDICATORS into INJURY_ILLNESS
    # LEADING_INDICATORS.RECORD_NO should match INJURY_ILLNESS.MASTER_RECORD_NO
    print(f"  Merging LEADING_INDICATORS to INJURY_ILLNESS...")
    injury_df = coalesce.merge_file_to_main(
        injury_df, injury_record_mutated, leading_file,
        overwrite_record="RECORD_NO",
        to_drop=[]
    )
    print(f"  After merge: {len(injury_df):,} rows, {len(injury_df.columns)} cols")
    
    # Step 3: Load and merge PROPERTY_DAMAGE_INCIDENT (has MASTER_RECORD_NO)
    property_file = "data/DIM_CONSOLIDATED_PROPERTY_DAMAGE_INCIDENT.csv"
    print(f"\n[3/4] Loading PROPERTY_DAMAGE_INCIDENT...")
    property_df = pd.read_csv(property_file, low_memory=False)
    print(f"  Rows: {len(property_df):,}, Cols: {len(property_df.columns)}")
    print(f"  Key column: MASTER_RECORD_NO")
    
    # Merge PROPERTY_DAMAGE into the combined injury_df
    print(f"  Merging PROPERTY_DAMAGE to combined remainder...")
    injury_df = coalesce.merge_file_to_main(
        injury_df, injury_record_mutated, property_file,
        overwrite_record="MASTER_RECORD_NO",
        to_drop=[]
    )
    print(f"  After merge: {len(injury_df):,} rows, {len(injury_df.columns)} cols")
    
    # Check for duplicates and aggregate if needed
    # Use the mutated key that we already created
    group_key = injury_record_mutated  # "MASTER_RECORD_NO_MUTATED"
    
    num_rows = len(injury_df)
    num_unique = injury_df[group_key].nunique()
    avg_per_key = num_rows / num_unique
    
    print(f"\n  Remainder data: {num_rows:,} rows, {num_unique:,} unique keys")
    print(f"  Avg rows per key: {avg_per_key:.2f}")
    
    if avg_per_key > 1.1:
        print(f"  [AGGREGATING] Reducing remainder to 1 row per key to prevent row explosion")
        # Group by the mutated key and take first value for each column
        agg_dict = {col: 'first' for col in injury_df.columns if col != group_key}
        injury_df = injury_df.groupby(group_key, as_index=False).agg(agg_dict)
        print(f"  After aggregation: {len(injury_df):,} rows")
    
    # Step 4: Now merge the combined remainder into main_df
    # injury_df has MASTER_RECORD_NO_MUTATED, but we need to match on RECORD_NO_MASTER in main
    # So we need to rename the join key
    print(f"\n[4/4] Merging combined remainder to main_df...")
    print(f"  Main DF will join on: {master_column_match_record}")
    
    # The injury_df has MASTER_RECORD_NO_MUTATED
    # We need to find if there's a RECORD_NO_MASTER column in injury_df to use
    if "RECORD_NO_MASTER" in injury_df.columns:
        # Create the mutated version for RECORD_NO_MASTER
        injury_df = utils.attach_system_to_record(injury_df, "RECORD_NO_MASTER", SYS_RECORD_FIELD, drop_sys=False)
        remainder_join_key = "RECORD_NO_MASTER_MUTATED"
        print(f"  Using RECORD_NO_MASTER_MUTATED as join key")
    else:
        # Fallback: assume MASTER_RECORD_NO is the same as RECORD_NO_MASTER
        remainder_join_key = injury_record_mutated
        print(f"  Using {remainder_join_key} as join key")
    
    # Check overlap
    main_keys = set(main_df[master_column_match_record].dropna().unique())
    rem_keys = set(injury_df[remainder_join_key].dropna().unique())
    overlap = main_keys & rem_keys
    
    print(f"\n  Main DF unique keys: {len(main_keys):,}")
    print(f"  Remainder DF unique keys: {len(rem_keys):,}")
    print(f"  Overlap: {len(overlap):,} ({len(overlap)/len(main_keys)*100:.1f}% of main)")
    
    # Use the coalesce workflow to properly merge and fill missing values
    print(f"\n  Analyzing columns for safe coalescing...")
    
    # Need to prepare the join key - rename remainder key to match the expected format
    # The subdf_to_maindf expects the sub_df to have a record column that will be mutated
    # Since we already have the mutated key, we need to work around this
    
    # Find the original record column name that was used
    if remainder_join_key == "RECORD_NO_MASTER_MUTATED":
        remainder_record_col = "RECORD_NO_MASTER"
    else:
        remainder_record_col = "MASTER_RECORD_NO"
    
    # Run the coalesce analysis
    # Note: injury_df already has the mutated key, so we pass it directly
    res = equal.propose_coalesce_with_reports(
        left_df=main_df,
        right_df=injury_df,
        key_left=master_column_match_record,
        key_right=remainder_join_key
    )
    
    safe = res["safe"]
    print(f"\n  SAFE to coalesce: {len(safe)} columns")
    print(f"  REVIEW: {len(res['review'])} columns")
    print(f"  AVOID: {len(res['avoid'])} columns")
    
    # Manual merge with coalescing (since injury_df already has mutated key)
    ns = "REMAINDER"  # Namespace for remainder columns
    
    # Namespace the injury_df columns (except the join key)
    injury_df_ns = utils.namespace_subcolumns(injury_df, keep_cols=[remainder_join_key], ns=ns)
    
    # Perform the merge
    merged = pd.merge(
        main_df, injury_df_ns,
        left_on=master_column_match_record,
        right_on=remainder_join_key,
        how='left'
    )
    
    # Coalesce safe columns
    print(f"\n  Coalescing {len(safe)} safe columns...")
    for col in safe:
        src = f"{col}__{ns}"
        if src in merged.columns and col in merged.columns:
            fills = (merged[col].isna() & merged[src].notna()).sum()
            if fills > 0:
                print(f"  [COALESCE] {col} <= {src}  filled={fills}")
                merged = coalesce.coalesce_into(merged, dest_col=col, src_col=src)
            merged.drop(columns=[src], inplace=True, errors="ignore")
    
    # Drop the duplicate join key
    if remainder_join_key in merged.columns and remainder_join_key != master_column_match_record:
        merged.drop(columns=[remainder_join_key], inplace=True, errors='ignore')
    
    main_df = merged
    print(f"\n  [RESULT] After remainder merge: {len(main_df):,} rows, {len(main_df.columns)} cols")
    
    return main_df


# This will be less error-prone, df each file and merge each df respectively to main 
def load_all_data_v2():

    # Set the Main DF 
    main_file = "data/DIM_CONSOLIDATED_LOSS_POTENTIAL.csv"
    loss_column_match_record = "RECORD_NO_LOSS_POTENTIAL" 
    main_df = pd.read_csv(main_file, low_memory = False)
    main_df = utils.attach_system_to_record(main_df, loss_column_match_record, SYS_RECORD_FIELD, drop_sys= False)

    utils.find_col(main_df, "no")

    # Attach loss Potential Files First 
    main_df = _attach_loss_potential_files(main_df, loss_column_match_record)
    print_df(main_df, f"{loss_column_match_record}_{MUTATED}", "After Loss Potential Files")
    
    # Attach action files (ACTION_PLAN + CAPA combined and aggregated)
    main_df = _attach_action_files(main_df, "RECORD_NO_MASTER")
    print_df(main_df, f"{loss_column_match_record}_{MUTATED}", "After Action Files")
    
    # Attach remainder files (INJURY_ILLNESS, LEADING_INDICATORS, PROPERTY_DAMAGE)
    main_df = _attach_remainder_files(main_df, "RECORD_NO_MASTER")
    print_df(main_df, f"{loss_column_match_record}_{MUTATED}", "After Remainder Files")

    main_df = aggregate_conflicts(main_df, flag_col="HAS_CONFLICT", drop_conflict_cols=False)
    utils.find_col(main_df, "system")


    return main_df



# To Be called after so that we have a file with only the description column
def get_description_df(df):
    return df



if __name__== "__main__":

    all_data_df = load_all_data_v2()
    get_description_df(all_data_df)


    # Conclusion, we can connect most files through loss potential, but we need a way to consolodite those files that don't have this field
    # Plan of Action: Create three df's and consolidate them
    #   1. df that consolidates all files that have references to loss_potential
    #   2. df that consolidates all action planns (action_id can be tagged in capa file)
    #   3. df that consolidtes master record (this should consolidate all files + the 2 previous dfs)



    # To Consolidate Record_NO in Files, we must check matches and discrepencies 