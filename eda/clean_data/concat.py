
import pandas as pd
import os 
import utils
import coalesce
import equal


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


def get_common_columns(df1, df2):
    common_cols = set(df1.columns).intersection(set(df2.columns))
    if len(common_cols) == 0:
        print("No matching Cols")
    else:
        print(f"Common columns: {common_cols}")
    return common_cols


def merge_file_to_main(main_df, main_record, file_path, overwrite_record = None, overwrite_system_field = None, to_drop = None):
    print(f"Processing {file_path}")

    sub_df = pd.read_csv(file_path, low_memory = False)
    return coalesce.merge_df_to_main(main_df, main_record, sub_df, overwrite_record, overwrite_system_field, to_drop, file_path)
    

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


def load_data_general(main_file, main_record, files, file_records):
    main_df = pd.read_csv(main_file, low_memory = False)
    main_record_mutated = f"{main_record}_{MUTATED}"
    main_df = utils.attach_system_to_record(main_df, main_record, SYS_RECORD_FIELD)

    for (path, record) in zip(files, file_records):
        main_df = merge_file_to_main(main_df, main_record_mutated, path, record )
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


# This will be less error-prone, df each file and merge each df respectively to main 
def load_all_data_v2():

    # Set the Main DF 
    main_file = "data/DIM_CONSOLIDATED_LOSS_POTENTIAL.csv"
    column_match_record = "RECORD_NO_LOSS_POTENTIAL" 
    main_df = pd.read_csv(main_file, low_memory = False)
    main_record = f"{column_match_record}_{MUTATED}"
    main_df = utils.attach_system_to_record(main_df, column_match_record, SYS_RECORD_FIELD, drop_sys= True)

    #Attach Consolidated Accidents First
    sub_file = "data/DIM_CONSOLIDATED_ACCIDENTS.csv"
    print(f"Processing File : {sub_file}")
    acc_df = pd.read_csv(sub_file, low_memory = False)
    matching_cols = get_common_columns(main_df, acc_df)
    print_df(main_df, main_record, "BEFORE")
    if len(matching_cols) == 0:
        main_df = coalesce.merge_df_to_main(main_df, main_record, acc_df, overwrite_record="RECORD_NO_LOSS_POTENTIAL")
    else:
        main_df = coalesce.consolidated_matching_cols(main_df, acc_df)
    print_df(main_df, main_record, "After")


    # Attach Hazard File 
    sub_file = "data/DIM_CONSOLIDATED_HAZARD_OBSERVATIONS.csv"
    print(f"Processing File : {sub_file}")
    hazard_df = pd.read_csv(sub_file, low_memory= False)
    matching_cols = get_common_columns(main_df, hazard_df)




# To Be called after so that we have a file with only the description column
def get_description_df(df):

    # find_col(df, "description")


    return df






if __name__== "__main__":

    all_data_df = load_all_data_v1()
    get_description_df(all_data_df)


    # Conclusion, we can connect most files through loss potential, but we need a way to consolodite those files that don't have this field
    # Plan of Action: Create three df's and consolidate them
    #   1. df that consolidates all files that have references to loss_potential
    #   2. df that consolidates all action planns (action_id can be tagged in capa file)
    #   3. df that consolidtes master record (this should consolidate all files + the 2 previous dfs)



    # To Consolidate Record_NO in Files, we must check matches and discrepencies 