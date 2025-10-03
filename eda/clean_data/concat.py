
import pandas as pd
import os 
import clean_data.utils as utils 
import clean_data.coalesce as coalesce
import clean_data.equal as equal


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


def subdf_to_maindf(main_df, main_record, sub_df, sub_record, file_path, safe):
    ns = os.path.splitext(os.path.basename(file_path))[0].replace("DIM_CONSOLIDATED_", "")

    merged = coalesce.merge_df_to_main(
        main_df, main_record, sub_df,
        overwrite_record=sub_record,
        to_drop=[],
        file_path=file_path,                # <<< ensures columns are named __{ns}
    )

    for col in safe:
        src = f"{col}__{ns}"
        if src in merged.columns and col in merged.columns:
            fills = (merged[col].isna() & merged[src].notna()).sum()
            print(f"[COALESCE] {col} <= {src}  filled={fills}")
            merged = coalesce.coalesce_into(merged, dest_col=col, src_col=src, create_conflict_flag=True)
            merged.drop(columns=[src], inplace=True, errors="ignore")
        else:
            print(f"[SKIP] {col}: dest? {col in merged.columns}  src? {src in merged.columns}")

    # optional: drop right join key to keep table tidy
    # merged.drop(columns=[f"{sub_record}_MUTATED"], inplace=True, errors="ignore")
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


# This will be less error-prone, df each file and merge each df respectively to main 
def load_all_data_v2():

    # Set the Main DF 
    main_file = "data/DIM_CONSOLIDATED_LOSS_POTENTIAL.csv"
    loss_column_match_record = "RECORD_NO_LOSS_POTENTIAL" 
    main_df = pd.read_csv(main_file, low_memory = False)
    main_record = f"{loss_column_match_record}_{MUTATED}"
    main_df = utils.attach_system_to_record(main_df, loss_column_match_record, SYS_RECORD_FIELD, drop_sys= True)

    #Attach Consolidated Accidents First
    sub_file = "data/DIM_CONSOLIDATED_ACCIDENTS.csv"
    print(f"Processing File : {sub_file}")
    acc_df = pd.read_csv(sub_file, low_memory = False)
    matching_cols = get_common_columns(main_df, acc_df)
    print_df(main_df, main_record, "BEFORE")
    if len(matching_cols) == 0:
        main_df = coalesce.merge_df_to_main(main_df, main_record, acc_df, overwrite_record=loss_column_match_record)
    else:
        main_df = coalesce.consolidated_matching_cols(main_df, acc_df)
    print_df(main_df, main_record, "After")


    # Attach Hazard File 
    sub_file = "data/DIM_CONSOLIDATED_HAZARD_OBSERVATIONS.csv"
    ns = os.path.splitext(os.path.basename(sub_file))[0].replace("DIM_CONSOLIDATED_", "")
    print(f"Processing File : {ns}")
    hazard_df = pd.read_csv(sub_file, low_memory= False)
    common = get_common_columns(main_df, hazard_df)
    safe, review, avoid = equal.propose_coalesce_columns(main_df, hazard_df, common)
    print("\nSAFE to coalesce:", safe)
    print("REVIEW first:", review)
    print("AVOID:", avoid)
    main_df = subdf_to_maindf(main_df, main_record, hazard_df, loss_column_match_record, sub_file, safe)
    print_df(main_df, main_record, "After Hazard")


    # Attach Near Misses
    sub_file = "data/DIM_CONSOLIDATED_NEAR_MISSES.csv" 
    ns = os.path.splitext(os.path.basename(sub_file))[0].replace("DIM_CONSOLIDATED_", "")
    print(f"Processing File : {ns}")
    misses_df = pd.read_csv(sub_file, low_memory= False)
    common = get_common_columns(main_df, misses_df)
    safe, review, avoid = equal.propose_coalesce_columns(main_df, misses_df, common)
    print("\nSAFE to coalesce:", safe)
    print("REVIEW first:", review)
    print("AVOID:", avoid)
    main_df = subdf_to_maindf(main_df, main_record, misses_df, loss_column_match_record, sub_file, safe)
    print_df(main_df, main_record, "After Near Misses")



    #Attach 


    return main_df



# To Be called after so that we have a file with only the description column
def get_description_df(df):

    # find_col(df, "description")


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