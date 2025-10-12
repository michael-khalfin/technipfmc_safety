
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


# Helper to Attach System of Record as Suffix to Desired Record_Col
def attach_system_to_record(df, record_col, system_col):
    df[system_col] = df[system_col].fillna(UNKNOWN).astype(str)
    df[record_col] = df[record_col].astype(str)
    
    df[record_col + f"_{MUTATED}"] = df[record_col] + "_" + df[system_col]
    return df


# Called by Merge_File_to_main to consilidate sub_df into main_df
# def merge_df_to_main(main_df, main_record, sub_df, sub_record, to_drop = None):
#     #Drop Requested Columns
#     if to_drop is not None:
#         sub_df = sub_df.drop(columns=to_drop, errors="ignore")

#     main_df[main_record] = main_df[main_record].astype(str)
#     sub_df[sub_record] = sub_df[sub_record].astype(str)

#     # Inner merge on record number (with system tag)
#     merged_df = pd.merge(main_df, sub_df, how='left', left_on=main_record, right_on=sub_record, suffixes=('', '_sub'))
#     check_duplicates(merged_df)
#     return merged_df


def namespace_subcolumns(sub_df, keep_cols,  ns):
    keep = set(keep_cols)
    newcols = {
        c: (c if c in keep else f"{c}__{ns}")
        for c in sub_df.columns
    }
    return sub_df.rename(columns=newcols)


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

    # Drop System Field 
    drop_cols = set(to_drop or [])
    drop_cols.update([sub_system_field, sub_record])  # keep mutated join key only
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
        f"{record}_MOD" : target_df[record + f"_{MUTATED}"].notna().sum()
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



def load_data():
    # Main File Should Be Accident Plans or Loss Potential
    main_df = pd.read_csv(MAIN_FILE, low_memory = False)
    #main_record = find_col(main_df, NO) # Can Subsittiate with RECORD_NO_LOSS_POTENTIAL instead of Master
    main_record = find_col(main_df, "RECORD_NO_LOSS_POTENTIAL") # we connect through this, might need to connect with master later on 
    main_record_mutated = f"{main_record}_{MUTATED}"

    # Attach System To Record 
    main_df = attach_system_to_record(main_df, main_record, SYS_RECORD_FIELD)
    #print(df[[main_record, SYS_RECORD_FIELD, main_record + "_MUTATED"]].head())
    print_df(main_df, main_record, "MAIN")


    # 1. Accidents (has RECORD_NO_LOSS_POTENTIAL, does this differ from master or?)
    main_df = merge_file_to_main(main_df, main_record_mutated, "data/DIM_CONSOLIDATED_ACCIDENTS.csv", overwrite_record= "RECORD_NO_LOSS_POTENTIAL")
    print_df(main_df, main_record, "MAIN")


    # 2. Hazard Observations (has RECORD_NO_LOSS_POTENTIAL, does this differ from master or?)
    main_df = merge_file_to_main(main_df, main_record_mutated, "data/DIM_CONSOLIDATED_HAZARD_OBSERVATIONS.csv", overwrite_record= "RECORD_NO_LOSS_POTENTIAL")
    print_df(main_df, main_record, "MAIN")

    # 3. Injury\Illness (Has MASTER_RECORD_NO, will need to merged based on this)
    # main_df = merge_file_to_main(main_df, main_record_mutated, "data/DIM_CONSOLIDATED_INJURY_ILLNESS.csv", overwrite_record="MASTER_RECORD_NO")
    # print_df(main_df, main_record, "MAIN")

    # 4. Leading Indicators
    # main_df = merge_file_to_main(main_df, main_record_mutated, "data/DIM_CONSOLIDATED_LEADING_INDICATORS.csv")
    # print_df(main_df, main_record, "MAIN")

    # 5. Near Misses (has RECORD_NO_LOSS_POTENTIAL, does this differ from master or?)
    main_df = merge_file_to_main(main_df, main_record_mutated, "data/DIM_CONSOLIDATED_NEAR_MISSES.csv", overwrite_record= "RECORD_NO_LOSS_POTENTIAL")
    print_df(main_df, main_record, "MAIN")

    # 6. Property Damage 
    #main_df = merge_file_to_main(main_df, main_record_mutated, "data/DIM_CONSOLIDATED_PROPERTY_DAMAGE_INCIDENT.csv", overwrite_record= "MASTER_RECORD_NO")
    #print_df(main_df, main_record, "MAIN")

    # 7. Action Plan (can connect with CAPA Action Plan through action No)


    # See if we Have Duplicats
    check_duplicates(main_df)
    

    return main_df


if __name__== "__main__":

    #This is just a draft
    load_data()

    # Conclusion, we can connect most files through loss potential, but we need a way to consolodite those files that don't have this field
    # Plan of Action: Create three df's and consolidate them
    #   1. df that consolidates all files that have references to loss_potential
    #   2. df that consolidates all action planns (action_id can be tagged in capa file)
    #   3. df that consolidtes master record (this should consolidate all files + the 2 previous dfs)