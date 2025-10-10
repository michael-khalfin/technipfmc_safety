import pandas as pd


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


def find_common_cols(df1: pd.DataFrame, df2:pd.DataFrame, show_column_details = True, show_values: int = 2):
    common_cols = df1.columns.intersection(df2.columns)
    print(f"    Shared columns ({len(common_cols)}): {list(common_cols)}\n")

    if show_column_details:
        for col in common_cols:
            print(f"\tColumn: {col}")

            # Unique values in df1
            unique_df1 = df1[col].dropna().unique()
            print(f"\t  df1: Unique count: {len(unique_df1)}")
            if show_values: print(f"\t  df1: Sample values: {unique_df1[:show_values]}")

            # Unique values in df2
            unique_df2 = df2[col].dropna().unique()
            print(f"\t  df2: Unique count: {len(unique_df2)}")
            if show_values: print(f"\t  df2: Sample values: {unique_df2[:show_values]}\n")
    return len(common_cols)