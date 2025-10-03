import pandas as pd



SYS_RECORD_FIELD = "SYSTEM_OF_RECORD"
MUTATED = "MUTATED"
NO = "NO"
UNKNOWN = "unknown"


def _normalize_series(s: pd.Series) -> pd.Series:
    # light normalization for robust equality checks
    s = s.astype("string")
    s = s.str.strip()
    # drop ".0", ".00", etc. common after CSV/Excel round-trips
    s = s.str.replace(r"\.0+$", "", regex=True)
    # uniform case for text-like codes
    s = s.str.lower()
    return s


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


def check_duplicates(main_df, remove= False):
    dups = main_df.columns[main_df.columns.duplicated(keep=False)]
    if len(dups) > 0:
        print("[WARN] Duplicated columns found:")
        for col in dups.unique():
            positions = [i for i, c in enumerate(main_df.columns) if c == col]
            print(f"  {col}: positions {positions}")

def namespace_subcolumns(sub_df, keep_cols,  ns):
    keep = set(keep_cols)
    newcols = {
        c: (c if c in keep else f"{c}__{ns}")
        for c in sub_df.columns
    }
    return sub_df.rename(columns=newcols)


def attach_system_to_record(df, record_col, system_col, drop_sys = False):
    # Use pandas StringDtype / category instead of Python object strings
    df[system_col] = df[system_col].fillna(UNKNOWN).astype('category')
    df[record_col] = df[record_col].astype('string')
    
    # Create New Col
    df[record_col + f"_{MUTATED}"] = df[record_col] + "_" + df[system_col].astype('string')
    
    #Drop Record and Sys cols
    dropped_cols = [record_col]
    if drop_sys: dropped_cols.append(system_col)

    df = df.drop(columns=dropped_cols)
    return df



def columns_equivalence_report(df: pd.DataFrame, col_a: str, col_b: str, show_conflicts: int = 5):
    if col_a not in df.columns or col_b not in df.columns:
        raise KeyError(f"Missing columns: {col_a if col_a not in df.columns else ''} {col_b if col_b not in df.columns else ''}")

    a_raw, b_raw = df[col_a], df[col_b]
    a = _normalize_series(a_raw)
    b = _normalize_series(b_raw)

    # Treat NaN==NaN as "equal" for the high-level “same-ness” question
    both_nan = a_raw.isna() & b_raw.isna()
    match_mask = (a.fillna("__NA__") == b.fillna("__NA__"))
    raw_equal_count = int(match_mask.sum())
    both_nan_count = int(both_nan.sum())

    # Where both present, check "true equal"
    both_present = ~a_raw.isna() & ~b_raw.isna()
    both_present_count = int(both_present.sum())
    true_equal_mask = both_present & (a == b)
    true_equal_count = int(true_equal_mask.sum())

    # Non-matches
    non_match_mask = ~match_mask
    one_sided_nan = non_match_mask & ((a_raw.isna() ^ b_raw.isna()))
    value_conflict = non_match_mask & both_present

    total = len(df)
    pct = lambda x: f"{(x/total):.2%}"

    print(f"\n[COMPARE] {col_a} vs {col_b}")
    print(f"  Total rows:                    {total}")
    print(f"  Equal (incl. NaN==NaN):        {raw_equal_count} ({pct(raw_equal_count)})")
    print(f"   └─ due to both NaN:           {both_nan_count} ({pct(both_nan_count)})")
    print(f"  Equal where both present:      {true_equal_count}/{both_present_count} "
          f"({(true_equal_count/max(both_present_count,1)):.2%} of both-present)")
    print(f"  One-sided NaN:                 {int(one_sided_nan.sum())} ({pct(int(one_sided_nan.sum()))})")
    print(f"  Value conflicts (both present):{int(value_conflict.sum())} ({pct(int(value_conflict.sum()))})")

    if show_conflicts and value_conflict.any():
        cols = [col_a, col_b]
        print("\n  Example value conflicts:")
        print(df.loc[value_conflict, cols].head(show_conflicts).to_string(index=False))

    # Useful return bundle
    return {
        "equal_mask": match_mask,
        "both_nan_mask": both_nan,
        "both_present_mask": both_present,
        "true_equal_mask": true_equal_mask,
        "one_sided_nan_mask": one_sided_nan,
        "value_conflict_mask": value_conflict,
        "equal_rate_including_nans": raw_equal_count / total if total else 1.0,
        "equal_rate_where_both_present": (true_equal_count / both_present_count) if both_present_count else 1.0,
    }

def columns_essentially_equal(
    df: pd.DataFrame, col_a: str, col_b: str,
    tol_all: float = 0.995,      # overall equality including NaN==NaN
    tol_both_present: float = 0.99  # equality where both are non-null
) -> bool:
    r = columns_equivalence_report(df, col_a, col_b, show_conflicts=0)
    return (r["equal_rate_including_nans"] >= tol_all) and (r["equal_rate_where_both_present"] >= tol_both_present)

