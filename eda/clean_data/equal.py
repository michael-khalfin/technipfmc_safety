import pandas as pd
import utils




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
