import pandas as pd
from collections import Counter
from typing import Dict

def _normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    s = s.str.lower()
    return s

def _value_multiset(s: pd.Series) -> Counter:
    # normalize, drop NaNs, count frequencies
    s = _normalize_series(s)
    s = s.dropna()
    return Counter(s.tolist())

def columns_value_equivalence_report(
    df_left: pd.DataFrame, col_left: str,
    df_right: pd.DataFrame, col_right: str,
    show_examples: int = 5
) -> Dict[str, object]:
    """
    Compare two columns *as multisets of values* (order-agnostic, no join key).
    Good for deciding if two columns across files are essentially the same field
    despite different row orders and sparsity.
    """
    if col_left not in df_left.columns:
        raise KeyError(f"Left missing column: {col_left}")
    if col_right not in df_right.columns:
        raise KeyError(f"Right missing column: {col_right}")

    L = _value_multiset(df_left[col_left])
    R = _value_multiset(df_right[col_right])

    # Basic counts
    nL = sum(L.values())
    nR = sum(R.values())

    # Intersection/union on *multisets*:
    # multiset intersection count = sum(min(freq_L, freq_R))
    # multiset union count       = sum(max(freq_L, freq_R))
    keys = set(L) | set(R)
    inter = sum(min(L[k], R[k]) for k in keys)
    union = sum(max(L[k], R[k]) for k in keys) if keys else 0

    # Jaccard on multisets; handle empty union
    jaccard = (inter / union) if union else 1.0

    # Coverage (how much of each side is explained by the other)
    coverage_L_in_R = (inter / nL) if nL else 1.0
    coverage_R_in_L = (inter / nR) if nR else 1.0

    # Symmetric difference diagnostics (by frequency difference)
    left_extra = Counter({k: max(L[k] - R.get(k, 0), 0) for k in L if L[k] > R.get(k, 0)})
    right_extra = Counter({k: max(R[k] - L.get(k, 0), 0) for k in R if R[k] > L.get(k, 0)})

    # Examples of disagreements (most frequent offenders)
    left_examples = left_extra.most_common(show_examples)
    right_examples = right_extra.most_common(show_examples)

    # Unique counts (distinct values)
    distinct_L = len(L)
    distinct_R = len(R)
    distinct_overlap = len(set(L) & set(R))

    report = {
        "left_col": col_left,
        "right_col": col_right,
        "left_non_null_count": nL,
        "right_non_null_count": nR,
        "distinct_left": distinct_L,
        "distinct_right": distinct_R,
        "distinct_overlap": distinct_overlap,
        "multiset_intersection": inter,
        "multiset_union": union,
        "jaccard_multiset": jaccard,                # 1.0 means identical multiset
        "coverage_left_in_right": coverage_L_in_R,  # % of left values explained by right
        "coverage_right_in_left": coverage_R_in_L,  # % of right values explained by left
        "top_left_only_examples": left_examples,    # [(value, extra_count), ...]
        "top_right_only_examples": right_examples,
    }

    # Friendly printout (optional)
    print(f"\n[CONTENT EQUIVALENCE] {col_left} (left) vs {col_right} (right)")
    print(f"  Non-null counts: L={nL}, R={nR}")
    print(f"  Distinct values: L={distinct_L}, R={distinct_R}, overlap={distinct_overlap}")
    print(f"  Multiset Jaccard: {jaccard:.4f}")
    print(f"  Coverage L in R:  {coverage_L_in_R:.4%}   | Coverage R in L: {coverage_R_in_L:.4%}")

    if left_examples:
        print("  Left extras (value, extra_count):", left_examples)
    if right_examples:
        print("  Right extras (value, extra_count):", right_examples)

    return report

def columns_essentially_equal_by_values(
    df_left: pd.DataFrame, col_left: str,
    df_right: pd.DataFrame, col_right: str,
    jaccard_tol: float = 0.995,
    coverage_tol: float = 0.99
) -> bool:
    """
    Boolean helper over the report. Tunable thresholds.
    """
    rep = columns_value_equivalence_report(df_left, col_left, df_right, col_right, show_examples=0)
    return (
        rep["jaccard_multiset"] >= jaccard_tol and
        rep["coverage_left_in_right"] >= coverage_tol and
        rep["coverage_right_in_left"] >= coverage_tol
    )


def propose_coalesce_columns(main_df, sub_df, common_cols,
                             min_coverage=0.95, review_band=(0.85, 0.95),
                             max_distinct=64):
    """
    Returns (safe, review, avoid) lists based on columns_value_equivalence_report.
    Uses coverage L in R as the primary signal. Limits to modest cardinality by default.
    """
    from collections import defaultdict
    safe, review, avoid = [], [], []
    for col in sorted(common_cols):
        rep = columns_value_equivalence_report(main_df, col, sub_df, col, show_examples=0)

        # basic guards
        if col in {"DATE_TIME_OF_INCIDENT"}:
            avoid.append(col); continue

        # cap cardinality for automatic coalesce
        if rep["distinct_left"] > max_distinct or rep["distinct_right"] > max_distinct:
            # allow dates through by name heuristic
            if not any(tok in col.lower() for tok in ["date","time","approval","closure","due","reported","incident"]):
                review.append(col); continue

        cov = rep["coverage_left_in_right"]
        if cov >= min_coverage:
            safe.append(col)
        elif review_band[0] <= cov < review_band[1]:
            review.append(col)
        else:
            avoid.append(col)


        cov2 = rep["coverage_right_in_left"]
        if cov2 >= min_coverage:
            safe.append(col)
        elif review_band[0] <= cov2 < review_band[1]:
            review.append(col)
        else:
            avoid.append(col)

    return list(set(safe)), list(set(review)), list(set(avoid))



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
