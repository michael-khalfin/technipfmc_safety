import pandas as pd
from collections import Counter
from typing import Optional, List, Dict, Tuple

def _normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    s = s.str.lower()
    return s

def _value_multiset(s: pd.Series) -> Counter:
    s = _normalize_series(s)
    s = s.dropna()
    return Counter(s.tolist())

def _pick_merged_col(aligned: pd.DataFrame, base: str, prefer: str) -> str:
    if base in aligned.columns:
        return base
    cand = f"{base}_{prefer}"
    if cand in aligned.columns:
        return cand
    alt = f"{base}_{'R' if prefer=='L' else 'L'}"
    if alt in aligned.columns:
        return alt
    raise KeyError(f"Column not found after merge: {base} (tried {cand} / {alt})")


def _ensure_mutated_key(
    df: pd.DataFrame,
    want_key_name: str,
    record_col: str = None,
    system_col: str = "SYSTEM_OF_RECORD",
    unknown: str = "unknown",
) -> tuple[pd.DataFrame, str]:
    """
    Return (df_view, key_name). If want_key_name doesn't exist,
    synthesize it as f"{record_col}_{system_col}" (with string dtypes).
    Does NOT mutate the original df.
    """
    if want_key_name in df.columns:
        return df, want_key_name
    if record_col is None:
        raise ValueError(
            f"Right key '{want_key_name}' not found and record_col=None; "
            f"provide record_col to synthesize."
        )
    tmp = df.copy()
    if system_col not in tmp.columns:
        raise KeyError(f"Right df missing '{system_col}' needed to synthesize '{want_key_name}'")

    tmp[system_col] = tmp[system_col].fillna(unknown).astype("string")
    tmp[record_col] = tmp[record_col].astype("string")
    tmp[want_key_name] = tmp[record_col].str.strip() + "_" + tmp[system_col].astype("string")
    return tmp, want_key_name


def columns_value_equivalence_report(
    df_left: pd.DataFrame, col_left: str,
    df_right: pd.DataFrame, col_right: str,
    show_examples: int = 0
) -> Dict[str, object]:
    if col_left not in df_left.columns:
        raise KeyError(f"Left missing column: {col_left}")
    if col_right not in df_right.columns:
        raise KeyError(f"Right missing column: {col_right}")

    L = _value_multiset(df_left[col_left]); nL = sum(L.values())
    R = _value_multiset(df_right[col_right]); nR = sum(R.values())

    keys = set(L) | set(R)
    inter = sum(min(L[k], R[k]) for k in keys)
    union = sum(max(L[k], R[k]) for k in keys) if keys else 0
    jaccard = (inter / union) if union else 1.0
    coverage_L_in_R = (inter / nL) if nL else 1.0
    coverage_R_in_L = (inter / nR) if nR else 1.0

    distinct_L, distinct_R = len(L), len(R)
    distinct_overlap = len(set(L) & set(R))

    left_extra = Counter({k: max(L[k] - R.get(k, 0), 0) for k in L if L[k] > R.get(k, 0)})
    right_extra = Counter({k: max(R[k] - L.get(k, 0), 0) for k in R if R[k] > L.get(k, 0)})
    left_examples = left_extra.most_common(show_examples)
    right_examples = right_extra.most_common(show_examples)

    return {
        "left_col": col_left, "right_col": col_right,
        "left_non_null_count": nL, "right_non_null_count": nR,
        "distinct_left": distinct_L, "distinct_right": distinct_R,
        "distinct_overlap": distinct_overlap,
        "multiset_intersection": inter, "multiset_union": union,
        "jaccard_multiset": jaccard,
        "coverage_left_in_right": coverage_L_in_R,
        "coverage_right_in_left": coverage_R_in_L,
        "top_left_only_examples": left_examples,
        "top_right_only_examples": right_examples,
    }

def _dedupe_on_key(df: pd.DataFrame, key: str, keep: str = "first") -> pd.DataFrame:
    if key not in df.columns:
        raise KeyError(f"Missing key column: {key}")
    return df.drop_duplicates(subset=[key], keep=keep)

def key_column_match_report(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    key_left: str,
    key_right: str = None,
    col_left: str = "",
    col_right: str = None,
    how: str = "inner",
    right_dedupe_keep: str = "first",
    normalize: bool = True,
    # NEW: allow building right key on the fly
    right_record_col: str = None,
    right_system_col: str = "SYSTEM_OF_RECORD",
) -> Dict[str, object]:
    key_right = key_right or key_left
    col_right = col_right or col_left

    if key_left not in left_df.columns:
        raise KeyError(f"Left missing key: {key_left}")
    if col_left not in left_df.columns:
        raise KeyError(f"Left missing column: {col_left}")
    if col_right not in right_df.columns:
        raise KeyError(f"Right missing column: {col_right}")

    # Ensure the right key exists (build it if needed)
    R_view, key_r = _ensure_mutated_key(
        right_df, want_key_name=key_right,
        record_col=right_record_col,
        system_col=right_system_col
    )
    if col_right not in R_view.columns:
        raise KeyError(f"Right missing column after key synthesis: {col_right}")

    # Deduplicate right on the synthetic/real key
    R = R_view[[key_r, col_right]].drop_duplicates(subset=[key_r], keep=right_dedupe_keep).copy()
    L = left_df[[key_left, col_left]].copy()

    aligned = pd.merge(L, R, how=how, left_on=key_left, right_on=key_r, suffixes=("_L", "_R"))

    left_name  = _pick_merged_col(aligned, col_left,  prefer="L")
    right_name = _pick_merged_col(aligned, col_right, prefer="R")

    a_raw = aligned[left_name]
    b_raw = aligned[right_name]
    a = _normalize_series(a_raw) if normalize else a_raw
    b = _normalize_series(b_raw) if normalize else b_raw

    matched_rows = len(aligned)
    both_nan      = a_raw.isna() & b_raw.isna()
    both_present  = a_raw.notna() & b_raw.notna()
    exact_equal   = both_present & (a == b)
    value_conflict= both_present & (a != b)
    one_nan_left  = a_raw.isna() & b_raw.notna()
    one_nan_right = a_raw.notna() & b_raw.isna()

    cnt_both_nan = int(both_nan.sum())
    cnt_exact    = int(exact_equal.sum())
    cnt_conflict = int(value_conflict.sum())
    cnt_lnan     = int(one_nan_left.sum())
    cnt_rnan     = int(one_nan_right.sum())
    cnt_both_pres= int(both_present.sum())

    def rate(x, d): return (x / d) if d else 1.0
    rates = {
        "equal_including_nan": rate(cnt_exact + cnt_both_nan, matched_rows),
        "equal_non_null":      rate(cnt_exact, cnt_both_pres),
        "conflict_rate":       rate(cnt_conflict, matched_rows),
        "one_nan_left_rate":   rate(cnt_lnan, matched_rows),
        "one_nan_right_rate":  rate(cnt_rnan, matched_rows),
    }

    return {
        "matched_rows": matched_rows,
        "counts": {
            "both_present": cnt_both_pres,
            "exact_equal": cnt_exact,
            "both_nan": cnt_both_nan,
            "one_nan_left": cnt_lnan,
            "one_nan_right": cnt_rnan,
            "value_conflict": cnt_conflict,
        },
        "rates": rates,
    }


def propose_coalesce_with_reports(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    key_left: str,
    cols: Optional[List[str]] = None,
    # thresholds...
    min_coverage: float = 0.95,
    review_band: Tuple[float, float] = (0.85, 0.95),
    max_distinct_auto: int = 64,
    min_equal_non_null: float = 0.95,
    max_conflict_rate: float = 0.02,
    how: str = "inner",
    right_dedupe_keep: str = "first",
    # right key synthesis
    key_right: str  = None,
    right_record_col: str  = None,
    right_system_col: str = "SYSTEM_OF_RECORD",
    # NEW knobs:
    allow_high_card_if_exact: bool = True,
    exact_equal_thresh: float = 0.999,    # promote if >=
    exact_conflict_ceiling: float = 1e-6, # and <=
    id_like_tokens: Tuple[str, ...] = ("record","id","key","action","master"),
    force_allow_ids: bool = True,
    # handling sparse overlaps
    min_both_present_rows: int = 10,
    min_fill_rate_for_safe: float = 0.10,   # ≥10% of rows get filled from right
) -> Dict[str, object]:
    key_right = key_right or key_left
    if cols is None:
        cols = sorted((set(left_df.columns) & set(right_df.columns)) - {key_left, key_right})

    out_per_col: Dict[str, Dict[str, object]] = {}
    safe, review, avoid = [], [], []

    for col in cols:
        # quick ID-name guard (unless explicitly overridden)
        name_hits_id = any(tok in col.lower() for tok in id_like_tokens)
        if name_hits_id and not force_allow_ids:
            rep_val = columns_value_equivalence_report(left_df, col, right_df, col, show_examples=0)
            rep_key = key_column_match_report(
                left_df, right_df, key_left=key_left, key_right=key_right,
                col_left=col, col_right=col, how=how, right_dedupe_keep=right_dedupe_keep,
                right_record_col=right_record_col, right_system_col=right_system_col
            )
            out_per_col[col] = {
                "value_report": rep_val, "key_report": rep_key, "decision": "avoid",
                "metrics": {
                    "coverage_L_in_R": rep_val["coverage_left_in_right"],
                    "coverage_R_in_L": rep_val["coverage_right_in_left"],
                    "jaccard_multiset": rep_val["jaccard_multiset"],
                    "equal_non_null":   rep_key["rates"].get("equal_non_null"),
                    "conflict_rate":    rep_key["rates"].get("conflict_rate") or 0.0,
                    "distinct_left":    rep_val["distinct_left"],
                    "distinct_right":   rep_val["distinct_right"],
                }
            }
            avoid.append(col)
            continue

        # 1) value-multiset view
        rep_val = columns_value_equivalence_report(left_df, col, right_df, col, show_examples=0)
        covL, covR = rep_val["coverage_left_in_right"], rep_val["coverage_right_in_left"]
        jacc      = rep_val["jaccard_multiset"]

        # 2) key-aligned view
        rep_key = key_column_match_report(
            left_df, right_df,
            key_left=key_left, key_right=key_right,
            col_left=col, col_right=col,
            how=how, right_dedupe_keep=right_dedupe_keep,
            right_record_col=right_record_col, right_system_col=right_system_col
        )

        rates  = rep_key["rates"]
        counts = rep_key["counts"]
        eq_nonnull = rates.get("equal_non_null")          # may be None if denom=0
        conflict   = rates.get("conflict_rate") or 0.0
        one_nan_L  = rates.get("one_nan_left_rate") or 0.0  # left is NaN, right has value
        both_pres  = counts.get("both_present") or 0

        # heuristics
        small_card = (
            rep_val["distinct_left"]  <= max_distinct_auto and
            rep_val["distinct_right"] <= max_distinct_auto
        ) or any(t in col.lower() for t in ["date","time","approval","closure","due","reported","incident"])

        # ----- decision logic -----
        decision = "avoid"

        # A) “Classic equality” path only when we have enough both-present rows
        if both_pres >= min_both_present_rows and eq_nonnull is not None:
            # High-cardinality exact promotion
            if allow_high_card_if_exact and (eq_nonnull >= exact_equal_thresh) and (conflict <= exact_conflict_ceiling):
                decision = "safe"
            # Standard SAFE
            elif small_card and (covL >= min_coverage or covR >= min_coverage) and \
                 (eq_nonnull >= min_equal_non_null) and (conflict <= max_conflict_rate):
                decision = "safe"
            # REVIEW band
            elif (covL >= review_band[0] or covR >= review_band[0]) and (eq_nonnull >= 0.80):
                decision = "review"
            else:
                decision = "avoid"

        # B) Fill-only path: no row has both values present on L & R
        elif both_pres == 0:
            # If we can fill a meaningful portion of blanks from right and conflicts are impossible by definition
            if one_nan_L >= min_fill_rate_for_safe and conflict == 0.0:
                decision = "safe"   # SAFE (fill-only)
            else:
                decision = "review" # uncertain payoff or too little to fill

        # C) Few both-present rows but not zero (sparse). Be conservative.
        else:
            if (covL >= review_band[0] or covR >= review_band[0]) and (conflict <= max_conflict_rate):
                decision = "review"
            else:
                decision = "avoid"

        out_per_col[col] = {
            "value_report": rep_val,
            "key_report": rep_key,
            "decision": decision,
            "metrics": {
                "coverage_L_in_R": covL,
                "coverage_R_in_L": covR,
                "jaccard_multiset": jacc,
                "equal_non_null": eq_nonnull,
                "conflict_rate": conflict,
                "distinct_left": rep_val["distinct_left"],
                "distinct_right": rep_val["distinct_right"],
                "both_present": both_pres,
                "one_nan_left_rate": one_nan_L,
            }
        }

        if decision == "safe" and both_pres == 0 and one_nan_L >= min_fill_rate_for_safe:
            out_per_col[col]["note"] = "fill-only"

        if decision == "safe":
            safe.append(col)
        elif decision == "review":
            review.append(col)
        else:
            avoid.append(col)

    return {
        "safe":   sorted(set(safe)),
        "review": sorted(set(review) - set(safe)),
        "avoid":  sorted(set(avoid) - set(safe) - set(review)),
        "per_column": out_per_col
    }
