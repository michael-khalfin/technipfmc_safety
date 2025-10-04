def _pct(x: float) -> str:
    try:
        return f"{100.0 * x:6.2f}%"
    except Exception:
        return "   n/a "

def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "n/a"

def pretty_print_coalesce_column(col: str, payload: dict) -> None:
    """Print one column's metrics from propose_coalesce_with_reports(...)."""
    decision = payload.get("decision", "?").upper()
    met = payload.get("metrics", {}) or {}
    vr  = payload.get("value_report", {}) or {}
    kr  = payload.get("key_report", {}) or {}

    # value-multiset
    jacc   = met.get("jaccard_multiset", 0.0)
    covLR  = met.get("coverage_L_in_R", 0.0)
    covRL  = met.get("coverage_R_in_L", 0.0)
    dL     = met.get("distinct_left", 0)
    dR     = met.get("distinct_right", 0)
    dOv    = vr.get("distinct_overlap", 0)

    # key-aligned
    rates  = kr.get("rates", {}) or {}
    counts = kr.get("counts", {}) or {}
    matched   = kr.get("matched_rows", 0)
    eq_nonnull= rates.get("equal_non_null", 0.0)
    eq_inc_na = rates.get("equal_including_nan", 0.0)
    conflict  = rates.get("conflict_rate", 0.0)
    one_nan_L = rates.get("one_nan_left_rate", 0.0)
    one_nan_R = rates.get("one_nan_right_rate", 0.0)
    both_pres = counts.get("both_present", 0)
    exact_eq  = counts.get("exact_equal", 0)
    both_nan  = counts.get("both_nan", 0)

    print(f"\n=== {col}  →  DECISION: {decision} ===")
    print("Value distribution (multiset, order-agnostic):")
    print(f"  Jaccard: {_pct(jacc)} | Coverage L→R: {_pct(covLR)} | R→L: {_pct(covRL)}")
    print(f"  Distinct L/R/overlap: {_fmt_int(dL)} / {_fmt_int(dR)} / {_fmt_int(dOv)}")

    print("Key-aligned (row-wise where keys match):")
    print(f"  Matched rows: {_fmt_int(matched)} | Both-present: {_fmt_int(both_pres)}")
    print(f"  Equal (non-null): {_fmt_int(exact_eq)} / {_fmt_int(both_pres)} ({_pct(eq_nonnull)})")
    print(f"  Equal (incl. NaN==NaN): {_pct(eq_inc_na)} | Conflicts: {_pct(conflict)}")
    print(f"  One-sided NaN  L: {_pct(one_nan_L)}  |  R: {_pct(one_nan_R)}")

def pretty_print_coalesce_report(report: dict, order=("safe","review","avoid")) -> None:
    """
    Pretty print the whole result from propose_coalesce_with_reports(...).
    Groups by decision in the given order.
    """
    per = report.get("per_column", {}) or {}
    # group columns by decision
    buckets = {k: [] for k in order}
    for col, payload in per.items():
        buckets.setdefault(payload.get("decision","other"), []).append((col, payload))

    # print a quick header
    print("\n######## Coalesce Proposal Summary ########")
    for k in order:
        cols = sorted([c for c,_ in buckets.get(k, [])])
        print(f"  {k.upper():<6}: {len(cols)}  -> {', '.join(cols[:12])}{' ...' if len(cols)>12 else ''}")
    others = [c for c,_ in buckets.get("other",[])]
    if others:
        print(f"  OTHER : {len(others)} -> {', '.join(sorted(others))}")

    # detailed per-column blocks
    for k in order:
        for col, payload in sorted(buckets.get(k, []), key=lambda x: x[0].lower()):
            pretty_print_coalesce_column(col, payload)
