# -*- coding: utf-8 -*-
"""
EDA on the Consolidated Cleaned Dataset Before Textual Mapping

  - Missingness by column/row
  - Co-missingness (phi between NaN indicators) for top-30 sparse columns
  - Coverage by org/system (non-null %)
  - Category top values
  - Text quick peeks
"""
# Import modules

import argparse
import os
import re
import math
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Command Line Interface
def get_args():
    p = argparse.ArgumentParser(
        description="Co-missingness-focused EDA (CSV or Excel)."
    )
    # Positional (required, order matters):
    p.add_argument("input_path", help="Path to data (.csv or .xlsx)")
    p.add_argument("out_dir",    help="Directory to write EDA outputs")

    # Optional flags:
    p.add_argument("--sheet", type=str, default=None,
                   help="Excel sheet name (ignored for CSV). Use None to use the first sheet.")
    p.add_argument("--topk_cats", type=int, default=30,
                   help="Top K categories for bar charts (default: 30)")
    return p.parse_args()

# Helper Functions
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def save_txt(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))

def coerce_bool(s: pd.Series) -> pd.Series:
    # Already boolean dtype
    if pd.api.types.is_bool_dtype(s):
        return s

    # Pure numeric dtype: treat 1→True, 0→False, else NaN
    if pd.api.types.is_numeric_dtype(s):
        return s.map(lambda x: True if x == 1 else (False if x == 0 else np.nan))

    # Mixed/object/string: normalize per-cell, robust to NA and mixed types
    yes = {"y","yes","true","t","1"}
    no  = {"n","no","false","f","0"}

    def to_bool(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip().lower()
        if x in yes: return True
        if x in no:  return False
        return np.nan

    return s.map(to_bool)

def is_textlike(s: pd.Series) -> bool:
    if s.dtype != object:
        return False
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    avg_len = sample.str.len().mean()
    digit_ratio = sample.apply(lambda x: sum(ch.isdigit() for ch in x)/max(1,len(x))).mean()
    return (avg_len >= 20) and (digit_ratio < 0.35)

def phi_nan(x: pd.Series, y: pd.Series) -> float:
    # Phi correlation between NaN indicators (missingness co-occurrence)
    a = x.isna().astype(int)
    b = y.isna().astype(int)
    n11 = int(((a==1) & (b==1)).sum())
    n10 = int(((a==1) & (b==0)).sum())
    n01 = int(((a==0) & (b==1)).sum())
    n00 = int(((a==0) & (b==0)).sum())
    n = n11 + n10 + n01 + n00
    denom = math.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    if n == 0 or denom == 0:
        return np.nan
    return (n11*n00 - n10*n01) / denom

# Main Function
def main():
    args = get_args()

    # Use the correct arg names and expand ~
    input_path = os.path.expanduser(args.input_path)
    OUT = ensure_dir(os.path.expanduser(args.out_dir))

    charts = ensure_dir(os.path.join(OUT, "charts"))
    tables = ensure_dir(os.path.join(OUT, "tables"))
    logs = ensure_dir(os.path.join(OUT, "logs"))

    # Load CSV or Excel
    ext = os.path.splitext(input_path)[1].lower()
    if ext in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(
            input_path,
            sheet_name=None if args.sheet in (None, "None") else args.sheet
        )
        # If multiple sheets returned and no sheet specified, take the first
        if isinstance(df, dict):
            df = df[next(iter(df))]
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file type: {ext} (expected .csv or .xlsx)")

    df.columns = [c.strip() for c in df.columns]
    save_txt(os.path.join(logs, "shape.txt"), f"Shape: {df.shape}")

    # Boolean normalization for known flags (optional)
    for c in [x for x in ["WORK_RELATED","TFMC_OWNED","NOTIFICATION_SENT","SIF_PREVENTION","STOPPED_WORK"] if x in df.columns]:
        df[c] = coerce_bool(df[c])

    # Boolean normalization for known flags (optional)
    for c in [x for x in ["WORK_RELATED","TFMC_OWNED","NOTIFICATION_SENT","SIF_PREVENTION","STOPPED_WORK"] if x in df.columns]:
        df[c] = coerce_bool(df[c])

    from pandas.api.types import (
        is_numeric_dtype, is_bool_dtype, is_object_dtype,
        is_datetime64_any_dtype, is_categorical_dtype
    )

    def feature_kind(s: pd.Series) -> str:
        if is_bool_dtype(s): return "boolean"
        if is_numeric_dtype(s): return "numeric"
        if is_categorical_dtype(s): return "string"
        if is_object_dtype(s): return "string"
        if is_datetime64_any_dtype(s): return "datetime"
        return "other"

    ft = pd.Series({c: feature_kind(df[c]) for c in df.columns}, name="feature_type")
    ft.to_frame().to_csv(os.path.join(tables, "00_feature_types.csv"))

    focused = ft.value_counts().reindex(["string", "numeric", "boolean"], fill_value=0)
    plt.figure(figsize=(6,4))
    plt.bar(focused.index, focused.values)
    plt.title("Feature counts: string vs numeric vs boolean")
    plt.ylabel("# of columns")
    plt.tight_layout()
    plt.savefig(os.path.join(charts, "00_feature_type_counts.png"))
    plt.close()

    # Buckets
    bool_cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    obj_cols  = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
    text_cols = [c for c in obj_cols if is_textlike(df[c])]
    cat_cols  = list(dict.fromkeys(obj_cols + bool_cols)) 

    # Missingness
    missing = df.isna().sum().sort_values(ascending=False).rename("missing_count")
    miss_pct = (missing/len(df)*100).round(2).rename("missing_pct")
    miss_df = pd.concat([missing, miss_pct], axis=1)
    miss_df.to_csv(os.path.join(tables, "10_missing_by_column.csv"))

    row_miss = df.isna().sum(axis=1)
    row_miss.to_frame("n_missing").to_csv(os.path.join(tables, "11_missing_by_row.csv"), index=False)
    plt.figure(figsize=(6,4))
    plt.hist(row_miss.values, bins=30)
    plt.title("Missing count per row")
    plt.xlabel("# missing fields"); plt.ylabel("rows")
    plt.tight_layout()
    plt.savefig(os.path.join(charts, "11_missing_by_row.png"))
    plt.close()

    # Co-missingness (phi of NaN) among top-30 sparse columns
    top_miss_cols = list(miss_df.sort_values("missing_pct", ascending=False).head(30).index)
    phi_records = []
    for i in range(len(top_miss_cols)):
        for j in range(i+1, len(top_miss_cols)):
            a, b = top_miss_cols[i], top_miss_cols[j]
            phi = phi_nan(df[a], df[b])
            phi_records.append({"col1": a, "col2": b, "phi_nan": phi})
    pd.DataFrame(phi_records).sort_values("phi_nan", ascending=False)\
      .to_csv(os.path.join(tables, "12_comissing_phi.csv"), index=False)

    # Coverage by group
    group_cols = [c for c in ["ORGANIZATION","GBU","BU","SYSTEM_OF_RECORD","INCIDENT_TYPE"] if c in df.columns]
    key_fields = [c for c in ["RISK_COLOR","MITIGATED_RISK","LOSS_POTENTIAL_SEVERITY","DATE_OF_CLOSURE","DUE_DATE","PERSON_RESPONSIBLE","WORKPLACE"] if c in df.columns]

    for g in group_cols:
        grp = df.groupby(g)[key_fields].apply(lambda x: x.notna().mean()*100)
        grp.columns = [f"{c}_coverage_pct" for c in grp.columns]
        grp.reset_index().to_csv(os.path.join(tables, f"13_coverage_by_{g}.csv"), index=False)

    # Quick chart for avg coverage (top 15 groups) using first grouping if available
    if group_cols:
        g = group_cols[0]
        grp = pd.read_csv(os.path.join(tables, f"13_coverage_by_{g}.csv"))
        cov_cols = [c for c in grp.columns if c.endswith("_coverage_pct")]
        if cov_cols:
            frame = grp.set_index(g)[cov_cols]
            top_groups = df[g].value_counts().head(15).index
            frame_small = frame.loc[frame.index.isin(top_groups)]
            if not frame_small.empty:
                avg_cov = frame_small.mean(axis=1).sort_values(ascending=False)
                plt.figure(figsize=(10,4))
                plt.bar(range(len(avg_cov)), avg_cov.values)
                plt.xticks(range(len(avg_cov)), avg_cov.index.astype(str), rotation=90)
                plt.ylabel("Avg field coverage (%)")
                plt.title(f"Average coverage of key fields by {g} (top 15)")
                plt.tight_layout()
                plt.savefig(os.path.join(charts, f"13_coverage_avg_by_{g}.png"))
                plt.close()

    # Category profiles
    for c in cat_cols:
        vc = df[c].astype(str).fillna("NA").value_counts()
        vc.head(args.topk_cats).to_csv(os.path.join(tables, f"50_top_values_{c}.csv"))
        top = vc.head(min(args.topk_cats, 25))
        plt.figure(figsize=(10, 0.35*len(top)+2))
        plt.barh(top.index[::-1], top.values[::-1])
        plt.title(f"Top {len(top)} values: {c}")
        plt.tight_layout()
        plt.savefig(os.path.join(charts, f"50_top_values_{c}.png"))
        plt.close()

    # Text quick peeks
    for c in text_cols:
        s = df[c].dropna().astype(str)
        lens = s.str.len()
        lens.describe().to_csv(os.path.join(tables, f"60_text_length_{c}.csv"))
        plt.figure(figsize=(6,4))
        plt.hist(lens.values, bins=30)
        plt.title(f"Text length: {c}")
        plt.xlabel("chars"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(charts, f"60_text_length_{c}.png"))
        plt.close()

        sample = s.sample(min(3000, len(s)), random_state=0)
        tokens = []
        for x in sample:
            toks = [t for t in re.split(r"[^A-Za-z]+", x.lower()) if t]
            tokens.extend(toks)
        uni = Counter(tokens).most_common(100)
        pd.DataFrame(uni, columns=["token","count"]).to_csv(os.path.join(tables, f"61_text_top_unigrams_{c}.csv"), index=False)

    with open(os.path.join(logs, "info.txt"), "w", encoding="utf-8") as f:
        df.info(buf=f)
    df.head(15).to_csv(os.path.join(tables, "01_head.csv"), index=False)
    df.tail(15).to_csv(os.path.join(tables, "02_tail.csv"), index=False)

    print(f"Done. Outputs in: {OUT}")

if __name__ == "__main__":
    main()
