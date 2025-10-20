import pandas as pd
import json, hashlib, re
from pathlib import Path

# ---------- Configuration ----------
src_path = "dev_sample.csv"          # your input CSV
out_path = "dev_sample_json.csv"  # output CSV

# ---------- Helpers ----------
def normalize_key(k: str) -> str:
    k = k.strip()
    k = re.sub(r"\s+", "_", k)
    k = re.sub(r"[^A-Za-z0-9_:/\-]", "", k)
    return k.lower()

def make_id(props, label_field=None):
    """Stable ID using hash of key fields or first few non-empty props."""
    id_basis_keys = []
    if label_field and props.get(label_field):
        id_basis_keys.append(label_field)
    if "id" in props and props["id"]:
        if "id" not in id_basis_keys:
            id_basis_keys.append("id")
    if not id_basis_keys:
        id_basis_keys = sorted(list(props.keys()))[:6]
    id_basis = "|".join(f"{k}={props.get(k,'')}" for k in id_basis_keys)
    return "urn:row:" + hashlib.sha1(id_basis.encode("utf-8")).hexdigest()[:16]

# ---------- Load CSV ----------
df = pd.read_csv(src_path, dtype=str, keep_default_na=False).fillna("")

# Normalize column names
col_map = {c: normalize_key(c) for c in df.columns}
df = df.rename(columns=col_map)

# Try to find a label column
label_candidates = [c for c in df.columns if c in {"name", "title", "id", "label"}]
label_field = label_candidates[0] if label_candidates else None

# ---------- Build node_json ----------
rows = []
for i, row in df.iterrows():
    props = {k: v.strip() for k, v in row.items() if str(v).strip() != ""}
    if not props:
        continue
    node_id = make_id(props, label_field)
    payload = {
        "_id": node_id,
        "_source_file": Path(src_path).name,
        "_row_index": int(i),
        "properties": props,
    }
    if label_field and props.get(label_field):
        payload["@label"] = props[label_field]
    rows.append({"_id": node_id, "node_json": json.dumps(payload, ensure_ascii=False)})

# ---------- Save ----------
out_df = pd.DataFrame(rows)
out_df.to_csv(out_path, index=False)
print(f"✅ Saved {len(out_df)} rows to {out_path}")
