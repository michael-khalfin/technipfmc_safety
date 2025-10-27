"""Utility to push CSV rows through a running Plumber instance and persist triples,
then materialize a simple knowledge graph as nodes/edges CSVs."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

# Default Modules 
EXTRACTORS = ["open_ie"]
LINKERS = ["FalconWikidataJoint"]
RESOLVERS = ["spacy_neural_coreference"]
PLUMBER_URL = "http://127.0.0.1:5000"

# Data Cols To Process
TEXT_COLUMN = "text"
ID_COLUMN = "RECORD_NO_LOSS_POTENTIAL"

# Path Configs
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "data" / "cleaned_description_translated.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_JSONL = OUTPUT_DIR / "plumber_triples.jsonl"
NODES_CSV_PATH  = OUTPUT_DIR / "nodes.csv"
EDGES_CSV_PATH = OUTPUT_DIR / "edges.csv"

# Config For Processing Due to Timout Issues and Efficiency
MAX_ROWS = None
SLEEP_SECONDS = 0.0
REQUEST_TIMEOUT = 120.0
LOG_EVERY = 50
MAX_WORKERS = 4
MAX_CHARS = None
RETRIES = 2
RETRY_WAIT = 5.0
START_INDEX = 0


"""
Yield Rows Until We Read Them
"""
def read_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row

"""
Build Neccessary Payload To Send to Plumber API 
"""
def build_payload(
    text: str,
    extractors: List[str],
    linkers: Optional[List[str]],
    resolvers: Optional[List[str]],
) -> Dict:
    payload = {
        "input_text": text,
        "extractor": extractors,
    }
    if linkers is not None:
        payload["linker"] = linkers
    if resolvers is not None:
        payload["resolver"] = resolvers
    return payload


"""
General Submission to Plumber API
"""
def submit_request(
    base_url: str,
    payload: Dict,
    timeout: float,
) -> List[Dict]:
    endpoint = f"{base_url.rstrip('/')}/run"
    response = requests.post(endpoint, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Plumber API returned {response.status_code}: {response.text}")
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response payload: {data}")
    return data

"""
Submit Function Utilized for Asynch Handling (Conccurent Request)
"""
def _submit(idx: int, row: Dict[str, str]) -> Tuple[int, Dict]:
        text = row.get(TEXT_COLUMN, "")
        payload = build_payload(text, EXTRACTORS, LINKERS, RESOLVERS)
        for attempt in range(1, max(1, RETRIES) + 1):
            try:
                triples = submit_request(PLUMBER_URL, payload, REQUEST_TIMEOUT)
                break
            except Exception as e:
                if attempt < max(1, RETRIES):
                    time.sleep(max(0.0, RETRY_WAIT))
                    continue
                raise
        rec = {"text": text, "triples": triples}
        if ID_COLUMN in row:
            rec["id"] = row[ID_COLUMN]
        return idx, rec



"""
(EXPERIMENTAL) Normalization of Tripets
"""
def _normalize_triple(item: Dict) -> Optional[Tuple[str, str, str]]:
    """Coerce a triple-like dict into (subject,predicate,object) strings if possible."""
    if not isinstance(item, dict):
        return None
    s = item.get("subject") or item.get("s") or item.get("subj")
    p = item.get("predicate") or item.get("p") or item.get("rel") or item.get("relation")
    o = item.get("object") or item.get("o") or item.get("obj")
    if isinstance(s, dict):
        s = s.get("text") or s.get("value") or s.get("label")
    if isinstance(p, dict):
        p = p.get("text") or p.get("value") or p.get("label")
    if isinstance(o, dict):
        o = o.get("text") or o.get("value") or o.get("label")
    if s and p and o:
        return str(s).strip(), str(p).strip(), str(o).strip()
    return None


"""
Process CSV to Extract Triplets from Plumber
"""
def process_csv() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {CSV_PATH}")

    # Prepare graph outputs
    nodes_csv = NODES_CSV_PATH or (OUTPUT_JSONL.parent / "nodes.csv")
    edges_csv = EDGES_CSV_PATH or (OUTPUT_JSONL.parent / "edges.csv")

    nodes = {}
    edge_count = 0

    # Collect rows upfront to enable parallel requests
    collected = []
    for idx, row in enumerate(read_rows(CSV_PATH)):
        text = (row.get(TEXT_COLUMN) or "").strip()
        if not text:
            continue

        # Use our configs yay
        if MAX_CHARS is not None and len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
        row = dict(row)
        row[TEXT_COLUMN] = text
        if idx >= START_INDEX:
            collected.append((idx, row))
        if MAX_ROWS is not None and len(collected) >= MAX_ROWS:
            break

    total = len(collected)
    if total == 0:
        print("No rows to process after filtering.")
        return

    # Create Node/Edge Dirs
    nodes_csv.parent.mkdir(parents=True, exist_ok=True)
    edges_csv.parent.mkdir(parents=True, exist_ok=True)
    edge_fields = ["src", "rel", "dst", "src_label", "dst_label", "row_id"]


    # To Check Time Elapsed
    start_time = time.time()
    processed = 0
    errors = 0
    results = {}
    
    
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with nodes_csv.open("w", encoding="utf-8", newline="") as nodes_file, edges_csv.open(
        "w", encoding="utf-8", newline=""
    ) as edges_file:
        nodes_writer = csv.DictWriter(nodes_file, fieldnames=["id", "label", "type"])
        nodes_writer.writeheader()

        edges_writer = csv.DictWriter(edges_file, fieldnames=edge_fields)
        edges_writer.writeheader()

        """
        Add Node to CSV (will need to move this out after)
        """
        def _add_node(label: str, ntype: str = "entity") -> str:
            def _node_id(label: str, ntype: str = "entity") -> str:
                key = f"{ntype}|{label}".lower().strip()
                h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
                return f"node:{h}"

            nid = _node_id(label, ntype)
            if nid not in nodes:
                node_rec = {"id": nid, "label": label, "type": ntype}
                nodes[nid] = node_rec
                nodes_writer.writerow(node_rec)
                nodes_file.flush()
            return nid

        # Open JSONL for streaming writes
        with OUTPUT_JSONL.open("w", encoding="utf-8") as out_jsonl:
            with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as ex:
                futs = {ex.submit(_submit, idx, row): (idx, row) for idx, row in collected}
                for fut in as_completed(futs):
                    idx, row = futs[fut]
                    try:
                        i, rec = fut.result()
                        results[i] = rec
                    except Exception as e:
                        errors += 1
                        rec = {"text": row.get(TEXT_COLUMN, ""), "triples": [], "error": str(e)}
                        results[idx] = rec

                    # Stream row immediately
                    out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_jsonl.flush()

                    # Update graph incrementally
                    for t in rec.get("triples", []):
                        norm = _normalize_triple(t)
                        if not norm:
                            continue
                        s, p, o = norm

                        # TODO: Make this into a helper function for better encapsulation
                        sid = _add_node(s, "entity")
                        oid = _add_node(o, "entity")
                        edge = {"src": sid, "rel": p, "dst": oid, "src_label": s, "dst_label": o}
                        if "row_id" in edge_fields and ID_COLUMN and ID_COLUMN in row:
                            edge["row_id"] = str(row[ID_COLUMN])
                        edges_writer.writerow(edge)
                        edges_file.flush()
                        edge_count += 1

                    processed += 1

                    if processed % max(1, LOG_EVERY) == 0:
                        elapsed = time.time() - start_time
                        avg = elapsed / processed if processed else 0.0
                        remaining = max(0, total - processed)
                        eta_sec = remaining * avg
                        pct = (processed / total) * 100.0
                        print(
                            f"[progress] {processed}/{total} ({pct:0.1f}%) | avg {avg:0.2f}s/row | elapsed {elapsed:0.1f}s | ETA {eta_sec:0.1f}s | errors {errors}"
                        )

                    if SLEEP_SECONDS:
                        time.sleep(SLEEP_SECONDS)

    total_elapsed = time.time() - start_time
    rps = processed / total_elapsed if total_elapsed > 0 and processed > 0 else 0.0
    print(
        f"Processed {processed} rows in {total_elapsed:0.1f}s ({rps:0.2f} rows/s); errors={errors}; results saved to {OUTPUT_JSONL}"
    )

    print(f"Nodes CSV: {nodes_csv} (unique nodes: {len(nodes)})")
    print(f"Edges CSV: {edges_csv} (edges: {edge_count})")


def main() -> None:
    process_csv()


if __name__ == "__main__":
    main()
