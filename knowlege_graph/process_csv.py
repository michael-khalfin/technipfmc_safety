"""Utility to push CSV rows through a running Plumber instance and persist triples,
then materialize a simple knowledge graph as nodes/edges CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import hashlib


DEFAULT_EXTRACTORS = ["open_ie"]
DEFAULT_LINKERS = ["FalconWikidataJoint"]
DEFAULT_RESOLVERS = ["spacy_neural_coreference"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ThePlumber triples extraction over a CSV file and build a graph.")
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Input CSV containing the text column to process.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of the column containing free-form text (default: text).",
    )
    parser.add_argument(
        "--id-column",
        default="RECORD_NO_LOSS_POTENTIAL",
        help="Optional identifier column to carry into the output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plumber_triples.jsonl"),
        help="Output JSONL file; one line per input row with extracted triples.",
    )
    parser.add_argument(
        "--nodes-csv",
        type=Path,
        default=None,
        help="Optional path to write deduplicated nodes CSV (id,label,type). Defaults next to --output.",
    )
    parser.add_argument(
        "--edges-csv",
        type=Path,
        default=None,
        help="Optional path to write edges CSV (src,rel,dst,src_label,dst_label[,row_id]). Defaults next to --output.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:5000",
        help="Base URL where the Plumber Flask API is listening.",
    )
    parser.add_argument(
        "--extractor",
        nargs="+",
        default=DEFAULT_EXTRACTORS,
        help="Extractor component keys to use (default: %(default)s).",
    )
    parser.add_argument(
        "--linker",
        nargs="+",
        default=DEFAULT_LINKERS,
        help="Linker component keys to use (default: %(default)s).",
    )
    parser.add_argument(
        "--resolver",
        nargs="+",
        default=DEFAULT_RESOLVERS,
        help="Resolver component keys to use (default: %(default)s).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of rows processed (useful for smoke-tests).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay (in seconds) between requests to avoid overloading the service.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print a progress line every N processed rows (default: 50).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel requests to Plumber (default: 4).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="If set, truncate input text to the first N characters before sending.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry failed requests up to N times (default: 2).",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=5.0,
        help="Seconds to wait between retries (default: 5).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N input rows (resume support).",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


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


def _default_graph_paths(output_path: Path) -> Tuple[Path, Path]:
    base_dir = output_path.parent
    return base_dir / "nodes.csv", base_dir / "edges.csv"


def process_csv(args: argparse.Namespace) -> None:
    if not args.csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv_path}")

    # Prepare graph outputs
    nodes_csv, edges_csv = _default_graph_paths(args.output)
    if args.nodes_csv is not None:
        nodes_csv = args.nodes_csv
    if args.edges_csv is not None:
        edges_csv = args.edges_csv

    nodes = {}
    edges = []

    def _add_node(label: str, ntype: str = "entity") -> str:
        def _node_id(label: str, ntype: str = "entity") -> str:
            key = f"{ntype}|{label}".lower().strip()
            h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            return f"node:{h}"
        nid = _node_id(label, ntype)
        if nid not in nodes:
            nodes[nid] = {"id": nid, "label": label, "type": ntype}
        return nid

    # Collect rows upfront to enable parallel requests
    collected = []
    for idx, row in enumerate(read_rows(args.csv_path)):
        text = (row.get(args.text_column) or "").strip()
        if not text:
            continue
        if args.max_chars is not None and len(text) > args.max_chars:
            text = text[: args.max_chars]
        row = dict(row)
        row[args.text_column] = text
        if idx >= args.start_index:
            collected.append((idx, row))
        if args.max_rows is not None and len(collected) >= args.max_rows:
            break

    total = len(collected)
    if total == 0:
        print("No rows to process after filtering.")
        return

    start_time = time.time()
    processed = 0
    errors = 0
    results = {}
    
    def _submit(idx: int, row: Dict[str, str]) -> Tuple[int, Dict]:
        text = row.get(args.text_column, "")
        payload = build_payload(text, args.extractor, args.linker, args.resolver)
        last_err = None
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                triples = submit_request(args.url, payload, args.timeout)
                break
            except Exception as e:
                last_err = e
                if attempt < max(1, args.retries) + 0:
                    time.sleep(max(0.0, args.retry_wait))
                else:
                    raise
        rec = {"text": text, "triples": triples}
        if args.id_column and args.id_column in row:
            rec["id"] = row[args.id_column]
        return idx, rec

    # Ensure output directory exists, then open JSONL for streaming writes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_jsonl:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futs = {ex.submit(_submit, idx, row): (idx, row) for idx, row in collected}
            for fut in as_completed(futs):
                idx, row = futs[fut]
                try:
                    i, rec = fut.result()
                    results[i] = rec
                except Exception as e:
                    errors += 1
                    rec = {"text": row.get(args.text_column, ""), "triples": [], "error": str(e)}
                    results[idx] = rec

                # Stream this row to JSONL immediately
                out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_jsonl.flush()

                # Update graph incrementally
                for t in rec.get("triples", []):
                    norm = _normalize_triple(t)
                    if not norm:
                        continue
                    s, p, o = norm
                    sid = _add_node(s, "entity")
                    oid = _add_node(o, "entity")
                    edge = {"src": sid, "rel": p, "dst": oid, "src_label": s, "dst_label": o}
                    if args.id_column and args.id_column in row:
                        edge["row_id"] = str(row[args.id_column])
                    edges.append(edge)

                processed += 1

                if processed % max(1, args.log_every) == 0:
                    elapsed = time.time() - start_time
                    avg = elapsed / processed if processed else 0.0
                    remaining = max(0, total - processed)
                    eta_sec = remaining * avg
                    pct = (processed / total) * 100.0
                    print(
                        f"[progress] {processed}/{total} ({pct:0.1f}%) | avg {avg:0.2f}s/row | elapsed {elapsed:0.1f}s | ETA {eta_sec:0.1f}s | errors {errors}"
                    )

                if args.sleep:
                    time.sleep(args.sleep)

    total_elapsed = time.time() - start_time
    rps = processed / total_elapsed if total_elapsed > 0 and processed > 0 else 0.0
    print(
        f"Processed {processed} rows in {total_elapsed:0.1f}s ({rps:0.2f} rows/s); errors={errors}; results saved to {args.output}"
    )

    # Write graph CSVs
    nodes_csv.parent.mkdir(parents=True, exist_ok=True)
    edges_csv.parent.mkdir(parents=True, exist_ok=True)

    with nodes_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "type"]) 
        w.writeheader()
        for n in nodes.values():
            w.writerow(n)

    edge_fields = ["src", "rel", "dst", "src_label", "dst_label"]
    if any("row_id" in e for e in edges):
        edge_fields.append("row_id")
    with edges_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=edge_fields)
        w.writeheader()
        for e in edges:
            w.writerow(e)

    print(f"Nodes CSV: {nodes_csv} (unique nodes: {len(nodes)})")
    print(f"Edges CSV: {edges_csv} (edges: {len(edges)})")


def main() -> None:
    args = parse_args()
    process_csv(args)


if __name__ == "__main__":
    main()
