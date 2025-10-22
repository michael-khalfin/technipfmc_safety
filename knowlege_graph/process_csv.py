#!/usr/bin/env python3
"""Utility to push CSV rows through a running Plumber instance and persist triples."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests


DEFAULT_EXTRACTORS = ["open_ie"]
DEFAULT_LINKERS = ["dummy"]
DEFAULT_RESOLVERS = ["dummy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ThePlumber triples extraction over a CSV file.")
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
    payload: Dict[str, object] = {
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


def process_csv(args: argparse.Namespace) -> None:
    if not args.csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv_path}")

    processed = 0
    with args.output.open("w", encoding="utf-8") as output_handle:
        for row in read_rows(args.csv_path):
            text = (row.get(args.text_column) or "").strip()
            if not text:
                continue
            payload = build_payload(text, args.extractor, args.linker, args.resolver)
            triples = submit_request(args.url, payload, args.timeout)
            record = {
                "text": text,
                "triples": triples,
            }
            if args.id_column and args.id_column in row:
                record["id"] = row[args.id_column]
            output_handle.write(json.dumps(record, ensure_ascii=False))
            output_handle.write("\n")
            processed += 1
            if args.max_rows is not None and processed >= args.max_rows:
                break
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Processed {processed} rows; results saved to {args.output}")


def main() -> None:
    args = parse_args()
    process_csv(args)


if __name__ == "__main__":
    main()
