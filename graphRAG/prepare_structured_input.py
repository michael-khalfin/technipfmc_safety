"""
Utility script to reshape the raw `data/cleaned_data.csv` export into a more
LLM-friendly CSV for GraphRAG.

The raw `text` column mixes the free-form narrative with long metadata blocks.
This script splits the narrative, normalizes metadata into bullet-pointed facts,
and writes a tidy two-column CSV (id + structured text) that GraphRAG can chunk
more consistently, which improves triple interpretability.
"""

from __future__ import annotations

import argparse
import re
import csv
from pathlib import Path
from typing import List, Tuple

import pandas as pd


PATTERN_MAP: Tuple[Tuple[str, str], ...] = (
    ("This case is a type of ", "INCIDENT_TYPE"),
    ("The type of impact specified is ", "IMPACT_TYPE"),
    ("The description of the severity selected in the Risk Matrix is ", "SEVERITY_DESC"),
    ("The severity value for this case is ", "SEVERITY_SCORE"),
    ("The description of the type of likelihood selected in the Risk Matrix is ", "LIKELIHOOD_DESC"),
    ("This case is (", "LIKELIHOOD_RANGE"),
    ("The value of the likelihood is ", "LIKELIHOOD_SCORE"),
    ("The criticality of this case, which is the risk weight of the loss potential, is ", "CRITICALITY"),
    ("The criticality of this case is ", "CRITICALITY"),
    ("The value of the mitigated risk is ", "MITIGATED_RISK"),
    ("The mitigated risk color is ", "MITIGATED_RISK_COLOR"),
    ("The risk color for this case is ", "RISK_COLOR"),
    ("The system of record used for this case is ", "SOURCE_SYSTEM"),
    ("The date and the time of the case is ", "EVENT_DATETIME"),
    ("The title of the case is ", "TITLE"),
    ("The General Business Unit involved in the case is ", "GENERAL_BUSINESS_UNIT"),
    ("The specific Business Unit involved in the case is ", "SPECIFIC_BUSINESS_UNIT"),
    ("This case occurred at the workplace: ", "WORKPLACE"),
    ("The name of the project is ", "PROJECT"),
    ("The client for this case is ", "CLIENT"),
    ("The case categorization is ", "CATEGORY"),
    ("The work process involved is ", "WORK_PROCESS"),
    ("The life saving rules used in this case are ", "LIFE_SAVING_RULES"),
    ("The case was reported on ", "REPORTED_DATE"),
    ("The status of the case is ", "STATUS"),
    ("The loss potential severity of the case was ", "LOSS_POTENTIAL"),
    ("The date when the incident has been approved as completed is ", "APPROVED_DATE"),
    ("The date when the incident was reported as closed is ", "CLOSED_DATE"),
    ("The deadline for the incident to be closed is ", "DEADLINE"),
    ("The operating center corresponding to the selected incident is ", "OPERATING_CENTER"),
    ("The location code of the incident is ", "LOCATION_CODE"),
)


def normalize_spaces(value: str) -> str:
    """Collapse repeated whitespace and strip surrounding spaces."""
    return " ".join(value.replace("\r", " ").replace("\n", " ").split()).strip()


def split_narrative(text: str) -> Tuple[str, str]:
    """
    Separate the free-form narrative (first paragraphs) from the templated metadata
    block that always starts with 'This case ...'.
    """
    sentinel = "This case"
    idx = text.find(sentinel)
    if idx == -1:
        return normalize_spaces(text), ""
    return normalize_spaces(text[:idx]), normalize_spaces(text[idx:])


def build_fact_sentences(meta_text: str, max_items: int = 14) -> List[str]:
    """
    Convert the templated metadata sentences into KEY: value bullet points so the
    extractor can anchor entities/relations to well-labelled facts.
    """
    if not meta_text:
        return []

    sentences = [
        normalize_spaces(chunk.strip(" ."))
        for chunk in re.split(r"(?<=[.!?])\s+", meta_text)
        if chunk.strip()
    ]

    facts: List[str] = []
    leftovers: List[str] = []

    for sentence in sentences:
        matched = False
        for prefix, label in PATTERN_MAP:
            if sentence.startswith(prefix):
                value = sentence[len(prefix) :].strip(" .")
                value = value.lstrip(", ").strip()
                facts.append(f"- {label}: {value}")
                matched = True
                break
        if not matched:
            leftovers.append(sentence)

    # Keep a few additional contextual statements (client, workplace, etc.).
    for sentence in leftovers:
        lowered = sentence.lower()
        if any(token in lowered for token in ("client", "workplace", "project", "equipment", "location", "life saving", "injury", "deadline")):
            facts.append(f"- CONTEXT: {sentence}")
        if len(facts) >= max_items:
            break

    return facts[:max_items]


def format_record(record_id: str, text: str) -> str:
    """Build the structured text blob consumed by GraphRAG."""
    narrative, meta = split_narrative(text)
    facts = build_fact_sentences(meta)

    lines: List[str] = [f"INCIDENT_ID: {record_id}"]

    if narrative:
        lines.extend(["NARRATIVE:", narrative])
    else:
        lines.extend(["NARRATIVE:", "Narrative not provided."])

    if facts:
        lines.append("")
        lines.append("FACTS:")
        lines.extend(facts)

    return "\n".join(lines).strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert cleaned safety data into a structured GraphRAG input CSV.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/cleaned_data.csv"),
        help="Path to the raw CSV that contains RECORD_NO_LOSS_POTENTIAL and text columns.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("graphRAG/input/dev_sample.csv"),
        help="Output CSV path for GraphRAG (two columns: id + structured text).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap for quicker experiments (samples uniformly at random).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed applied when --limit is set.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source CSV not found: {args.source}")

    df = pd.read_csv(args.source, dtype={"RECORD_NO_LOSS_POTENTIAL": "string"})
    df = df.dropna(subset=["text"]).copy()
    df["RECORD_NO_LOSS_POTENTIAL"] = df["RECORD_NO_LOSS_POTENTIAL"].str.strip()
    df = df.drop_duplicates(subset=["RECORD_NO_LOSS_POTENTIAL"], keep="last")

    if args.limit is not None and args.limit < len(df):
        df = (
            df.sample(n=args.limit, random_state=args.seed)
            .sort_values("RECORD_NO_LOSS_POTENTIAL")
            .reset_index(drop=True)
        )

    df["text"] = [
        format_record(record_id or "UNKNOWN", text_value)
        for record_id, text_value in zip(df["RECORD_NO_LOSS_POTENTIAL"], df["text"])
    ]

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    df[["RECORD_NO_LOSS_POTENTIAL", "text"]].to_csv(
        args.dest,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )

    print(f"Wrote {len(df):,} structured incidents to {args.dest}")


if __name__ == "__main__":
    main()
