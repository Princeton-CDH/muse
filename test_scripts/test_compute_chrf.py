#!/usr/bin/env python3
"""
Test script for compute_chrf() function.

Usage:
    python test_scripts/test_compute_chrf.py mt_test_data.jsonl
"""

import json
import sys
from pathlib import Path

from muse.metrics import compute_chrf


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_scripts/test_compute_chrf.py <mt_corpus.jsonl>")
        sys.exit(1)

    mt_file = Path(sys.argv[1])
    if not mt_file.exists():
        print(f"Error: File not found: {mt_file}")
        sys.exit(1)

    mt_records = []
    with mt_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                mt_records.append(json.loads(line))

    print(f"Testing {len(mt_records)} translations\n")

    by_direction = {}
    for record in mt_records:
        direction = f"{record['src_lang']}→{record['tr_lang']}"
        if direction not in by_direction:
            by_direction[direction] = []
        by_direction[direction].append(record)

    for direction, records in by_direction.items():
        scores = [
            compute_chrf(tr_text=r["tr_text"], ref_text=r["ref_text"]) for r in records
        ]
        avg = sum(scores) / len(scores)
        print(f"{direction}: {avg:.2f} (n={len(scores)})")

    all_scores = [
        compute_chrf(tr_text=r["tr_text"], ref_text=r["ref_text"]) for r in mt_records
    ]

    print(f"\nOverall: {sum(all_scores) / len(all_scores):.2f}")
    print(f"Range: {min(all_scores):.2f} - {max(all_scores):.2f}")


if __name__ == "__main__":
    main()
