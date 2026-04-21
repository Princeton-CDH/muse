# Copyright Center for Digital Humanities, Princeton University 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""
Test script for compute_comet() function.

Usage:
    python test_scripts/test_compute_comet.py mt_test_data.jsonl
"""

import sys
from pathlib import Path

import orjsonl

from muse.evaluation.metrics import compute_comet


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_scripts/test_compute_comet.py <mt_corpus.jsonl>")
        sys.exit(1)

    mt_file = Path(sys.argv[1])
    if not mt_file.exists():
        print(f"Error: File not found: {mt_file}")
        sys.exit(1)

    mt_records = orjsonl.load(mt_file)

    print(f"Testing {len(mt_records)} translations\n")

    by_direction = {}
    for record in mt_records:
        direction = f"{record['src_lang']}→{record['tr_lang']}"
        if direction not in by_direction:
            by_direction[direction] = []
        by_direction[direction].append(record)

    for direction, records in by_direction.items():
        scores = [
            compute_comet(
                tr_text=r["tr_text"],
                src_text=r["src_text"],
                ref_text=r["ref_text"],
            )
            for r in records
        ]
        avg = sum(scores) / len(scores)
        print(f"{direction}: {avg:.4f} (n={len(scores)})")

    all_scores = [
        compute_comet(
            tr_text=r["tr_text"],
            src_text=r["src_text"],
            ref_text=r["ref_text"],
        )
        for r in mt_records
    ]

    print(f"\nOverall: {sum(all_scores) / len(all_scores):.4f}")
    print(f"Range: {min(all_scores):.4f} - {max(all_scores):.4f}")


if __name__ == "__main__":
    main()
