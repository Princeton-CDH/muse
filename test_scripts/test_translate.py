# Copyright Center for Digital Humanities, Princeton University 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""
Quick test script tests the translate() function.
Models will download automatically on first run - may take a few minutes.
Test cases are loaded from test_cases.jsonl, which contains real sentences
from the parallel corpus.
"""

import json
import sys
from pathlib import Path

from muse.translation.translate import translate


def main():
    if len(sys.argv) != 2:
        print("Usage: test_translate.py <input.jsonl>")
        sys.exit(1)
    # Load test cases from file
    test_cases_file = Path(sys.argv[1])
    test_cases = []

    with test_cases_file.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            test_cases.append(
                {
                    "id": record["id"],
                    "lang": record["lang"],
                    "text": record["text"],
                    "en_tr": record["en_tr"],
                }
            )

    print(f"Loaded {len(test_cases)} test cases from {test_cases_file.name}")

    # Models to test
    models = [
        "nllb",
        "hymt",
        "madlad",
    ]

    for model in models:
        print(f"\n{'=' * 80}")
        print(f"Testing model: {model}")
        print(f"{'=' * 80}\n")

        for tc in test_cases:
            src_lang = tc["lang"]
            text = tc["text"]
            reference = tc["en_tr"]

            # Translate
            result = translate(
                model=model,
                src_lang=src_lang,
                tgt_lang="en",
                text=text,
                verbose=True,
            )

            # Display results
            print(f"Model: {model}")
            print(f"[ID {tc['id']}] {src_lang}→en")
            print(f"Original:    {text}")
            print(f"Translation: {result}")
            print(f"Reference:   {reference}")
            print()


if __name__ == "__main__":
    main()
