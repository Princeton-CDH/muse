"""
Quick test script tests the translate() function.
Models will download automatically on first run - may take a few minutes.
"""

import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent / "src"))

from muse.translation.translate import translate

# Test data - single words and sentences
test_cases = [
    # Spanish sentences
    ("es", "en", "Me gusta la música clásica."),
    ("es", "en", "¿Dónde está la biblioteca?"),
    # Chinese sentences
    ("zh", "en", "我喜欢听音乐。"),
    ("zh", "en", "今天天气很好。"),
    # Japanese sentences
    ("ja", "en", "音楽を聴くのが好きです。"),
    ("ja", "en", "今日はいい天気ですね。"),
]

# Models to test
models = [
    "facebook/nllb-200-3.3B",
    "tencent/HY-MT1.5-7B",
    "google/madlad400-7b-mt",
]

for model in models:
    print(f"Testing model: {model}")

    for src, tgt, text in test_cases:
        result = translate(
            model=model,
            src_lang=src,
            tgt_lang=tgt,
            text=text,
            verbose=True,
        )
        print(f"{src}→{tgt}: '{text}' → '{result}'\n")

    print()
