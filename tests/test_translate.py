"""
Test script for the unified translate() function.
This script tests all 3 supported models and error handling.
"""

from muse.translation.translate import translate

print("=" * 80)
print("Testing Unified translate() Function")
print("=" * 80)

# Test 1: Error handling for unsupported model
print("\n[Test 1] Testing error handling for unsupported model...")
try:
    result = translate("unsupported/model", "en", "es", "hello")
    print("ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"✓ Expected error caught: {e}")

# Test 2: Error handling for invalid language
print("\n[Test 2] Testing error handling for invalid language...")
try:
    result = translate("tencent/HY-MT1.5-7B", "invalid_lang", "en", "test")
    print("ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"✓ Expected error caught: {e}")

# Test 3: HY-MT translation (Chinese to English)
print("\n[Test 3] Testing HY-MT: Chinese to English...")
print("Input: 音乐理论")
try:
    result = translate(
        model="tencent/HY-MT1.5-7B",
        src_lang="zh",
        tgt_lang="en",
        text="音乐理论",
        verbose=True,
    )
    print(f"✓ Translation: {result}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: NLLB translation (Japanese to English)
print("\n[Test 4] Testing NLLB: Japanese to English...")
print("Input: 音楽理論")
try:
    result = translate(
        model="facebook/nllb-200-3.3B",
        src_lang="ja",
        tgt_lang="en",
        text="音楽理論",
        verbose=True,
    )
    print(f"✓ Translation: {result}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: MADLAD translation (English to Spanish)
print("\n[Test 5] Testing MADLAD: English to Spanish...")
print("Input: music theory")
try:
    result = translate(
        model="google/madlad400-7b-mt",
        src_lang="en",  # Accepted but not used by MADLAD
        tgt_lang="es",
        text="music theory",
        verbose=True,
    )
    print(f"✓ Translation: {result}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 6: Reverse translation (English to Chinese)
print("\n[Test 6] Testing HY-MT: English to Chinese...")
print("Input: Hello, how are you?")
try:
    result = translate(
        model="tencent/HY-MT1.5-7B",
        src_lang="en",
        tgt_lang="zh",
        text="Hello, how are you?",
        verbose=True,
    )
    print(f"✓ Translation: {result}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("Testing Complete!")
print("=" * 80)
