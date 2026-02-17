"""
Google Cloud API - Connectivity Tests

This script tests Google Cloud API connectivity and infrastructure:
- Authentication and credentials validation
- API connectivity and response times
- API response metadata

For translation quality testing, use test_translate.py with model="google/translation-llm"

Authentication:
    Set up credentials before running:
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
"""

import os
import time
from pathlib import Path

from google.cloud import translate_v3

from muse.translation.translate import google_cloud_translate

# API configuration
PROJECT_ID = "cdh-muse"
CREDENTIALS_FILE = "cdh-muse-6950d66acf83.json"


def test_credentials_validation():
    """Test that credentials are properly configured."""
    print("\n=== Test 1: Credentials Validation ===")

    # Check if credentials file exists
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        print(f"✓ GOOGLE_APPLICATION_CREDENTIALS set: {creds_path}")
        if Path(creds_path).exists():
            print("✓ Credentials file exists")
        else:
            print(f"✗ Credentials file not found at {creds_path}")
            return False
    else:
        print("⚠ GOOGLE_APPLICATION_CREDENTIALS not set, using default credentials")

    # Try to initialize client
    try:
        translate_v3.TranslationServiceClient()
        print("✓ Successfully initialized TranslationServiceClient")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        return False


def test_api_connectivity():
    """Test basic API connectivity with a simple translation."""
    print("\n=== Test 2: API Connectivity ===")

    try:
        start_time = time.time()
        result = google_cloud_translate(
            src_lang="en",
            tgt_lang="es",
            text="Hello",
            project_id=PROJECT_ID,
            verbose=False,
        )
        elapsed = time.time() - start_time

        print("✓ API call successful")
        print(f"✓ Response time: {elapsed:.2f} seconds")
        print(f"✓ Translation result: '{result}'")
        return True
    except Exception as e:
        print(f"✗ API call failed: {e}")
        return False


def test_api_metadata():
    """Test API response metadata and model information."""
    print("\n=== Test 3: API Response Metadata ===")

    try:
        client = translate_v3.TranslationServiceClient()
        parent = f"projects/{PROJECT_ID}/locations/us-central1"
        model_path = f"{parent}/models/general/translation-llm"

        response = client.translate_text(
            contents=["Hello"],
            target_language_code="es",
            source_language_code="en",
            parent=parent,
            model=model_path,
            mime_type="text/plain",
        )

        print(f"✓ Model used: {response.translations[0].model}")
        print(f"✓ Translation: {response.translations[0].translated_text}")
    except Exception as e:
        print(f"✗ Failed to get metadata: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("Google Cloud API - Connectivity Tests")
    print("=" * 80)

    # Check credentials first
    if not test_credentials_validation():
        print(
            "\n⚠ Credentials not configured. Set GOOGLE_APPLICATION_CREDENTIALS and try again."
        )
        print(f'   export GOOGLE_APPLICATION_CREDENTIALS="{CREDENTIALS_FILE}"')
        exit(1)

    # Run tests
    test_api_connectivity()
    test_api_metadata()

    print("\n" + "=" * 80)
    print("All connectivity tests completed!")
    print("=" * 80)
    print("\nFor translation quality testing, run:")
    print("  python test_scripts/test_translate.py")
