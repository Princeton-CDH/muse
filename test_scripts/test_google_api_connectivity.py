"""
This script only tests Google Cloud API connectivity.
For translation quality testing, use test_translate.py with model="google/translation-llm"
Before running, set up credentials:
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
"""

import os
import time
from pathlib import Path

from google.cloud import translate_v3

# API configuration
PROJECT_ID = "cdh-muse"
CREDENTIALS_FILE = "cdh-muse-6950d66acf83.json"
TEST_TEXT = "Hello World"


def run_connectivity_test():
    """Run Google Cloud API connectivity test."""
    # Check credentials
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

    # Initialize client
    try:
        client = translate_v3.TranslationServiceClient()
        print("✓ Successfully initialized TranslationServiceClient")
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        return False

    # Test API call
    try:
        start_time = time.time()
        parent = f"projects/{PROJECT_ID}/locations/us-central1"
        model_path = f"{parent}/models/general/translation-llm"

        response = client.translate_text(
            contents=[TEST_TEXT],
            target_language_code="es",
            source_language_code="en",
            parent=parent,
            model=model_path,
            mime_type="text/plain",
        )
        elapsed = time.time() - start_time

        print("✓ API call successful")
        print(f"✓ Response time: {elapsed:.2f} seconds")
        print(f"✓ Model used: {response.translations[0].model}")
        print(f"Source text: {TEST_TEXT}")
        print(f"Translation result: {response.translations[0].translated_text}")
        return True
    except Exception as e:
        print(f"✗ API call failed: {e}")
        return False


if __name__ == "__main__":
    print("Google Cloud API - Connectivity Tests")
    print()

    if not run_connectivity_test():
        print("\n⚠ Test failed. Set GOOGLE_APPLICATION_CREDENTIALS and try again.")
        print(f'   export GOOGLE_APPLICATION_CREDENTIALS="{CREDENTIALS_FILE}"')
        exit(1)
