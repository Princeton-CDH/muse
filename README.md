# muse

Repository for CDH-RSE partnership Multilingual Semantic Embeddings

## Google Cloud Translation Setup

To use the Google Cloud Translation LLM model:

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → IAM & Admin → Service Accounts
2. Create a service account with **Cloud Translation API User** role
3. Create and download a JSON key file
4. Set the environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```
