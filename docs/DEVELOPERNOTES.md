# Developer Notes

## Google Cloud Translation Setup

The MUSE project supports Google Cloud's Translation LLM (TLLM) model for machine translation. This requires Google Cloud CLI (gcloud) setup and authentication.

### Prerequisites

1. **Install Google Cloud CLI**

   - Follow instructions at: https://cloud.google.com/sdk/docs/install
   - Verify installation: `gcloud --version`

2. **Authenticate with Application Default Credentials**

   ```bash
   gcloud auth application-default login
   ```

3. **Set required environment variables**

   ```bash
   export GOOGLE_CLOUD_PROJECT="cdh-muse"
   ```
