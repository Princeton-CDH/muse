# Developer Notes

## Google Cloud Translation Setup

The MUSE project supports Google Cloud's Translation LLM (TLLM) model for machine translation. This requires Google Cloud CLI (gcloud) setup and authentication.

### Installing Google Cloud CLI

Install `gcloud` using the [provided installation guide](https://cloud.google.com/sdk/docs/install).

To verify the installation run:

```bash
gcloud --version
```

### Authentication with Application Default Credentials (ADC)

For Google Cloud authentication, we will rely on the ADC file that can be generated with the following command:

```bash
gcloud auth application-default login
```

The ADC file is written to the following location:
`~/.config/gcloud/application_default_credentials.json`

#### Working with Multiple Google Cloud Projects

If you’ve used `gcloud` for other projects, make sure that your local ADC file corresponds to the correct project. **Switching configs within `gcloud` will not update the ADC file.** However, `gcloud` will provide a warning if the activated (quota/billing) project does not match the one in the ADC file.

To switch quote projects run:

```bash
gcloud auth application-default set-quote-project [project id]
```

Alternatively, a different credential file may be selected by setting the `GOOGLE_APPLICATION_CREDENTIALS` environmental variable. See the [Google ADC guide](https://docs.cloud.google.com/docs/authentication/application-default-credentials) for more information.

## HuggingFace Authentication Setup

Several translation models in the MUSE project are hosted on HuggingFace and require authentication to access. This includes Google's TranslateGemma model, which is a gated model requiring license acceptance.

### Creating a HuggingFace Account

If you don't have a HuggingFace account:

1. Visit [https://huggingface.co/join](https://huggingface.co/join)
2. Create a free account
3. Verify your email address

### Accepting Model License

For gated models like TranslateGemma:

1. Visit the model page: [https://huggingface.co/google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it)
2. Click "Agree and access repository" to accept the license terms
3. Wait for approval (usually instant for open models)

### Generating an Access Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a descriptive name (e.g., "muse-translation")
4. Select "Read" access type (sufficient for downloading models)
5. Click "Generate token"
6. Copy the token (you won't be able to see it again)

### Authentication Methods

Choose one of the following methods to authenticate:

#### Method 1: HuggingFace CLI (Recommended)

Install the HuggingFace CLI if not already installed:

```bash
pip install huggingface_hub
```

Login with your token:

```bash
huggingface-cli login
```

Paste your token when prompted. This stores your credentials in `~/.cache/huggingface/token`.

#### Method 2: Environment Variable

Set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN="your_token_here"
```

Add this to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist across sessions.

#### Method 3: Programmatic (Not Recommended for Development)

You can pass the token directly in code, but this is not recommended for security reasons:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "google/translategemma-4b-it",
    token="your_token_here"
)
```

### Troubleshooting

**Error: "401 Client Error: Unauthorized"**
- Ensure you've accepted the model license on HuggingFace
- Verify your token has "Read" permissions
- Check that authentication is properly configured using one of the methods above

**Error: "Repository Not Found"**
- Verify the model name is correct
- Ensure you've accepted the license for gated models
- Check your internet connection

**Error: "Rate limit exceeded"**
- HuggingFace has rate limits for unauthenticated requests
- Ensure you're properly authenticated
- Wait a few minutes and try again
