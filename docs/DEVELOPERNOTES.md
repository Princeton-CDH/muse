# Developer Notes

## Google Cloud Translation LLM (TLLM) Setup

The MuSE project supports Google Cloud's Translation LLM (TLLM) model for machine translation. This requires Google Cloud CLI (gcloud) setup and authentication.

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

## TranslateGemma Setup

The MUSE project supports Google's TranslateGemma model for machine translation. This requires HuggingFace authentication and license acceptance.

### Installing HuggingFace CLI

Install the HuggingFace CLI using pip:

```bash
pip install huggingface-hub
```

Verify installation:

```bash
hf --version
```

### Generating an Access Token

To generate an access token:

1. Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token" and select "Read" access type
3. Copy the token and use it with `hf auth login`

For more information on security tokens, see the [HuggingFace documentation](https://huggingface.co/docs/hub/security-tokens).

### Authentication with HuggingFace CLI

For HuggingFace authentication, use the HuggingFace CLI to login with your access token:

```bash
hf auth login
```

The token is stored in the following location:
`~/.cache/huggingface/token`

#### Accepting Model License

For gated models like TranslateGemma, you must accept the license:

1. Visit the model page: [https://huggingface.co/google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it)
2. Click "Acknowledge license" to accept the license terms
