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
