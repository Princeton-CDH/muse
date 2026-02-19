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

#### Dealing with Multiple Google Cloud Projects

If you’ve used `gcloud` for other projects, make sure that the ADC corresponds to the correct project. **Switching configs within `gcloud` will not update the ADC file.** However, `gcloud` will provide a warning if the activated (quota/billing) project does not match the one in the ADC file.
