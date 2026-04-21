# muse

This repository contains in-progress experimental research software for the CDH project [MuSE (Multilingual Semantic Embeddings)](https://cdh.princeton.edu/projects/muse/).

For developer setup instructions, including Google Cloud Translation configuration, see [DEVELOPERNOTES.md](docs/DEVELOPERNOTES.md).

## Phase 1

The first phase of the project, we assess how well off-the-shelf multilingual translation models perform in the music-theoretical domain.

### Models

We evaluate three models: a commercial state-of-the-art model and two open-weights models available on 🤗 Hugging Face.

1. **TTLM**. Google's [Translation LLM (TTLM) model](https://docs.cloud.google.com/translate/docs/translation-llm) available through Google Cloud Translation.

2. **HY-MT1.5**. Tencent's Hunyuan Translation Model Version 1.5. We use the [1.8B parameter model](https://huggingface.co/tencent/HY-MT1.5-1.8B).

3. **TranslateGemma**. Google's [TranslateGemma](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/) translation model that supports over 400 languages. We use the [4B parameter model](https://huggingface.co/google/translategemma-4b-it). _Note: This is a gated model which will require authentication via HuggingFace. See [DEVELOPERNOTES.md](docs/DEVELOPERNOTES.md) for more details._

During this phase, we also experimented with two additional open-weights models available on 🤗 Hugging Face. While we ultimately chose not to include them in our evaluation, our translation module supports them.

- **NLLB-200**. Facebook AI Research's No Language Left Behind (NLLB) translation model that supports over 200 languages. We used the [3.3B parameter model](https://huggingface.co/facebook/nllb-200-3.3B).

- **MADLAD-400**. Google's MADLAD-400 translation model that supports over 400 languages. We used the [3B parameter model](https://huggingface.co/google/madlad400-3b-mt).

### Software Pipeline

The software for this phase can be broken into four stages: (1) building parallel corpora, (2) translating these corpora, (3) evaluating these translations via machine translation metics, and (4) evaluating these machine translations via human annotation tasks.
Each of these stages corresponds to a module within the `muse` package.

1. `parallel_corpus`: module for building the parallel text copora used to assess select machine translation models
2. `translation`: module for generating machine translations with select machien translation models
3. `evaluation`: module for evaluating machine translations via quantitative metrics
4. `annotation`: module for supporting our Prodigy annotation tasks

### Additional Materials

Below is a list of additional materials created during this phase:

- `notebooks/mt_browser.py`: `marimo` notebook for viewing and exploring machine translation corpora and their quantiative evaluation scores
- `docs/data-design.md`: living document for recording the current designs of the various data produced during the workflows of this project
- `docs/della-guide.md`: guide for running MuSE translations jobs on Della
- `examples/slurm/translate-della.slurm`: example slurm script for running machine translation jobs on Della
- `test_scripts`: test scripts created during development
