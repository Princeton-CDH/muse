# muse

This repository contains in-progress experimental research software for the CDH project [MuSE (Multilingual Semantic Embeddings)](https://cdh.princeton.edu/projects/muse/).

For developer setup instructions, including Google Cloud Translation configuration, see [docs/DEVELOPERNOTES.md](docs/DEVELOPERNOTES.md).

## Phase 1

The first phase of the project, we will assess how well off-the-shelf multilingual translation models perform in the music-theoretical domain.

### Models

Three models will be evaluated: a commercial state-of-the-art model and two open-weights models available on 🤗 Hugging Face.

1. **TTLM**. Google's [Translation LLM (TTLM) model](https://docs.cloud.google.com/translate/docs/translation-llm) available through Google Cloud Translation.

2. **HY-MT1.5**. Tencent's Hunyuan Translation Model Version 1.5. We use the [1.8B parameter model](https://huggingface.co/tencent/HY-MT1.5-1.8B).

3. **MADLAD-400**. Google's MADLAD-400 translation model that supports over 400 languages. We use the [3B parameter model](https://huggingface.co/google/madlad400-3b-mt).

### Software Pipeline

The software for this phase can be broken into the three stages: (1) building parallel corpora, (2) translating this corpora, and (3) evaluating the resulting machine translations. Each of these stages corresponds to a module within the `muse` software package.

1. `parallel_corpus`: Contains code for building the parallel text corpora used to assess the machine translation models.

2. `translation`: Contains code for generating machine translations.

3. `evaluation`: Contains code for evaluating machine translations.
