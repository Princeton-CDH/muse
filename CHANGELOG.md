# CHANGELOG

## 0.2.0

### Annotation

- `build_report` script for building annotation results from exported Prodigy annotation data

## 0.1.1

- Update `concept-eval` recipe to fix cross-session filtering error

## 0.1.0

- Phase 1 release

### Documentation

- Data design doc
- Guide for using della and example slurm script

### Parallel corpus creation

- `build_sentence` script for building a parallel sentence-level text corpus from Notion concept data
- `build_paragraph` script for building a parallel paragraph-level text corpus from MTO parallel article data

### Machine translation

- `translate_corpus` script for translating a parallel text corpus for a given machine translation model
- Utilities for generating machine translations using the following models:
  HY-MT 1.5 (1.8B), NLLB-200 (3.3B), MADLAD-400 (3B), TranslateGemma (4B), and Google's Translation LLM model (via Google Cloud Translate API)

### Evaluation

- `evaluate_corpus` script for computing machine translation metrics for a translation corpus
- Utilities for computing ChrF, COMET, and COMET-Kiwi machine translation metrics
- Notebook for exploring machine translations and their corresponding machine translation metrics

### Annotation

- `concept-eval` Prodigy recipe for evaluating the translations of musical concepts
- `build_notion_concept_tasks` script for preparing input for `concept-eval` Prodigy recipe
- Prodigy command recipe for reporting annotation progress
- Documentation of annotation tasks and guide for using Prodigy

### Misc

- Pre-commit hooks for ruff formatter and linter, yamlfmt, mdformat, uv.lock update check, GitHub Actions validator, other common file checks
- GitHub Actions for ruff formatter and linter checks as well as changelog update checker
