# Data Design Document

*This serves as a living document to record the current designs of the various data
produced during the workflows of this project.*

## Parallel Text Corpus

These corpora serve as the initial data form for Phase 1. A parallel text corpus is a JSONL
file whose records correspond to non-English musical theoretical texts paired with their *professional* English translations.

Phase 1 uses two corpora: a sentence-level corpus built from Notion annotation data and a paragraph-level-corpus build from side-by-side translations for 5 MTO articles.

### Required Fields

A parallel text corpus has the following four fields regardless of the contents of the corpus:

| Field Name | Type | Description |
| --- | --- | --- |
| id | int | Unique identifier for the parallel text pair |
| lang | str | Language of the original text (ISO 639-1 language code) |
| text | str | Original text |
| en_tr | str | Professional English translation of the text |

In Phase 1, the languages of the original text are limited to Chinese (zh), Japanese (ja), and Spanish (es).

### Optional Fields

Sentence- and paragraph-level corpora will have additional optional fields for providing additional

#### Parallel Sentence Corpus Fields

A parallel sentence corpus includes contains the following additional information extracted from the Notion annotations:

| Field Name | Type | Description |
| --- | --- | --- |
| cite | str | Citation for the original sentence |
| tr_cite | str | Citation for the translated sentence |
| term | str | The musical-theoretical term associated with the sentence |

#### Parallel Paragraph Corpus Fields

A parallel paragraph corpus includes the following additional information about the associated Music Theory Online (MTO) articles:

| Field Name | Type | Description |
| --- | --- | --- |
| doc_id | str | MTO article ID (e.g., 30.4.12) |
| par_id | str | MTO paragraph ID (e.g., 1.20) |

## Machine Translation Corpus

A machine translation corpus is a JSONL file whose records correspond to individual translations by a specific translation model. For Phase 1, a machine translation corpus contains the translations of (1) the English translations of a the original texts of a parallel text corpus and (2) translations of the professional English translation back into the language of the original text (i.e., back translation).

### Fields

At this time there are no optional fields for a machine translation corpus.

| Field Name | Type | Description |
| --- | --- | --- |
| tr_id | str | Unique identifier (UUID) for the machine translation |
| pair_id | int | Parallel text record ID |
| model | str | Model used for translation |
| src_lang | str | Language of the source text (ISO 639-1 language code) |
| tr_lang | str | Language of the translation (ISO 639-1 language code) |
| src_text | str | Source text |
| ref_text | str | Reference translation |
| tr_text | str | Machine translated text |

So, the records for the English translation and back translation of a parallel text record will have the following values:

| Field | English Translation | Back Translation |
| --- | :-: | :-: |
| src_lang | `lang` of parallel text record | en |
| tr_lang | en | `lang` of the parallel text record |
| src_text | `text` of parallel text record | `en_tr` of parallel text record |
| ref_text | `en_tr` of parallel text record | `text` of parallel text record |

## Machine Translation Metric Scores

To evaluate the machine translations generated during Phase 1, machine translation metrics are computed for each translation. These scores are saved as a CSV with the following form:

| Field Name | Type | Description |
| --- | --- | --- |
| tr_id | str | Machine translation record ID |
| chrf | float | ChrF score for the translation |
| comet | float | COMET score for the translation |
| comet-kiwi | float | CometKiwi score for the translation |
