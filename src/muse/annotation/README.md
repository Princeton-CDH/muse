# Annotation Module

This module contains all code related to our Prodigy annotation tasks for Phase 1 of MuSE, namely Prodigy recipes and scripts to build Prodigy data inputs.

## Annotation Tasks

In Phase 1, we conduct two annotation tasks one focused on evaluating the machine translations of music-theoretical concepts and the other on evaluating the translation quality of machine translations of paragraphs from music theory journal articles.

### Term Evaluation

In this annotation task, we evaluate how well machine translation models translate music theoretical concepts into English.
In this task, for a given musical concept and a non-English source text that mentions this concept, we evaluate a machine translation of this source text by
(a) identifying the translation(s) of the musical concept within the machine translation and (b) labeling the quality of this translation with a working evaluation typology.

#### Related Content

- Recipe `concept-eval` in `annotation_recipes.py`: annotation recipe for this task
- Recipe `muse-task-progress` in `command_recipes.py`: command recipe for reporting annotation progress
- `build_notion_concept_tasks.py`: script for building task input

### Paragraph Evaluation

🚧 This task is still being designed and has not yet been implemented. 🚧

## Using Prodigy

We do not include `prodigy` as a dependency of our `muse` package and recommend using a separate virtual environment for `prodigy`. Note that `prodigy` requires a license key to install.

### Running Recipes

To run custom Prodigy recipes, the source file must be passed in via the `-F` input flag.

#### `concept-eval`

This annotation recipe can be run using the following command:

```bash
prodigy concept-eval dataset_name task_input.jsonl -F /path/to/annotation_recipes.py [-I instructions.html]
```

where

- `dataset_name` is the name that will be associated with the resulting annotations saved to Prodigy's database
- `task_input.jsonl` is the input JSONL produced by `build_notion_concept_tasks.py`
- `path/to/annotation_recipes.py` is the path to `annotation_recipess.py` which contains the `concept-eval` recipe
- `instructions.html` is an optional HTML file containing annotation instructions to be displayed in the Prodigy web application

##### Environment Variables

This recipe also requires the `PRODIGY_ALLOWED_SESSIONS` environment variable to be set to the list of the expected annotator session names.
These names must be include a language prefix with teh following form: `[ISO 639-1 code]-` (e.g., `pt-`, `zh-`)

#### `muse-task-progress`

This command recipe reports the progress of an ongoing MuSE annotation task.
It is designed to be compatible with both our term and paragraph evaluation annotation tasks.
It reports language- and annotator-level progress.

This recipe can be run using the following command:

```bash
prodigy muse-task-progress dataset_name -F /path/to/command_recipes.py [-s task_input.jsonl]
```

where

- `dataset_name` is the name of the dataset used by the annotation task instance of interest
- `/path/to/command_recipes.py` is the path to `command_recipes.py` which contains the `muse-task-progress` recipe
- `task_input.jsonl` is the input JSONL used by the annotation task which can optionally be provided to report progress percentages

### Exporting Results

To export the annotations for a given Prodigy annotation task run, Prodigy's built-in `db-out` command recipe.
This recipe will generated a JSONL file of the saved annotations within the database.

This recipe can be run using the following command:

```bash
prodigy db-out dataset_name > output.jsonl
```
