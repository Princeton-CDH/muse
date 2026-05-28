# Copyright Center for Digital Humanities, Princeton University 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""
Constants used by the annotation submodule
"""

#: The error typology for the concept-eval task
CONCEPT_EVAL_TYPOLOGY = {
    "correct": "Correct translation",
    "translated": "Should not translate",
    "missing": "Omitted or missing",
    "ils": "Incorrect lexical selection",
    "dit": "Disambiguation issue in target",
    "untranslated": "Incorrectly left untranslated",
    "other": "Other error",
}
