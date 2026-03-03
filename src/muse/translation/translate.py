"""
Library for performing machine translation (MT) for a given language for various
MT models. All supported models are can translate Chinese, Japanese, and Spanish
to English and vice versa.

The translate() function provides a unified interface for translating text across
multiple models. Model-specific functions (hymt_translate, nllb_translate,
madlad_translate, gemma_translate, google_cloud_translate) are also available for
direct use.
"""

import os
from timeit import default_timer as timer

import google.auth
from google.cloud import translate_v3
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from muse.translation.gemma_langs import lang_index as gemma_lang_idx
from muse.translation.hymt_langs import lang_idx_en as hymt_lang_idx_en
from muse.translation.hymt_langs import lang_idx_zh as hymt_lang_idx_zh
from muse.translation.nllb_langs import lang_index as nllb_lang_idx

# Workaround to reuse loaded models / tokenizers
LOADED_MODEL = {
    "model_name": None,
    "model": None,
    "tokenizer": None,
}


def get_max_new_tokens(input_token_len: int) -> int:
    """
    Helper function that sets the restriction for model generation based on
    the model input's token length. This is used by all HuggingFace translate
    functions.

    Currently, it returns double the input length.
    """
    return 2 * input_token_len


def hymt_translate(
    src_lang: str,
    tgt_lang: str,
    text: str,
    model_name: str = "tencent/HY-MT1.5-1.8B",
    verbose: bool = False,
) -> str:
    """
    Translate text written in source language to target language with Tencent's
    Hunyuan Translation Model Version 1.5 (HY-MT-1.5). Languages are specified with
    their ISO 639-1 codes.

    By default, the 1.8B translation model (tencent/HY-MT1.5-1.8B) is used, but an
    alternative model may be specified via `model_name`.
    """
    # Validate input languages
    if src_lang not in hymt_lang_idx_en:
        raise ValueError(f"Source language '{src_lang}' is not supported")
    if tgt_lang not in hymt_lang_idx_en:
        raise ValueError(f"Target language '{tgt_lang}' is not supported")

    # Construct model prompt using templates described in model card:
    # https://huggingface.co/tencent/HY-MT1.5-1.8B
    if src_lang == "zh" or tgt_lang == "zh":
        # Use Chinese template when Chinese is the source or target language
        tgt_lang_name = hymt_lang_idx_zh[tgt_lang]
        prompt = (
            f"将以下文本翻译为{tgt_lang_name}，注意只需要输出翻译后的结果，"  # noqa: RUF001
            f"不要额外解释：\n\n{text}"  # noqa: RUF001
        )
    else:
        # For all other language pairs, use English prompt
        tgt_lang_name = hymt_lang_idx_en[tgt_lang]
        prompt = (
            f"Translate the following segment into {tgt_lang_name}, without "
            f"additional explanation.\n\n{text}"
        )

    # Get tokenizer and model
    ## Load model and tokenizer if it's not the currently loaded model
    if model_name != LOADED_MODEL["model_name"]:
        if verbose:
            start = timer()
        LOADED_MODEL["model_name"] = model_name
        LOADED_MODEL["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        LOADED_MODEL["model"] = AutoModelForCausalLM.from_pretrained(model_name)
        if verbose:
            print(f"Loaded tokenizer & model in {timer() - start:.0f} seconds")
    tokenizer = LOADED_MODEL["tokenizer"]
    model = LOADED_MODEL["model"]

    # Generate model input
    messages = [{"role": "user", "content": prompt}]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    input_len = tokenized_chat[0].size()[0]
    if verbose:
        print(f"Input length: {input_len} tokens")

    # Generate translation
    if verbose:
        start = timer()
    outputs = model.generate(
        tokenized_chat.to(model.device),
        max_new_tokens=get_max_new_tokens(input_len),
    )
    if verbose:
        print(f"Generated model output in {timer() - start:.0f} seconds")
    # Model output begins with initial prompt
    tr_tokens = outputs[0][input_len:]
    if verbose:
        # Report generated output length excluding the prompt prefix
        print(f"Output length: {outputs[0].size()[0] - input_len} tokens")
    tr_text = tokenizer.decode(tr_tokens, skip_special_tokens=True)

    return tr_text


def nllb_translate(
    src_lang: str,
    tgt_lang: str,
    text: str,
    model_name: str = "facebook/nllb-200-3.3B",
    verbose: bool = False,
) -> str:
    """
    Translate text written in source language to target language with Meta's
    No Language Left Behind (NLLB) NLLB-200 translation models. Languages are
    specified with their ISO 639-1 codes.

    By default, the 3.3B translation model (facebook/nllb-200-3.3B) is used,
    but an alternative model may be specified via `model_name`.
    """
    # Validate input languages
    if src_lang not in nllb_lang_idx:
        # Only used for input validation, not generation
        raise ValueError(f"Source language '{src_lang}' is not supported")
    if tgt_lang not in nllb_lang_idx:
        raise ValueError(f"Target language '{tgt_lang}' is not supported")

    # Get tokenizer and model
    ## Load models if it's not the currently loaded model
    if model_name != LOADED_MODEL["model_name"]:
        if verbose:
            start = timer()
        LOADED_MODEL["model_name"] = model_name
        LOADED_MODEL["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        LOADED_MODEL["model"] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if verbose:
            print(f"Loaded tokenizer & model in {timer() - start:.0f} seconds")
    tokenizer = LOADED_MODEL["tokenizer"]
    model = LOADED_MODEL["model"]

    # Generate model input
    ## Set source language for proper tokenization
    tokenizer.src_lang = nllb_lang_idx[src_lang]
    model_inputs = tokenizer(text, return_tensors="pt")
    input_len = model_inputs["input_ids"][0].size()[0]
    if verbose:
        print(f"Input length: {input_len} tokens")

    # Generate translation
    if verbose:
        start = timer()
    outputs = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(nllb_lang_idx[tgt_lang]),
        max_new_tokens=get_max_new_tokens(input_len),
    )
    if verbose:
        print(f"Generated model output in {timer() - start:.0f} seconds")
    tr_tokens = outputs[0]
    if verbose:
        # Report generated output length excluding the prompt prefix
        print(f"Output length: {outputs[0].size()[0]} tokens")
    tr_text = tokenizer.decode(tr_tokens, skip_special_tokens=True)

    return tr_text


def madlad_translate(
    tgt_lang: str,
    text: str,
    model_name: str = "google/madlad400-3b-mt",
    verbose: bool = False,
) -> str:
    """
    Translate text written to target language with Google's MADLAD-400
    translation models. Languages are specified with their ISO 639-1 codes.
    The MADLAD-400 translation models were trained the MADLAD-400 dataset and
    have a T5 architecture.

    By default, the 3B translation model (google/madlad400-3b-mt) is used,
    but an alternative model may be specified via `model_name`.
    """
    # NOTE: Target language is not validated.
    #       These maodels use BCP-47 for language codes, which uses ICO-639-1
    #       wehn applicable and ISO-693-3 codes otherwise. See the paper for a
    #       full list of the 419 supported languages and their codes:
    #       https://arxiv.org/pdf/2309.04662#subsection.A.1

    # Get tokenizer and model
    ## Load models if it's not the currently loaded model
    if model_name != LOADED_MODEL["model_name"]:
        if verbose:
            start = timer()
        LOADED_MODEL["model_name"] = model_name
        LOADED_MODEL["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        LOADED_MODEL["model"] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if verbose:
            print(f"Loaded tokenizer & model in {timer() - start:.0f} seconds")
    tokenizer = LOADED_MODEL["tokenizer"]
    model = LOADED_MODEL["model"]

    # Generate model input
    model_inputs = tokenizer(f"<2{tgt_lang}> {text}", return_tensors="pt")
    input_len = model_inputs["input_ids"][0].size()[0]
    if verbose:
        print(f"Input length: {input_len} tokens")

    # Generate translation
    if verbose:
        start = timer()
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=get_max_new_tokens(input_len),
    )
    if verbose:
        print(f"Generated model output in {timer() - start:.0f} seconds")
    tr_tokens = outputs[0]
    if verbose:
        # Report generated output length excluding the prompt prefix
        print(f"Output length: {outputs[0].size()[0]} tokens")
    tr_text = tokenizer.decode(tr_tokens, skip_special_tokens=True)

    return tr_text


def gemma_translate(
    src_lang: str,
    tgt_lang: str,
    text: str,
    model_name: str = "google/translategemma-4b-it",
    verbose: bool = False,
) -> str:
    """
    Translate text written in source language to target language with Google's
    TranslateGemma model. Languages are specified with their ISO 639-1 codes.

    By default, the 4B instruction-tuned model (google/translategemma-4b-it) is used,
    but an alternative model may be specified via `model_name`.

    Note: This model requires HuggingFace authentication. See docs/DEVELOPERNOTES.md
    for setup instructions.
    """
    # Validate input languages
    if src_lang not in gemma_lang_idx:
        raise ValueError(f"Source language '{src_lang}' is not supported")
    if tgt_lang not in gemma_lang_idx:
        raise ValueError(f"Target language '{tgt_lang}' is not supported")

    # Get language names for prompt
    src_lang_name = gemma_lang_idx[src_lang]
    tgt_lang_name = gemma_lang_idx[tgt_lang]

    # Construct prompt using TranslateGemma's recommended format
    system_message = (
        f"You are a professional {src_lang_name} ({src_lang}) to "
        f"{tgt_lang_name} ({tgt_lang}) translator. Your goal is to "
        f"accurately convey the meaning and nuances of the original "
        f"{src_lang_name} text while adhering to {tgt_lang_name} "
        f"grammar, vocabulary, and cultural sensitivities."
    )

    # Get tokenizer and model
    # Load model and tokenizer if it's not the currently loaded model
    if model_name != LOADED_MODEL["model_name"]:
        if verbose:
            start = timer()
        try:
            LOADED_MODEL["model_name"] = model_name
            LOADED_MODEL["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
            LOADED_MODEL["model"] = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            # Check if error is related to authentication
            error_str = str(e).lower()
            if "401" in error_str or "authentication" in error_str or "gated" in error_str:
                err_msg = (
                    f"Failed to load model '{model_name}'. This model requires "
                    "HuggingFace authentication. See docs/DEVELOPERNOTES.md for "
                    "setup instructions."
                )
                raise RuntimeError(err_msg) from e
            # Re-raise original exception if not authentication-related
            raise
        if verbose:
            print(f"Loaded tokenizer & model in {timer() - start:.0f} seconds")
    tokenizer = LOADED_MODEL["tokenizer"]
    model = LOADED_MODEL["model"]

    # Generate model input using chat template
    # TranslateGemma requires a specific message format with source/target language codes
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": src_lang,
                    "target_lang_code": tgt_lang,
                    "text": text,
                }
            ],
        }
    ]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_len = tokenized_chat[0].size()[0]
    if verbose:
        print(f"Input length: {input_len} tokens")

    # Generate translation
    if verbose:
        start = timer()
    outputs = model.generate(
        tokenized_chat.to(model.device),
        max_new_tokens=get_max_new_tokens(input_len),
    )
    if verbose:
        print(f"Generated model output in {timer() - start:.0f} seconds")
    # Model output begins with initial prompt
    tr_tokens = outputs[0][input_len:]
    if verbose:
        # Report generated output length excluding the prompt prefix
        print(f"Output length: {outputs[0].size()[0] - input_len} tokens")
    tr_text = tokenizer.decode(tr_tokens, skip_special_tokens=True)

    return tr_text


def google_cloud_translate(
    src_lang: str,
    tgt_lang: str,
    text: str,
    verbose: bool = False,
) -> str:
    """
    Translate text using Google Cloud Translate API with Translation LLM (TLLM) model.
    Languages are specified with their ISO 639-1 codes (e.g., "zh", "ja", "es", "en").

    Requires gcloud CLI authentication. See docs/DEVELOPERNOTES.md for setup.

    Args:
        src_lang: Source language ISO 639-1 code
        tgt_lang: Target language ISO 639-1 code
        text: Text to translate from source to target language
        verbose: If True, print timing information

    Returns:
        Translated text as a string

    Raises:
        RuntimeError: If there is an issue loading the Google Application
                     Default Credentials (ADC)
    """
    # Get project id from Google Application Default Credentials
    try:
        _, project_id = google.auth.default()
    except Exception as e:
        err_msg = (
            "Issue loading Application Default Credentials (ADC). "
            "See developer notes for more details."
        )
        raise RuntimeError(err_msg) from e

    # Default to us-central 1 if not set in environment
    region = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

    if verbose:
        start = timer()

    client = translate_v3.TranslationServiceClient()

    if verbose:
        print(
            f"Initialized Google Cloud Translate client in {timer() - start:.2f} seconds"
        )

    parent = f"projects/{project_id}/locations/{region}"
    model_path = f"{parent}/models/general/translation-llm"

    if verbose:
        start = timer()

    response = client.translate_text(
        contents=[text],
        target_language_code=tgt_lang,
        source_language_code=src_lang,
        parent=parent,
        model=model_path,
        mime_type="text/plain",
    )

    if verbose:
        print(f"Received translation response in {timer() - start:.2f} seconds")

    translated_text = response.translations[0].translated_text

    return translated_text


def translate(
    model: str,
    src_lang: str,
    tgt_lang: str,
    text: str,
    verbose: bool = False,
) -> str:
    """
    Translate text using a specified translation model. This function provides a
    unified interface for translating text across multiple translation models by
    routing to the appropriate model-specific implementation based on the model
    parameter.

    Supported models:
        - hymt: Tencent's Hunyuan Translation Model v1.5 (1.8B)
        - madlad: Google's MADLAD-400 (3B)
        - nllb: Meta's No Language Left Behind (3.3B)
        - gemma: Google's TranslateGemma (4B)
        - googe_tllm: Google Cloud Translation LLM (TLLM)

    Languages are specified using ISO 639-1 codes (e.g., "zh", "ja", "es", "en").
    Language validation is delegated to the model-specific functions, so supported
    languages vary by model. The MADLAD model does not use the source language
    parameter internally, but it is accepted for API consistency.

    Args:
        model: Model identifier (must be one of the supported models)
        src_lang: Source language ISO 639-1 code
        tgt_lang: Target language ISO 639-1 code
        text: Text to translate from source to target language
        verbose: If True, print timing information and token counts

    Returns:
        Translated text as a string

    Raises:
        ValueError: If the specified model is not supported, or if the source/target
                    languages are not supported by the chosen model
    """

    if model == "hymt":
        return hymt_translate(src_lang, tgt_lang, text, verbose=verbose)
    elif model == "nllb":
        return nllb_translate(src_lang, tgt_lang, text, verbose=verbose)
    elif model == "madlad":
        # MADLAD does not use src_lang parameter
        return madlad_translate(tgt_lang, text, verbose=verbose)
    elif model == "gemma":
        return gemma_translate(src_lang, tgt_lang, text, verbose=verbose)
    elif model == "google_tllm":
        return google_cloud_translate(src_lang, tgt_lang, text, verbose=verbose)
    else:
        raise ValueError(f"Unsupported model: {model}")
