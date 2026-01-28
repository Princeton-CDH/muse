"""
Library for performing machine translation (MT) for a given language for various
MT models. All supported models are can translate Chinese, Japanese, and Spanish
to English and vice versa.
"""

from timeit import default_timer as timer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from muse.translation.hymt_langs import (
    lang_idx_en as hymt_lang_idx_en,
    lang_idx_zh as hymt_lang_idx_zh,
)


def hymt_translate(
    src_lang: str,
    tgt_lang: str,
    text: str,
    model_name: str = "tencent/HY-MT1.5-1.8B",
    verbose: bool = False) -> str:
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
            f"将以下文本翻译为{tgt_lang_name}，注意只需要输出翻译后的结果，"
            f"不要额外解释：\n\n{text}"
        )
    else:
        # For all other language pairs, use English prompt
        tgt_lang_name = hymt_lang_idx_en[tgt_lang]
        prompt = (
            f"Translate the following segment into {tgt_lang_name}, without "
            f"additional explanation.\n\n{text}"
        )
    
    # Initialize tokenizer and model
    if verbose:
        start = timer()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if verbose:
        print(f"Loaded tokenizer & model in {timer()-start:.0f} seconds")

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
    outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)
    if verbose:
        print(f"Generated model output in {timer()-start:.0f} seconds")
    # Model output begins with initial prompt
    tr_tokens = outputs[0][input_len:]
    if verbose:
        # Report generated output length excluding the prompt prefix
        print(f"Output length: {outputs[0].size()[0] - input_len} tokens")
    tr_text = tokenizer.decode(tr_tokens, skip_special_tokens=True)
    
    return tr_text
