# Copyright Center for Digital Humanities, Princeton University 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""
Data type for NLLB-200's supported languages. See the model's github repo
for more details: https://github.com/facebookresearch/fairseq/blob/nllb/README.md
"""

# Language codes gathered from the HuggingFace model's
# special_tokens_map.json
lang_index = {
    "af": "afr_Latn",  # Afrikaans
    "ak": "aka_Latn",  # Akan
    "am": "amh_Ethi",  # Amharic
    "ar": "arb_Arab",  # Assume Arabic maps to Modern Standard Arabic
    "as": "asm_Beng",  # Assamese
    "ay": "ayr_Latn",  # Assume Aymara maps to Central Aymara
    "ba": "bak_Cyrl",  # Bashkir
    "be": "bel_Cyrl",  # Belarusian
    "bg": "bul_Cyrl",  # Bulgarian
    "bm": "bam_Latn",  # Bambara
    "bn": "ben_Beng",  # Bengali
    "bo": "bod_Tibt",  # Standard Tibetan
    "bs": "bos_Latn",  # Bosnian
    "ca": "cat_Latn",  # Catalan
    "cs": "ces_Latn",  # Czech
    "cy": "cym_Latn",  # Welsh
    "da": "dan_Latn",  # Danish
    "de": "deu_Latn",  # German
    "dz": "dzo_Tibt",  # Dzongkha
    "el": "ell_Grek",  # Greek
    "en": "eng_Latn",  # English
    "eo": "epo_Latn",  # Esperanto
    "es": "spa_Latn",  # Spanish
    "et": "est_Latn",  # Estonian
    "eu": "eus_Latn",  # Basque
    "ee": "ewe_Latn",  # Ewe
    "fo": "fao_Latn",  # Faroese
    "fi": "fin_Latn",  # Finnish
    "fj": "fij_Latn",  # Fijian
    "fr": "fra_Latn",  # French
    "ga": "gle_Latn",  # Irish
    "gd": "gla_Latn",  # Scottish Gaelic
    "gl": "glg_Latn",  # Galician
    "gn": "grn_Latn",  # Guarani
    "gu": "guj_Gujr",  # Gujarati
    "ha": "hau_Latn",  # Hausa
    "he": "heb_Hebr",  # Hebrew
    "hi": "hin_Deva",  # Hindi
    "hr": "hrv_Latn",  # Croatian
    "ht": "hat_Latn",  # Haitian Creole
    "hu": "hun_Latn",  # Hungarian
    "hy": "hye_Armn",  # Armenian
    "id": "ind_Latn",  # Indonesian
    "ig": "ibo_Latn",  # Igbo
    "is": "isl_Latn",  # Icelandic
    "it": "ita_Latn",  # Italian
    "ja": "jpn_Jpan",  # Japanese
    "jv": "jav_Latn",  # Javanese
    "ka": "kat_Geor",  # Georgian
    "kg": "kon_Latn",  # Kikongo
    "ki": "kik_Latn",  # Kikuyu
    "kk": "kaz_Cyrl",  # Kazakh
    "km": "khm_Khmr",  # Khmer
    "kn": "kan_Knda",  # Kannada
    "ko": "kor_Hang",  # Korean
    "ky": "kir_Cyrl",  # Kyrgyz
    "lb": "ltz_Latn",  # Luxembourgish
    "lg": "lug_Latn",  # Ganda
    "li": "lim_Latn",  # Limburgish
    "ln": "lin_Latn",  # Lingala
    "lo": "lao_Laoo",  # Lao
    "lt": "lit_Latn",  # Lithuanian
    "mg": "plt_Latn",  # Assume Malagasy maps to Plateau Malagasy
    "mi": "mri_Latn",  # Maori
    "ml": "mal_Mlym",  # Malayalam
    "mk": "mkd_Cyrl",  # Macedonian
    "mn": "khk_Cyrl",  # Assume Mongolian maps to Halh Mongolian
    "mr": "mar_Deva",  # Marathi
    "ms": "zsm_Latn",  # Assume Malay maps to Standard Malay
    "mt": "mlt_Latn",  # Maltese
    "my": "mya_Mymr",  # Burmese
    "nb": "nob_Latn",  # Norwegian Bokmål
    "ne": "npi_Deva",  # Nepali
    "nl": "nld_Latn",  # Dutch
    "nn": "nno_Latn",  # Norwegian Nynorsk
    "ny": "nya_Latn",  # Nyanja
    "oc": "oci_Latn",  # Occitan
    "om": "gaz_Latn",  # Assume Oromo maps to West Central Oromo
    "or": "ory_Orya",  # Assume Oriya maps to Odia
    "pa": "pan_Guru",  # Assume Punjabi maps to Eastern Panjabi
    "pl": "pol_Latn",  # Polish
    "ps": "pbt_Arab",  # Southern Pashto
    "pt": "por_Latn",  # Portuguese
    "qu": "quy_Latn",  # Assume Quechua maps to Ayacucho Quechua
    "rn": "run_Latn",  # Rundi
    "ro": "ron_Latn",  # Romanian
    "ru": "rus_Cyrl",  # Russian
    "rw": "kin_Latn",  # Kinyarwanda
    "sa": "san_Deva",  # Sanskrit
    "sc": "srd_Latn",  # Sardinian
    "sd": "snd_Arab",  # Sindhi
    "sg": "sag_Latn",  # Sango
    "si": "sin_Sinh",  # Sinhala
    "sk": "slk_Latn",  # Slovak
    "sl": "slv_Latn",  # Slovenian
    "sm": "smo_Latn",  # Samoan
    "sn": "sna_Latn",  # Shona
    "so": "som_Latn",  # Somali
    "sq": "als_Latn",  # Assume Albanian maps to Tosk Albanian
    "sr": "srp_Cyrl",  # Serbian
    "ss": "ssw_Latn",  # Swati
    "st": "sot_Latn",  # Southern Sotho
    "sv": "swe_Latn",  # Swedish
    "sw": "swh_Latn",  # Swahili
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "tg": "tgk_Cyrl",  # Tajik
    "th": "tha_Thai",  # Thai
    "ti": "tir_Ethi",  # Tigrinya
    "tk": "tuk_Latn",  # Turkmen
    "tl": "tgl_Latn",  # Tagalog
    "tn": "tsn_Latn",  # Tswana
    "tr": "tur_Latn",  # Turkish
    "ts": "tso_Latn",  # Tsonga
    "tt": "tat_Cyrl",  # Tatar
    "tw": "twi_Latn",  # Twi
    "ug": "uig_Arab",  # Uyghur
    "uk": "ukr_Cyrl",  # Ukrainian
    "ur": "urd_Arab",  # Urdu
    "uz": "uzn_Latn",  # Assume Uzbek maps to Northern Uzbek
    "vi": "vie_Latn",  # Vietnamese
    "wo": "wol_Latn",  # Wolof
    "xh": "xho_Latn",  # Xhosa
    "yi": "ydd_Hebr",  # Assume Yiddish maps to Eastern Yiddish
    "yo": "yor_Latn",  # Yoruba
    "zh": "zho_Hans",  # Assume Chinese maps to simplified Chinese
    "zu": "zul_Latn",  # Zulu
    ## Currently excluded supported languages
    # "ace_Arab",  # Acehnese (Arabic script)
    # "ace_Latn",  # Acehnese (Latin script)
    # "acm_Arab",  # Mesopotamian Arabic
    # "acq_Arab",  # Ta'izzi-Adeni Arabic
    # "aeb_Arab",  # Tunisian Arabic
    # "ajp_Arab",  # South Levantine Arabic
    # "apc_Arab",  # North Levantine Arabic
    # "ars_Arab",  # Najdi Arabic
    # "ary_Arab",  # Moroccan Arabic
    # "arz_Arab",  # Egyptian Arabic
    # "ast_Latn",  # Asturian
    # "awa_Deva",  # Awadhi
    # "azb_Arab",  # South Azerbaijani
    # "azj_Latn",  # North Azerbaijani
    # "ban_Latn",  # Balinese
    # "bem_Latn",  # Bemba
    # "bho_Deva",  # Bhojpuri
    # "bjn_Arab",  # Banjar (Arabic script)
    # "bjn_Latn",  # Banjar (Latin script)
    # "bug_Latn",  # Buginese
    # "ceb_Latn",  # Cebuano
    # "cjk_Latn",  # Chokwe
    # "ckb_Arab",  # Central Kurdish
    # "crh_Latn",  # Crimean Tatar
    # "dik_Latn",  # Southwestern Dinka
    # "dyu_Latn",  # Dyula
    # "fon_Latn",  # Fon
    # "fur_Latn",  # Friulian
    # "fuv_Latn",  # Nigerian Fulfulde
    # "hne_Deva",  # Chhattisgarhi
    # "ilo_Latn",  # Ilocano
    # "kab_Latn",  # Kabyle
    # "kac_Latn",  # Jingpho
    # "kam_Latn",  # Kamba
    # "kas_Arab",  # Kashmiri (Arabic script)
    # "kas_Deva",  # Kashmiri (Devanagari script)
    # "knc_Arab",  # Central Kanuri (Arabic script)
    # "knc_Latn",  # Central Kanuri (Latin script)
    # "kbp_Latn",  # Kabiyè
    # "kea_Latn",  # Kabuverdianu
    # "kmb_Latn",  # Kimbundu
    # "kmr_Latn",  # Northern Kurdish
    # "lij_Latn",  # Ligurian
    # "lmo_Latn",  # Lombard
    # "ltg_Latn",  # Latgalian
    # "lua_Latn",  # Luba-Kasai
    # "luo_Latn",  # Luo
    # "lus_Latn",  # Mizo
    # "lvs_Latn",  # Standard Latvian
    # "mag_Deva",  # Magahi
    # "mai_Deva",  # Maithili
    # "min_Latn",  # Minangkabau
    # "mni_Beng",  # Meitei
    # "mos_Latn",  # Mossi
    # "nso_Latn",  # Northern Sotho
    # "nus_Latn",  # Nuer
    # "pag_Latn",  # Pangasinan
    # "pap_Latn",  # Papiamento
    # "pes_Arab",  # Western Persian
    # "prs_Arab",  # Dari
    # "sat_Beng",  # Santali
    # "scn_Latn",  # Sicilian
    # "shn_Mymr",  # Shan
    # "sun_Latn",  # Sundanese
    # "szl_Latn",  # Silesian
    # "taq_Latn",  # Tamasheq (Latin script)
    # "taq_Tfng",  # Tamasheq (Tifinagh script)
    # "tpi_Latn",  # Tok Pisin
    # "tum_Latn",  # Tumbuka
    # "tzm_Tfng",  # Central Atlas Tamazight
    # "umb_Latn",  # Umbundu
    # "vec_Latn",  # Venetian
    # "war_Latn",  # Waray
    # "yue_Hant",  # Yue Chinese
    # "zho_Hant",  # Traditional Chinese
}
