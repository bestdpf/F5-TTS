import os
import threading

from transformers import BertTokenizerFast
from lingua import Language, LanguageDetectorBuilder
from phonemizer.phonemize import  BACKENDS, _phonemize, Separator

LANG_ID = {
    # 'unknow': 0,
    # 'en': 1,
    # 'es': 2,
    # 'fr': 3,
    # 'ko': 4,
    # 'ja': 5,
    # 'ar': 6,
    # 'nl': 7,
    # 'zh-cn': 8,
    # 'zh-tw': 9,
    # 'de': 11,
    # 'pl': 12,
    # 'pt': 13,
    # 'it': 14,
    Language.ENGLISH: 1,
    Language.SPANISH: 2,
    Language.FRENCH: 3,
    Language.KOREAN: 4,
    Language.JAPANESE: 5,
    Language.ARABIC: 6,
    Language.DUTCH: 7,
    Language.CHINESE: 8,
    Language.GERMAN: 9,
    Language.POLISH: 10,
    Language.PORTUGUESE: 11,
    Language.ITALIAN: 12
}

LANG_LOCALE = {
    Language.ENGLISH: 'en-us',
    Language.SPANISH: 'es',
    Language.FRENCH: 'fr-fr',
    Language.KOREAN: 'ko',
    Language.JAPANESE: 'ja',
    Language.ARABIC: 'ar',
    Language.DUTCH: 'nl',
    Language.CHINESE: 'cmn',
    Language.GERMAN: 'de',
    Language.POLISH: 'pl',
    Language.PORTUGUESE: 'pt-br',
    Language.ITALIAN: 'it'
}



detector = LanguageDetectorBuilder.from_languages(*LANG_ID.keys()).build()

def get_text_lang_id(text):
    try:
        lang_name = detector.detect_language_of(text)
    except:
        lang_name = 'unknow'
    ret = LANG_ID.get(lang_name, 0)
    if ret == 0:
        print(f'get 0 from {text} and {lang_name}')
    return ret

def get_text_lang_locale(text):
    try:
        lang_name = detector.detect_language_of(text)
    except:
        lang_name = 'unknow'
    ret = LANG_ID.get(lang_name, 0)
    lang_locale = LANG_LOCALE.get(lang_name, 'en-us')
    if ret == 0:
        print(f'get 0 from {text} and {lang_name}')
    return ret, lang_locale


PHN = {}

def get_phonizer(language='en-us'):
    global PHN
    if language in PHN:
        return PHN[language]
    with threading.Lock():
        if language not in PHN:
            PHN[language] = BACKENDS['espeak'](
            language,
            punctuation_marks=';:,.!?¡¿—…"«»“”' + '！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.',
            preserve_punctuation=True,
            with_stress=False,
            tie=False,
            language_switch='remove-flags',
            words_mismatch='ignore',
            logger=None)
        return PHN[language]

def run_phn(text, language='en-us', phone_sep=' '):
    phonemizer = get_phonizer(language)
    separator = Separator(phone=phone_sep, word='\t', syllable='|')
    return _phonemize(phonemizer, [text], separator, False, 1, False, True)[0].replace('\t', ' ')


PHN_TOKENIZER = None
def get_phn_tokenizer(device='cpu'):
    global PHN_TOKENIZER
    if PHN_TOKENIZER is not None:
        return PHN_TOKENIZER
    with threading.Lock():
        if PHN_TOKENIZER is None:
            PHN_TOKENIZER_DIR = os.environ.get('PHN_TOKENIZER_DIR', '/home/projects/u6554606/llm/split_phn_tokenizer')
            PHN_TOKENIZER = BertTokenizerFast.from_pretrained(PHN_TOKENIZER_DIR, device_map=device)
        return PHN_TOKENIZER

