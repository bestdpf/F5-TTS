import os
import threading

from lingua import Language, LanguageDetectorBuilder

from f5_tts.model.tokenizer import VoiceBpeTokenizer, split_sentence


LANG_ID = {
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
    Language.ITALIAN: 12,
    Language.TURKISH: 13,
    Language.RUSSIAN: 14,
    Language.CZECH: 15,
    Language.HUNGARIAN: 16,
    Language.HINDI: 17
}

LANG_LOCALE = {
    Language.ENGLISH: 'en',
    Language.SPANISH: 'es',
    Language.FRENCH: 'fr',
    Language.KOREAN: 'ko',
    Language.JAPANESE: 'ja',
    Language.ARABIC: 'ar',
    Language.DUTCH: 'nl',
    Language.CHINESE: 'zh',
    Language.GERMAN: 'de',
    Language.POLISH: 'pl',
    Language.PORTUGUESE: 'pt',
    Language.ITALIAN: 'it',
    Language.TURKISH: 'tr',
    Language.RUSSIAN: 'ru',
    Language.CZECH: 'cs',
    Language.HUNGARIAN: 'hu',
    Language.HINDI: 'hi'
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


XTTS_TPKENIZER = None
def get_xtts_tokenizer():
    global XTTS_TPKENIZER
    if XTTS_TPKENIZER is not None:
        return XTTS_TPKENIZER
    with threading.Lock():
        if XTTS_TPKENIZER is None:
            vob_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "vocab.json")
            XTTS_TPKENIZER = VoiceBpeTokenizer(vocab_file=vob_file)
        return XTTS_TPKENIZER