import tensorflow as tf
import json

lang_to_code = {}
code_to_lang = {}


def load_mapping(path):
    global lang_to_code, code_to_lang

    for split in ["train", "validation"]:
        labels_path = tf.io.gfile.join(path, f"{split}.json")
        labels = json.loads(tf.io.gfile.GFile(labels_path, "r").read())

        for label in labels:
            code = int(label["id"])
            caption = label["label"]
            lang_to_code[caption] = code
            code_to_lang[code] = caption


def lang_encode(lang):
    if not lang:
        return -1
    elif lang in lang_to_code:
        return lang_to_code[lang]
    else:
        raise ValueError(f"Language {lang} not found in mapping")


def lang_decode(code):
    if code in code_to_lang:
        return code_to_lang[code]
    else:
        raise ValueError(f"Code {code} not found in mapping")


def get_encodings():
    return lang_to_code, code_to_lang
