import tensorflow as tf
import json

import openai
import argparse
import os
import tqdm
import numpy as np
import multiprocessing
import argparse
import os
import openai
import tqdm
import code


PROMPT = (
    "Generate %d variations of the following command: %s\nNumber them like 1. 2. 3.\nBe concise and use synonyms.\n"
)

client = openai.OpenAI()



lang_to_code = {}
code_to_lang = {}
NONE = -1


def load_mapping(path, constructor=dict, augmented=False):
    global lang_to_code, code_to_lang
    if not augmented:
        encode_path = tf.io.gfile.join(path, "language_encodings.json")
        decode_path = tf.io.gfile.join(path, "language_decodings.json")
    else:
        encode_path = tf.io.gfile.join(path, "language_encodings_aug.json")
        decode_path = tf.io.gfile.join(path, "language_decodings_aug.json")
    lang_to_code = constructor(json.loads(tf.io.gfile.GFile(encode_path, "r").read()))
    code_to_lang = constructor({int(k): v for k, v in json.loads(tf.io.gfile.GFile(decode_path, "r").read()).items()})


def flush_mapping(path):
    encode_path = tf.io.gfile.join(path, "language_encodings.json")
    decode_path = tf.io.gfile.join(path, "language_decodings.json")
    with tf.io.gfile.GFile(encode_path, "w") as f:
        f.write(json.dumps(dict(lang_to_code)))
    with tf.io.gfile.GFile(decode_path, "w") as f:
        f.write(json.dumps(dict(code_to_lang)))


def lang_encode(lang):
    if not lang:
        return NONE
    elif lang in lang_to_code:
        return lang_to_code[lang]
    else:
        code = len(lang_to_code)
        lang_to_code[lang] = code
        code_to_lang[code] = lang
        return code


rng = np.random.RandomState(0)


def lang_decode(code, aug=True):
    global rng
    if code == NONE:
        return None
    text = code_to_lang[code]
    choices = text.split("\n")
    return rng.choice(choices) if (rng and aug) else choices[0]


def lang_encodings():
    return code_to_lang


def augment(args):
    code, lang = args
    langs = lang.split("\n")
    langs = [l if l.endswith(".") else l + "." for l in langs]
    prompt = PROMPT % (5, langs[0])
    if len(langs) > 1:
        prompt += "Start with:\n"
        for i, l in enumerate(langs):
            prompt += f"{i+1}. {l}\n"
    
    

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
        ],
    )
    
    response = response.choices[0].message.content
    
    
    try:
        new_langs = response.split("\n")
        new_langs = [l[3:] for l in new_langs]
    except:
        print("Error parsing response")
        new_langs = []

    new_lang = "\n".join(langs + new_langs)

    return new_lang



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_paraphrases", type=int, default=5)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    print(args)

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    
    path = args.path
    load_mapping(path)
    has_variants = [lang for lang in lang_to_code if "\n" in lang]
    print(f"Loaded {len(lang_encodings())} languages, {len(has_variants)} have variants")

    PROMPT = (
        "Generate %d variations of the following command: %s\nNumber them like 1. 2. 3.\nBe concise and use synonyms.\n"
    )

    client = openai.OpenAI()

    new_code_to_lang = manager.dict()
    new_lang_to_code = manager.dict()
    
    

    with multiprocessing.Pool(20) as p:
        results = p.imap(augment, [(code, lang) for code, lang in code_to_lang.items()])

        for new_lang in tqdm.tqdm(results, total=len(code_to_lang)):
            new_code_to_lang[code] = new_lang
            new_lang_to_code[new_lang] = code

    encode_path = tf.io.gfile.join(path, "language_encodings_aug.json")
    decode_path = tf.io.gfile.join(path, "language_decodings_aug.json")
    with tf.io.gfile.GFile(encode_path, "w") as f:
        f.write(json.dumps(dict(new_lang_to_code)))
    with tf.io.gfile.GFile(decode_path, "w") as f:
        f.write(json.dumps(dict(new_code_to_lang)))

