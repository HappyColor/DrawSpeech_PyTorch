
from string import punctuation
from g2p_en import G2p

import numpy as np
import re

from drawspeech.utilities.text import text_to_sequence

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_english(text, return_word_phone_alignment=False, always_use_g2p=False, g2p=None, verbose=True):
    text = text.rstrip(punctuation)

    if always_use_g2p:
        lexicon = None
        assert g2p is not None  # initialize once can save time
    else:
        lexicon_path = "drawspeech/utilities/text/lexicon/librispeech-lexicon.txt"
        lexicon = read_lexicon(lexicon_path)
        g2p = G2p()
    
    phones = []
    word_phone_alignment = []
    words = re.split(r"([,;.\-\?\!\s+])", text)

    if always_use_g2p:
        phones = list(filter(lambda p: p != " ", g2p(text)))
    else:
        for w in words:
            if w.lower() in lexicon:
                # phones += lexicon[w.lower()]
                p = lexicon[w.lower()]
                phones += p
            else:
                # phones += list(filter(lambda p: p != " ", g2p(w)))
                p = list(filter(lambda p: p != " ", g2p(w)))
                phones += p
            if len(p) > 0:
                word_phone_alignment.append((w, p, len(p)))

    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    if verbose:
        print("Raw Text Sequence: {}".format(text))
        print("Phoneme Sequence: {}".format(phones))
    sequence = text_to_sequence(phones, ["english_cleaners"])

    if return_word_phone_alignment:
        return sequence, phones, word_phone_alignment
    else:
        return sequence, phones