# encoding=utf-8

import re
import nltk
from typing import List

def tokenize_by_punctuation(msg: str) -> str:
    # should not use _
    punctuation = r'([!"#$%&\'()*+,-./:;<=>?@\[\]^`{|}~]|\\(?!n))'
    new_msg = re.sub(punctuation, r' \1 ', msg)
    # rules below won't be used
    id_regex = r'< (commit_id|issue_id) >'
    new_msg = re.sub(id_regex, r'<\1>', new_msg)
    new_msg = " ".join(re.sub(r'\n', ' <nl> ', new_msg).split())
    return new_msg


def camel_case_split(identifier):
    return re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', identifier)).strip().split()


def tokenize_identifier_raw(token, keep_underscore=True):
    regex = r'(_+)' if keep_underscore else r'_+'
    id_tokens = []
    for t in re.split(regex, token):
        if t:
            id_tokens += camel_case_split(t)
    return list(filter(lambda x: len(x) > 0, id_tokens))


def tokenize_identifier(token, with_con=False):
    if with_con:
        id_tokens = " <con> ".join(tokenize_identifier_raw(token, keep_underscore=True)).split()
    else:
        id_tokens = [t.lower() for t in tokenize_identifier_raw(token, keep_underscore=False)]
    return id_tokens


def tokenize_text(text):
    str_tokens = []
    nltk_tokenized = " ".join(nltk.word_tokenize(text))
    content_tokens = re.sub(r'([-!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r' \1 ', nltk_tokenized).split()
    for t in content_tokens:
        str_tokens += tokenize_identifier(t)
    return str_tokens


def tokenize_text_with_con(text):
    def _tokenize_word(word):
        new_word = re.sub(r'([-!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r' \1 ', word)
        subwords = nltk.word_tokenize(new_word)
        new_subwords = []
        for w in subwords:
            new_subwords += tokenize_identifier_raw(w, keep_underscore=True)
        return new_subwords

    tokens = []
    for word in text.split():
        if not word:
            continue
        tokens += " <con> ".join(_tokenize_word(word)).split()
    return tokens


if __name__ == "__main__":
    pass
    
