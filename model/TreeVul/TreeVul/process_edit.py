from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
import difflib
from typing import List
from allennlp.data.tokenizers.token_class import Token
from copy import deepcopy
import dataclasses

EMPTY_TOKEN = Token(text="<empty>", text_id=50265, type_id=0)
EQUAL_TOKEN = Token(text="equal")  # text_id=None, type_id=None
DELETE_TOKEN = Token(text="delete")
INSERT_TOKEN = Token(text="insert")
REPLACE_TOKEN = Token(text="replace")


def _heuristic_replace_match(a_tokens: List[Token], b_tokens: List[Token]):
    diff_seqs = []
    a_len = len(a_tokens)
    b_len = len(b_tokens)
    delta_len = max(a_len - b_len, b_len - a_len)
    if a_len != b_len:
        # don't take spaces into consideration when aligning two sequences
        a_first_token = a_tokens[0].text[1:] if 'Ġ' in a_tokens[0].text else a_tokens[0].text
        b_first_token = b_tokens[0].text[1:] if 'Ġ' in b_tokens[0].text else b_tokens[0].text
        a_last_token = a_tokens[-1].text[1:] if 'Ġ' in a_tokens[-1].text else a_tokens[-1].text
        b_last_token = b_tokens[-1].text[1:] if 'Ġ' in b_tokens[-1].text else b_tokens[-1].text
        head_ratio = difflib.SequenceMatcher(None, a_first_token, b_first_token).quick_ratio()
        tail_ratio = difflib.SequenceMatcher(None, a_last_token, b_last_token).quick_ratio()
        if head_ratio >= tail_ratio:
            if a_len > b_len:
                b_tokens += [deepcopy(EMPTY_TOKEN) for _ in range(delta_len)]
            else:
                a_tokens += [deepcopy(EMPTY_TOKEN) for _ in range(delta_len)]
        else:
            if a_len > b_len:
                b_tokens = [deepcopy(EMPTY_TOKEN) for _ in range(delta_len)] + b_tokens
            else:
                a_tokens = [deepcopy(EMPTY_TOKEN) for _ in range(delta_len)] + a_tokens
    assert len(a_tokens) == len(b_tokens)
    for at, bt in zip(a_tokens, b_tokens):
        if at.text == "<empty>":
            diff_seqs.append([at, bt, deepcopy(INSERT_TOKEN)])
        elif bt.text == "<empty>":
            diff_seqs.append([at, bt, deepcopy(DELETE_TOKEN)])
        else:
            diff_seqs.append([at, bt, deepcopy(REPLACE_TOKEN)])
    return diff_seqs


def construct_diff_sequence(a: List[Token], b: List[Token]) -> List[List[Token]]:
    # don't take spaces into consideration when aligning two sequences
    a_text = [t.text[1:] if 'Ġ' in t.text else t.text for t in a]
    b_text = [t.text[1:] if 'Ġ' in t.text else t.text for t in b]
    diff = difflib.SequenceMatcher(None, a_text, b_text)

    diff_seqs = []
    for op, a_i, a_j, b_i, b_j in diff.get_opcodes():
        a_tokens = a[a_i:a_j]
        b_tokens = b[b_i:b_j]
        if op == "delete":
            for at in a_tokens:
                diff_seqs.append([at, deepcopy(EMPTY_TOKEN), deepcopy(DELETE_TOKEN)])
        elif op == "insert":
            for bt in b_tokens:
                diff_seqs.append([deepcopy(EMPTY_TOKEN), bt, deepcopy(INSERT_TOKEN)])
        elif op == "equal":
            for at, bt in zip(a_tokens, b_tokens):
                diff_seqs.append([at, bt, deepcopy(EQUAL_TOKEN)])
        else:
            diff_seqs += _heuristic_replace_match(a_tokens, b_tokens)

    return diff_seqs


if __name__ == '__main__':
    PTM = "microsoft/codebert-base"
    tokenizer = PretrainedTransformerTokenizer(PTM, add_special_tokens=False, max_length=64)

    rem_diff = "if (base_len < off + len || res_sz < len)"
    add_diff = "if (GIT_ADD_SIZET_OVERFLOW(&end, off, len) || base_len < end || res_sz < len)"

    rem_diff = tokenizer.tokenize(rem_diff)
    add_diff = tokenizer.tokenize(add_diff)

    print(len(rem_diff), rem_diff)
    print(len(add_diff), add_diff)

    edit_sequence = construct_diff_sequence(a=rem_diff, b=add_diff)

    rem_diff = []
    add_diff = []
    edit_to_id = {"equal": 1, "delete": 2, "insert": 3, "replace": 4}
    empty_token = "<empty>"
    for rem_token, add_token, edit_token in edit_sequence:
        print(rem_token, add_token, edit_token)

        edit_id = edit_to_id[edit_token.text]
        if rem_token.text != empty_token:
            rem_diff.append(dataclasses.replace(rem_token, type_id=edit_id))
        if add_token.text != empty_token:
            add_diff.append(dataclasses.replace(add_token, type_id=edit_id))

    print(len(rem_diff), [(t.text, t.type_id) for t in rem_diff])
    print(len(add_diff), [(t.text, t.type_id) for t in add_diff])
    