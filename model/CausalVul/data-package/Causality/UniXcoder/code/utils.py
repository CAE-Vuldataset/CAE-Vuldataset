import os
import sys
import random
import dataclasses
import tree_sitter
from typing import Set
from tree_sitter_languages import get_parser
import pandas as pd


def add_pythonpath(path):
    if path not in sys.path:
        sys.path.append(path)


project_root = os.getcwd().split("Causality")[0] + "Causality"
add_pythonpath(f"{project_root}/NatGen")
print(sys.path)
from src.data_preprocessors.transformations.no_transform import NoTransformation
from src.data_preprocessors.transformations.dead_code_api_inserter import DeadCodeAPIInserter


def get_children(node, fn, reverse=False):
    it = node.children
    if reverse:
        it = reversed(it)
    if isinstance(fn, str):
        fn_str = str(fn)
        fn = lambda n: n.type == fn_str
    return [c for c in it if fn(c)]

def get_child(node, fn, reverse=False):
    return next(iter(get_children(node, fn, reverse=reverse)), None)

def is_type(typename):
    def fn(n):
        return n.type == typename
    return fn

parser = get_parser("c")
def get_api_names(code):
    if isinstance(code, str):
        code = code.encode()
    tree = parser.parse(code)
    # print_tree(tree.root_node)
    return extract_api_calls(tree)

def extract_api_calls(tree):
    api_names = []
    q = [tree.root_node]
    while len(q) > 0:
        n = q.pop()
        if n.type == "call_expression":
            ident = get_child(n, "identifier")
            if ident is not None:
                call_name = ident.text.decode()
                api_names.append(call_name)
        if n.type not in ("string_literal", "char_literal"):
            q.extend(reversed(list(n.children)))
    return list(set(api_names))


lang = "c"
path=f"{project_root}/NatGen/parser/languages.so"
dead_code_inserter = DeadCodeAPIInserter(path, lang)
def insert_dead_code(code, dead_code=None):
    starters = [
        'while ( _i_0 > _i_0 ) { ',
        'if ( _i_7 > _i_7 ) { ', 
        'if ( _i_9 > _i_9 ) { ',
        'while ( false ) { ',
        'while ( _i_7 > _i_7 ) { ',
        'for ( ; false ; ) { ',
        'while ( _i_3 < _i_3 ) { '
    ]
    for dd_cd in dead_code:
        dead_code_wrapper = random.choice(starters)
        dead_code_body = f"{dead_code_wrapper} {dd_cd} (); " + "}"
        code = dead_code_inserter.transform_code(code, dead_code_body)[0]
    return code


def test():
    code = """int main(int argc, char **argv)
    {
        int x = 10 + 100;       // immediate expression - easy to filter out
        int y = 10 + x;         // not immediate but known provenance
        int z = 10 + y + argc;  // unknown provenance
        char *s = (char *)malloc(z + y + x);
        s[10] = 'a';
        int result = (int)s[10];
        printf("%c\n", *s);
        free(s);
        free(s);
        s = 0;
        struct foo* ss;
        ss->bar = x;
        *ss.baz = y;
        memset();
        return result;
    }
    """
    print(get_api_names(code))

def test_dead_code():
    code = """int main(int argc, char **argv)
    {
        int x = 10 + 100;       // immediate expression - easy to filter out
        int y = 10 + x;         // not immediate but known provenance
        int z = 10 + y + argc;  // unknown provenance
        char *s = (char *)malloc(z + y + x);
        s[10] = 'a';
        int result = (int)s[10];
        printf("%c\n", *s);
        free(s);
        free(s);
        s = 0;
        struct foo* ss;
        ss->bar = x;
        *ss.baz = y;
        memset();
        return result;
    }
    """

    dead_code_from = """int main(int argc, char **argv)
    {
        int x = 10 + 100;       // immediate expression - easy to filter out
        int y = 10 + x;         // not immediate but known provenance
        int z = 10 + y + argc;  // unknown provenance
        char *s = (char *)malloc(z + y + x);
        s[10] = 'a';
        int result = (int)s[10];
        printf("%c\n", *s);
        free(s);
        free(s);
        s = 0;
        s += 1;
        s++;
        struct foo* ss;
        ss->bar = x;
        *ss.baz = y;
        calloc(x, y);
        return result;
    }
    """
    api_names = get_api_names(dead_code_from)
    print(insert_dead_code(code, api_names))

# test_dead_code()
