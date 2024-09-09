import os
import sys
import itertools
import difflib
import codecs
import subprocess
import clang.cindex
from pathlib import Path
from clang.cindex import CursorKind
from multiprocessing import Pool
from unidiff import PatchSet
import traceback
import re
import csv
from datetime import datetime

def tokenize_pre_and_post(tmpfile,path_to_diff):
    try:
        patch = PatchSet.from_filename(tmpfile, encoding='utf-8') 

        pre_version_function = [line[1:] for line in patch[0][0].source] 
        post_version_function = [line[1:] for line in patch[0][0].target]

        pre_version_function_str = ''.join(pre_version_function)
        post_version_function_str = ''.join(post_version_function)

        if ''.join(pre_version_function_str) == ''.join(post_version_function_str):
            return

        # Comments have been removed so add comment tokens for line delimiters
        pre_version_function_str = pre_version_function_str.replace('\n',' //<S2SV>\n ').replace('\\ //<S2SV>\n','\\\n')
        post_version_function_str = post_version_function_str.replace('\n',' //<S2SV>\n').replace('\\ //<S2SV>\n','\\\n')

        index = clang.cindex.Index.create()
        tu_pre = index.parse('tmp.c', unsaved_files=[('tmp.c', pre_version_function_str)])
        tu_post = index.parse('tmp.c', unsaved_files=[('tmp.c', post_version_function_str)])

        pre_version_function_path =  'old.tokens'
        post_version_function_path = 'new.tokens'



        pre_tokens = ""
        post_tokens = ""

        for token in tu_pre.cursor.get_tokens():
            pre_tokens+=repr(token.spelling.replace(' ', '<S2SV_blank>'))[1:-1] + ' '
        for token in tu_post.cursor.get_tokens():
            post_tokens+=repr(token.spelling.replace(' ', '<S2SV_blank>'))[1:-1] + ' '
        if pre_tokens == post_tokens:
            return

        with codecs.open(pre_version_function_path, 'w', 'utf-8') as f:
            f.write(pre_tokens)

        with codecs.open(post_version_function_path, 'w', 'utf-8') as f:
            f.write(post_tokens)
    except Exception as e:
        print("Tokenize error: " + str(e))
        print(traceback.format_exc())
        return

def removeComment(path_to_file,tmpfile):
    result = subprocess.run(["gcc", "-fpreprocessed", "-dD", "-E", "-P", str(path_to_file)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_path = Path(tmpfile)
    try:
        no_comment_file_content = result.stdout.decode('utf-8')
        with codecs.open(output_path, 'w', 'utf-8') as f:
            f.write(no_comment_file_content)
            f.close()
    except Exception as e:
        print("Error removing comments: "+str(e))

def get_func_pair_diff(pre_version_file, post_version_file):
    try:
        index = clang.cindex.Index.create()

        removeComment(pre_version_file,"/tmp/S2SV_pre.c") 
        removeComment(post_version_file,"/tmp/S2SV_post.c")

        pre_version_file_lines = open("/tmp/S2SV_pre.c").read().splitlines(True) 
        post_version_file_lines = open("/tmp/S2SV_post.c").read().splitlines(True)

        pre_version_file_tu = index.parse("/tmp/S2SV_pre.c") 
        post_version_file_tu = index.parse("/tmp/S2SV_post.c")

        pre_version_file_func_decl_cursor = []
        post_version_file_func_decl_cursor = []
        
        for cursor in pre_version_file_tu.cursor.walk_preorder():
            if cursor.kind == CursorKind.FUNCTION_DECL and str(cursor.extent.start.file) == "/tmp/S2SV_pre.c":
                pre_version_file_func_decl_cursor.append(cursor)
        for cursor in post_version_file_tu.cursor.walk_preorder():
            if cursor.kind == CursorKind.FUNCTION_DECL and str(cursor.extent.start.file) == "/tmp/S2SV_post.c":
                post_version_file_func_decl_cursor.append(cursor)

        func_decl_cursor_key = lambda c: (c.spelling, c.location.line)

        pre_version_file_func_decl_cursor.sort(key=func_decl_cursor_key)
        post_version_file_func_decl_cursor.sort(key=func_decl_cursor_key)

        pre_index = 0
        post_index = 0
        pre_max_index = len(pre_version_file_func_decl_cursor)
        post_max_index = len(post_version_file_func_decl_cursor)

    except Exception as e:
        print(traceback.format_exc())
        return

    while(pre_index < pre_max_index):
        try:
            pre_func_decl_cursor = pre_version_file_func_decl_cursor[pre_index]
            pre_func_start_line_number = pre_func_decl_cursor.extent.start.line-1
            pre_func_end_line_number = pre_func_decl_cursor.extent.end.line
            for index in range(post_index, post_max_index):
                post_func_decl_cursor = post_version_file_func_decl_cursor[index]
                post_func_start_line_number = post_func_decl_cursor.extent.start.line-1
                post_func_end_line_number = post_func_decl_cursor.extent.end.line
                if(pre_func_decl_cursor.spelling == post_func_decl_cursor.spelling and
                   pre_version_file_lines[pre_func_end_line_number-1].strip()[-1] == '}' and
                   post_version_file_lines[post_func_end_line_number-1].strip()[-1] == '}'):
                    pre_func_decl_cursor_lines = pre_version_file_lines[pre_func_start_line_number:pre_func_end_line_number]
                    post_func_decl_cursor_lines = post_version_file_lines[post_func_start_line_number:post_func_end_line_number]
                    diff = list(difflib.unified_diff(pre_func_decl_cursor_lines, post_func_decl_cursor_lines, fromfile=pre_func_decl_cursor.spelling+'.function', tofile=pre_func_decl_cursor.spelling+'.function',  n=1000000))
                    if diff:
                        func_decl_diff_file = 'readelf.c__' + pre_func_decl_cursor.spelling + '__' + str(pre_func_decl_cursor.location.line) + '.diff'
                        with codecs.open("/tmp/S2SV_func.diff", 'w', 'utf-8') as f:
                            f.write(''.join(diff))
                        tokenize_pre_and_post("/tmp/S2SV_func.diff",func_decl_diff_file)
                    post_index = index + 1
                    break
            pre_index += 1
        except Exception as e:
            pre_index += 1
            print(traceback.format_exc())
            continue

def get_token_pair_diff(pre_version_file,pre_version_file_str,
                          post_version_file_str, num_tokens):
    try:
        pre_token_per_line = pre_version_file_str.replace(' ','\n') 
        post_token_per_line = post_version_file_str.replace(' ','\n')                 
        diff = list(difflib.unified_diff(pre_token_per_line.splitlines(True), post_token_per_line.splitlines(True), fromfile=str(pre_version_file), tofile='post_version',  n=1000000))
        if diff:
            if num_tokens == 1000: 
                return (pre_version_file_str,post_version_file_str)
            # States:
            #  0: start
            #  1,2: preamble processing 
            #  3: idle (no current delta)
            #  100: gathering delete
            #  101 to 10x: post-delete tokens
            #  200: gathering modify
            #  201 to 20x: post-modify tokens
            #  300: gathering addition
            #  301 to 30x: post-addition tokens
            state = 0;  # Start state
            src = ""
            src_line = ""
            bugtag = False
            tgt = ""
            pre_tokens = ["<S2SV_null>"] * num_tokens
            post_tokens = ["<S2SV_null>"] * num_tokens
            for t in diff:
                t = t.replace('\n','')
                if t.startswith("--- ") or t.startswith("+++ ") or t.startswith("@@ "):
                    if state > 2:
                        print(f'ERROR: preamble line {t} occurred in unexpected location')
                    state += 1 # State will be 3 at start of real tokens
                elif state < 3:
                    print(f'ERROR: token line {t} occurred before preamble done')
                elif t.startswith(" "):
                    if t != " //<S2SV>":
                        src_line += t[1:] + ' '
                    elif bugtag:
                        src += "<S2SV_StartBug> "+src_line+"<S2SV_EndBug> "
                        bugtag = False
                        src_line = ""
                        continue
                    else:
                        src += src_line
                        src_line = ""
                        continue
                    if state == 3: # Continue idle state
                        pre_tokens = pre_tokens[1:num_tokens] + [t[1:]] 
                    elif state % 100 == num_tokens-1:
                        post_tokens = post_tokens[1:num_tokens] + [t[1:]]
                        if state >= 300: # addition
                            tgt += '<S2SV_ModStart> '+' '.join(pre_tokens)+' '+' '.join(new_tokens)+' '
                        elif state >= 200: # modify
                            tgt += '<S2SV_ModStart> '+' '.join(pre_tokens)+' '+' '.join(new_tokens)+' <S2SV_ModEnd> '+' '.join(post_tokens)+' '
                        elif state >= 100: # delete
                            tgt += '<S2SV_ModStart> '+' '.join(pre_tokens)+' <S2SV_ModEnd> '+' '.join(post_tokens)+' '
                        state = 3
                        pre_tokens=post_tokens
                    else:
                        state += 1   # Advance post_token count
                        post_tokens = post_tokens[1:num_tokens] + [t[1:]]
                elif t.startswith("-"):
                    if t != "-//<S2SV>":
                        src_line += t[1:] + ' '
                    elif bugtag:
                        src += "<S2SV_StartBug> "+src_line+"<S2SV_EndBug> "
                        bugtag = False
                        src_line = ""
                        continue
                    else:
                        src += src_line
                        src_line = ""
                        continue
                    if state == 3: # Enter from idle state
                        bugtag=True
                        state = 100 # Assume delete at first
                        new_tokens = []
                    elif state >= 300: # Addition changes to modification
                        new_tokens += post_tokens[num_tokens - (state % 100):num_tokens]
                        state = 200
                    elif state >= 200: # Accumulate any post tokens we may have
                        new_tokens += post_tokens[num_tokens - (state % 100):num_tokens]
                        state = 200
                    elif state > 100: # Post count after delete changes to modification
                        new_tokens += post_tokens[num_tokens - (state % 100):num_tokens]
                        state = 200
                    post_tokens = ["<S2SV_null>"] * num_tokens
                        
                elif t.startswith("+"):
                    if t == "+//<S2SV>":
                        continue
                    if state == 3: # Enter from idle state
                        bugtag=True
                        state = 300 # Assume addition at first
                        new_tokens = [t[1:]]
                    elif state >= 300: 
                        new_tokens += post_tokens[num_tokens - (state % 100):num_tokens]+[t[1:]]
                        if state > 300: # Check if we started accumulating post tokens
                            state = 200 
                    elif state >= 200: # accumulate any post tokens we may have
                        new_tokens += post_tokens[num_tokens - (state % 100):num_tokens]+[t[1:]]
                        state = 200 # Modified
                    elif state >= 100: # delete changes to modify
                        new_tokens += post_tokens[num_tokens - (state % 100):num_tokens]+[t[1:]]
                        state = 200 # Change to modified
                    post_tokens = ["<S2SV_null>"] * num_tokens
                    
            # Fix end-of-file post tokens by putting <S2SV_null> at end
            post_tokens = post_tokens[num_tokens-(state % 100):num_tokens]+ \
                          post_tokens[0:num_tokens-(state % 100)]
            
            if state >= 300: # addition
                tgt += '<S2SV_ModStart> '+' '.join(pre_tokens)+' '+' '.join(new_tokens)+' '
            elif state >= 200: # modify
                tgt += '<S2SV_ModStart> '+' '.join(pre_tokens)+' '+' '.join(new_tokens)+' <S2SV_ModEnd> '+' '.join(post_tokens)+' '
            elif state >= 100: # delete
                tgt += '<S2SV_ModStart> '+' '.join(pre_tokens)+' <S2SV_ModEnd> '+' '.join(post_tokens)+' '
            if not tgt:
                print(f'ERROR: {pre_version_file_str} found no target changes in {diff}')
            return (src.strip(),tgt.strip())
        else:
            print(f'No diff found for {pre_version_file}')
            sys.exit(2)
    except Exception as e:
        print("Get token pair fail: "+str(e))


def main(argv):
    pre_version_file = argv[1]
    # print(pre_version_file)
    post_version_file = argv[2]
    get_func_pair_diff(pre_version_file, post_version_file)
    print("extract success!")
    num_tokens = 3
    cwe_id = argv[3]

    cve_id = argv[4]
    base_path = argv[5]
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    base_name = os.path.basename(argv[1])
    base = base_name.split('#')
    index = base[0]
    print(base_name)
    

    pre_version_file = 'old.tokens'
    post_version_file = 'new.tokens'

    pre_version_file_str = open(pre_version_file).read()
    post_version_file_str = open(post_version_file).read()
    if pre_version_file_str.endswith(' '):
        pre_version_file_str=pre_version_file_str[:-1]
    if post_version_file_str.endswith(' '):
        post_version_file_str=post_version_file_str[:-1]
    (src, tgt) = get_token_pair_diff(pre_version_file, pre_version_file_str, post_version_file_str, num_tokens)
    src_lines = cwe_id +' '+src+'\n'
    tgt_lines = tgt+'\n'

    filename = "#".join(base[:-2]) + "#.csv"
    # filename = base_name + '#.csv'
    # fields = ["cwe_id", "source", "target", "project_and_commit_id", "cve_id", "original_address", "time"]
    fields = ['index','cve_id', 'cwe_id', 'source', 'target']

    with open(base_path+filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        writer.writerow({'index': index,'cve_id': cve_id, 'cwe_id': cwe_id, 'source': src_lines, 'target': tgt_lines})  
        # writer.writerow({'cwe_id': cwe_id, 'source': src_lines, 'target': tgt_lines, "project_and_commit_id": project, 'cve_id': cve_id, "original_address": address, "time": time})    
       

if __name__=="__main__":
    main(sys.argv)
