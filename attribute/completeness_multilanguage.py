import re
import json
import time
import io
import sys
import pandas as pd
import csv
csv.field_size_limit(sys.maxsize)


# Remove comments
def remove_comments(text):
    def replacer(match):
        code = match.group(1)
        comment = match.group(2)
        in_quotes = False
        for i, char in enumerate(code):
            if char == '"':
                in_quotes = not in_quotes
            elif char == '/' and i + 1 < len(code) and code[i + 1] == '/' and not in_quotes:
                return code[:i].rstrip()
        # print(comment )
        return code + (comment if in_quotes and comment is not None else '')

    pattern = re.compile(r'^(.*?)(\s*//.*)?$', re.MULTILINE)
    result = pattern.sub(replacer, text)
    
    return result


# Check if the brackets match
def check_brackets_complete(text):

    
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  
    text = text.lstrip()
    text = re.sub(r'(["\']).*?\1', '', text)
    text = remove_comments(text)
    text = re.sub(r'\s+\Z', '', text)

    stack = []
    bracket_pairs = {'{': '}', '[': ']', '(': ')'}
    line_number = 0

    for line in text.splitlines():
        line_number += 1
        for char in line:
            if char in bracket_pairs.keys():
                stack.append((char, line_number))
            elif char in bracket_pairs.values():
                if not stack or bracket_pairs[stack.pop()[0]] != char:
                    return "False"
    
    if stack:
        return "False"
    else:
        return "True"


count = 0
data = []
json_file_path = ''
output_path = ''


with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

for record in data:

    function = record.get('func_before')
    completeness = check_brackets_complete(function)
    if completeness == 'False':
        count += 1
    record['completeness'] = completeness
print(f"The number of incomplete samples is: {count}")
with open(output_path, 'w', encoding='utf-8') as file:
     json.dump(data, file, indent=4)

