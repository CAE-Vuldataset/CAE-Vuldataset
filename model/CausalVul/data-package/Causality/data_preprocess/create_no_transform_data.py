import os
import sys
import json
import argparse
from tqdm import tqdm

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)
# project_root = os.getcwd().split("Causality")[0]
project_root = '/home/nfs/share/backdoor2023/Defect/CausalVul/data-package/Causality'
add_path(f"{project_root}/Causality")
add_path(f"{project_root}/NatGen")
sys.path.append('/home/nfs/share/backdoor2023/Defect/CausalVul/data-package/Causality/NatGen')

from src.data_preprocessors import *
language = 'c'
# parser_path = project_root + '/NatGen/parser/languages.so'
parser_path = '/home/nfs/share/backdoor2023/Defect/CausalVul/data-package/Causality/NatGen/parser/languages.so'

def read_jsonl(data_path):
    f = open(data_path, 'r')
    # data = [json.loads(ex) for ex in f]
    data = json.loads(f.read())
    f.close()
    return data

def write_jsonl(data_path, data):
    f = open(data_path, 'w')
    f.write('\n'.join([json.dumps(ex) for ex in data]))
    f.close()

def get_no_transform(data, part):
    no_transform = NoTransformation(parser_path, language)
    for i in tqdm(range(len(data)), ncols=100, desc=part, mininterval=60):
        data[i]['func'] = no_transform.transform_code(data[i]['func'])[0]
    return data

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Which data to run?")
    arg_parser.add_argument('--dataset_dir', type=str, default='')
    args = arg_parser.parse_args()

    if len(args.dataset_dir) == 0: 
        exit(0)

    for part in ['train', 'train_sp', 'test', 'val']:
    # for part in ['train_sp']: 
        data = read_jsonl(f"{args.dataset_dir}/{part}.json")
        final_data = get_no_transform(data, part)
        if part == 'train_sp':
            write_jsonl(f"{args.dataset_dir}/train_no_transform_with_xp.jsonl", final_data)
        else:
            write_jsonl(f"{args.dataset_dir}/{part}_no_transform.jsonl", final_data)
