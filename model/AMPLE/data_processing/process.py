import os
import json
import time
import subprocess
import datetime
import signal
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
import shutil

def move_file(files_dir, file_name):
    input_ddir = '/home/nfs/share/backdoor2023/Defect/AMPLE/data_processing'
    output_ddir = files_dir.replace('files', 'graph')
    os.makedirs(output_ddir, exist_ok=True)
    
    input_dir = os.path.join(input_ddir, file_name)
    # print(input_dir)
    shutil.move(input_dir, output_ddir)
    return os.path.join(output_ddir, file_name)

def _process(item):
    # try:
    timeout = 5
    files_dir, file_path = item
    slicer = "bash /home/nfs/share/backdoor2023/Defect/AMPLE/data_processing/joern/slicer.sh " + files_dir + "  " + str(file_path) + "  1 " + "parsed/" + str(file_path)
    
    start0 = datetime.datetime.now()
    process1 = subprocess.Popen(slicer, shell = True)
    while process1.poll() is None:
        time.sleep(0.2)
        end0 = datetime.datetime.now()
        if (end0-start0).seconds > timeout:
            os.kill(process1.pid, signal.SIGKILL)
            os.waitpid(-1, os.WNOHANG)
            return None
    try:
        graph_dir = move_file(files_dir, file_path)
    except:
        return None
    return 'SUCC', os.path.join(graph_dir, 'tmp', file_path, 'edges.csv'), os.path.join(graph_dir, 'tmp', file_path, 'nodes.csv'), file_path

def _set_path_key(item):
    files_dir, file_path = item
    output_ddir = files_dir.replace('files', 'graph')
    graph_dir = os.path.join(output_ddir, file_path)
    return 'SUCC', os.path.join(graph_dir, 'tmp', file_path, 'edges.csv'), os.path.join(graph_dir, 'tmp', file_path, 'nodes.csv'), file_path

def set_path_key(files_dir):
    files = os.listdir(files_dir)
    accnum = defaultdict(int)
    idx2path = defaultdict(lambda: defaultdict(str))

    for file_path in files:
        accnum['TRY'] += 1
        item = _set_path_key((files_dir, file_path))
        flag, edges_path, nodes_path, _file_path = item
        # print(f"file_path = {file_path}")
        # exit(0)
        idx2path[int(file_path.replace('.c', ''))]['edges_path'] = edges_path
        idx2path[int(file_path.replace('.c', ''))]['nodes_path'] = nodes_path
        idx2path[int(file_path.replace('.c', ''))]['file_path'] = os.path.join(files_dir, file_path)
        accnum[flag] += 1
    accnum["RATE"] = accnum['SUCC'] / accnum['TRY']
    print(json.dumps(accnum, indent=4))
    return idx2path

def set_path_key_all(data_dir):
    for dt in ['train', 'test', 'val']:
        with open(os.path.join(data_dir, f"{dt}.json"), 'r') as f:
            objs = json.loads(f.read())
        files_dir = os.path.join(data_dir, "files", dt)
        idx2path = set_path_key(files_dir)
        for obj in objs:
            # obj['label'] = obj['target']
            obj['edges_path'] = idx2path[obj['index']]['edges_path']
            obj['nodes_path'] = idx2path[obj['index']]['nodes_path']
            obj['file_path'] = idx2path[obj['index']]['file_path']
        with open(os.path.join(data_dir, f"{dt}.json"), 'w') as f:
            f.write(json.dumps(objs, indent=4))

def process(files_dir):
    files = os.listdir(files_dir)
    accnum = defaultdict(int)
    idx2path = defaultdict(lambda: defaultdict(str))

    # with Pool() as pool:
    #     for flag, edges_path, nodes_path, file_path in tqdm(pool.imap_unordered(_process, [(files_dir, file_path) for file_path in files]), desc=f"[{len(files)}]"):
    #         idx2path[int(file_path.replace('.c', ''))]['edges_path'] = edges_path
    #         idx2path[int(file_path.replace('.c', ''))]['nodes_path'] = nodes_path
    #         idx2path[int(file_path.replace('.c', ''))]['file_path'] = os.path.join(files_dir, file_path)
    #         accnum[flag] += 1
    
    for file_path in files:
        accnum['TRY'] += 1
        item = _process((files_dir, file_path))
        if item is None:
            continue
        flag, edges_path, nodes_path, _file_path = item
        idx2path[int(file_path.replace('.c', ''))]['edges_path'] = edges_path
        idx2path[int(file_path.replace('.c', ''))]['nodes_path'] = nodes_path
        idx2path[int(file_path.replace('.c', ''))]['file_path'] = os.path.join(files_dir, file_path)
        accnum[flag] += 1
    accnum["RATE"] = accnum['SUCC'] / accnum['TRY']
    print(json.dumps(accnum, indent=4))
    return idx2path

def json_to_files(data_dir):
    for dt in ['train', 'test', 'val']:
        with open(os.path.join(data_dir, f"{dt}.json"), 'r') as f:
            objs = json.loads(f.read())
        files_dir = os.path.join(data_dir, "files", dt)
        os.makedirs(files_dir, exist_ok=True)
        for obj in tqdm(objs, ncols=100, desc=f'[{dt}] split files'):
            if 'index' not in obj: obj['index'] = obj['idx']
            with open(os.path.join(files_dir, f"{obj['index']}.c"), 'w') as f:
                f.write(obj['func'])
        idx2path = process(files_dir)
        for obj in objs:
            # obj['label'] = obj['target']
            obj['edges_path'] = idx2path[obj['index']]['edges_path']
            obj['nodes_path'] = idx2path[obj['index']]['nodes_path']
            obj['file_path'] = idx2path[obj['index']]['file_path']
        with open(os.path.join(data_dir, f"{dt}.json"), 'w') as f:
            f.write(json.dumps(objs, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()
    json_to_files(args.data_dir)
    # set_path_key_all(args.data_dir)

