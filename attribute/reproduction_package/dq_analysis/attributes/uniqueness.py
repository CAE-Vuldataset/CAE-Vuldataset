"""
Analyse duplicates in a dataset.
Consider near and exact matching.
"""
import sys
import csv
import ast
import gzip
import subprocess
import time
import pandas as pd
from data import Data
from ast import literal_eval
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

if len(sys.argv) < 2:
    print("Usage: python dq_analysis/attributes/uniqueness.py [prepare/measure] <dataset>")
    exit()


def process_row(row):
    return [row[2], [a[0] for a in ast.literal_eval(row[4])]]

def identify_duplicates(dataset):
    """
    Identify near duplicates using the duplicate code detector tool
      from Allamanis (2018)

    IMPORTANT: Requires tokens to be generated first via currency preparation
    """
    csv.field_size_limit(100000000)
    headers = []
    data = []

    start = time.time()

    # Process tokenized files
    with open(f'{path}{dataset}/tokens.csv') as file:
        reader = csv.reader(file)
        headers = next(reader)
        print("1")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_row, reader)
            data = list(results)
            print("2")
    print("3")
    df = pd.DataFrame(data, columns=['filename','tokens'])
    middle = time.time()
    print('Process time: ', middle - start)

    # Output in JSONL format
    df.to_json(f'{path}{dataset}/tokens.jsonl', orient='records', lines=True)
    # Output in GZIP format
    with open(f'{path}{dataset}/tokens.jsonl', 'rb') as src, gzip.open(f'{path}{dataset}/tokens.jsonl.gz', 'wb') as dst:
        dst.writelines(src)

    # Run duplicate detector tool
    p = subprocess.Popen(f"dotnet run DuplicateCodeDetector.csproj --input={path}{dataset}/tokens.jsonl.gz",
                         cwd='near-duplicate-code-detector/DuplicateCodeDetector/', shell=True)
    p.wait()

    # Move the output
    p = subprocess.Popen(f"mv near-duplicate-code-detector/DuplicateCodeDetector/DuplicateCodeDetector.csproj.json {path}{dataset}/unique_clusters.csv", shell=True)
    p.wait()

    end = time.time()
    print('Tool time: ', end - start)


def process_cluster(x, vuln, nonvuln):
    cluster0, cluster1 = [], []
    for id in x:
        if int(id) in nonvuln:
            cluster0.append(int(id))
        if int(id) in vuln:
            cluster1.append(int(id))
    return cluster0, cluster1

def get_duplicate_clusters(dataset, type=3):
    """
    Return the within-class fuzzy duplicates of a dataset,
        using similarity matching.
    """

    # Load the data
    data = Data(dataset).get_dataset()
    vuln = data[data.Vulnerable == 1].UID.tolist()
    nonvuln = data[data.Vulnerable == 0].UID.tolist()

    # Read near duplicate matching output
    if type == 1:
        with open(f'{path}{dataset}/consistent_clusters.csv') as duplicates:
            clusters = literal_eval(duplicates.read())
    elif type == 3:
        with open(f'{path}{dataset}/unique_clusters.csv') as duplicates:
            clusters = literal_eval(duplicates.read())
    
    class_clusters = []

    # Split clusters by class
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda x: process_cluster(x, vuln, nonvuln), clusters)
        for cluster0, cluster1 in results:
            if len(cluster0) > 1:
                class_clusters.append(cluster0)
            if len(cluster1) > 1:
                class_clusters.append(cluster1)
    
    return class_clusters


def count_near_unique(dataset, output_path, type=3):
    """
    Count number of unique files using near duplicate matching,
        performed using the Jacquard Index and implemented via Allamanis (2018)
    """
    print('-'*3 + dataset + ' Type ' + str(type) + '-'*3)
    df = Data(dataset).get_dataset()
    # Get duplicates
    class_clusters = get_duplicate_clusters(dataset, type)
    duplicates = [int(y) for x in class_clusters for y in x[1:]]
    # Get unique
    df['Uniqueness'] = ~df.UID.isin(duplicates)

    unique = df[~df.UID.isin(duplicates)]
    unique = unique.dropna()
    num_unique = len(unique)
    

    print(f"NEAR unique: {num_unique} / {len(df)}")
    print(f"{dataset} Uniqueness = {num_unique / len(df)}")

    df.to_csv(output_path, index=False)
    print(f"Modified dataset saved to {output_path}")


if __name__ == '__main__':
    path =''
    if sys.argv[1] == 'prepare':
        identify_duplicates(sys.argv[2])
    elif sys.argv[1] == 'measure':
        count_near_unique(sys.argv[2], path+sys.argv[2]+'/Uniqneness.csv')
    else:
        print(f"ERROR: Unknown command line argument: \"{sys.argv[1]}\"")
