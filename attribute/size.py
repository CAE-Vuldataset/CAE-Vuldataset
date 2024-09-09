import json
import csv
import pickle
import re
import os
import pyarrow.parquet as pq
import sys
csv.field_size_limit(sys.maxsize)


# Count the number of records in a json file
def count_records_json(path: str) -> int:
    with open(path) as f:
        data = json.load(f)
        count = len(data)

    return count


# Count the number of records in a csv file
def count_records_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader) - 1  # Subtract 1 to exclude header
    return row_count


# Count the number of files in the current folder
def count_current_file(directory):
    try:
        files = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]

        print(len(files))
        return len(files)
    except Exception as e:
        return f"Exception: {e}"


# Count the number of files in all folders
def count_all_file(directory):
    try:
        file_count = 0
        for root, dirs, files in os.walk(directory):
            file_count += len(files)

        print(file_count)
        return file_count
    except Exception as e:
        return f"Exception: {e}"


# Counts the number of files in the folder containing the keyword
def count_keyword_file(path, keyword):
    count = 0
    c = 0
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            if keyword in dir_name:
                bad_folder_path = os.path.join(root, dir_name)
                c += 1
                count += len(os.listdir(bad_folder_path))
    print(c)
    return count


# For D2A dataset
def D2A(path):
    count=0
    try:
        with open(path, 'rb') as pickle_file:
            try:
                while True:
                    record = pickle.load(pickle_file)
                    count+=1
            except EOFError:
                pass  
        print(f'Successfully printed all records from {path}')
        return count
    except Exception as e:
        print(f'Error: {e}')


# For Le et al.'s dataset, convert parquet file to json file
def convert_parquet_to_json(parquet_file_path, json_file_path):
    try:
        table = pq.read_table(parquet_file_path)
        dataframe = table.to_pandas()
        dataframe.to_json(json_file_path, orient='records', lines=True)

        print(f'Successfully converted {parquet_file_path} to {json_file_path}')
    except Exception as e:
        print(f'Error: {e}')


def main():

    path = ''
    count_records_json(path)


if __name__ == "__main__":
    main()
