import json
import csv
import sys
import os
csv.field_size_limit(sys.maxsize)

# Count the number of records in the json file that satisfy the conditions.
def count_condition_items_json(path, key, value) -> int:
    with open(path, "r") as f:
        data = json.load(f)

    count = 0
    for item in data:
        if item[key] == value:
            count += 1
    
    return count

# Count the number of records in the csv file that satisfy the conditions.
def count_condition_items_csv(path, key: int, value) -> int:
    count = 0
    with open(path, "r") as f:
        reader = csv.reader(f)
        # headers = next(reader, None)
        for row in reader:
            # print(row[title])
            if row[key] == value:
                count += 1
    
    return count

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
    

def main():

    path = ''
    key = 'category'
    value = 'security'
    count_condition_items_json(path, key, value)


if __name__ == "__main__":
    main()