import json
import csv
import sys
import os
import re
csv.field_size_limit(sys.maxsize)


# Search all folders and files containing the keyword
def find_files_with_keyword(path, keyword):
    matching_paths = []

    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            if keyword.lower() in dir_name.lower():
                # print(dir_name)
                matching_paths.append(os.path.join(root, dir_name))

        for file_name in files:
            if keyword.lower() in file_name.lower():
                matching_paths.append(os.path.join(root, file_name))
    
    return matching_paths


# Extract all CWE
def extract_and_unique_cwe_numbers(path):
    matching_paths=find_files_with_keyword(path, "CWE")
    cwe_numbers = set()

    # Extract key figures using regular expressions
    # CWE-123
    pattern = re.compile(r'CWE[\s\-]*(\d{1,4})', re.IGNORECASE)
    for path in matching_paths:
        # print(path)
        match = pattern.search(path)
        if match:
            # print(path)
            cwe_numbers.add(match.group(1))
    # CWE123
    pattern = re.compile(r'CWE(\d{1,4})', re.IGNORECASE)
    for path in matching_paths:
        # print(path)
        match = pattern.search(path)
        if match:
            # print(path)
            cwe_numbers.add(match.group(1))


    def process_and_sort_list(input_list):
        processed_list = sorted(set(element.zfill(4) for element in input_list))

        return processed_list
    
    def count_keyword_in_files(path,keyword):
        count = 0
        for root, dirs, files in os.walk(path):
            for file_name in files:
                # print(file_name)
                if keyword.lower() in file_name.lower():
                    # print(file_name)
                    count += 1
        print(count)
        return count

    # print(list(cwe_numbers))
    CWE_list=process_and_sort_list(list(cwe_numbers))

    # Check if CWE-0000 is present
    if "0000" in CWE_list:
        print("There is CWE-000!")

    # Check if CWE-Other is present
    count = count_keyword_in_files(path,"CWE-Other")
    print(f"There are {count} of CWE-Other files in the folder.")

    print(CWE_list)
    

# Count how many values correspond to a specific key in a csv file
def collect_unique_keyword_values_csv(path, keyword):
    unique_lang_values = set()

    with open(path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            lang_value = row.get(keyword)
            if lang_value:
                unique_lang_values.add(lang_value)

    return list(unique_lang_values)


# Count how many values correspond to a specific key in a json file
def collect_unique_keyword_values_json(path,keyword):
    unique_lang_values = set()

    with open(path, 'r') as file:
        data = json.load(file)
        for record in data:
            # print(record)
            lang_value = record.get(keyword)
            if lang_value:
                # for id in lang_value:
                unique_lang_values.add(lang_value)

    return list(unique_lang_values)


# For Magma dataset
def Magma(path,keyword):
    files_with_keyword = set()

    def search_folders(current_folder):
        for root, dirs, _ in os.walk(current_folder):
            for folder in dirs:
                if keyword in folder.lower():
                    files_with_keyword.add(folder)

    search_folders(path)
    return list(files_with_keyword)

# For Lipp et al.'s dataset
def Lipp(path,key):
    unique_cwe_values = set()

    with open(path, 'r') as file:
        data = json.load(file)
        for record in data:
            # print(data[record].get("functions"))
            cwe_value = data[record].get(key)
            if cwe_value:
                unique_cwe_values.add(cwe_value)
    print(list(unique_cwe_values))
    return list(unique_cwe_values)


# For Detect0day dataset
def Detect0day(folder_path):
    middle_parts = set()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        
        if os.path.isfile(file_path):
            parts = filename.split('.')
            if len(parts) >= 3:
                middle_part = parts[1]+'.'+parts[2]
                middle_parts.add(middle_part)
    print(list(middle_parts))
    return list(middle_parts)


# For Funded dataset
def Funded(folder_path, keyword):
    cwe_zip_files = set()

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".zip") and keyword.lower() in file.lower():
                # Remove the ".zip" extension and add to the set
                file_name_without_extension = os.path.splitext(file)[0]
                cwe_zip_files.add(file_name_without_extension)

    return list(cwe_zip_files)


def main():

    path = ''
    keyword = 'CWE'
    extract_and_unique_cwe_numbers(path, keyword)


if __name__ == "__main__":
    main()