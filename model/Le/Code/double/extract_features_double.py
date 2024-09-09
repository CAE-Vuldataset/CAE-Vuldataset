import time
import pandas as pd
import numpy as np
import sys
from collections import Counter
import ast


def filter_code(vuln_code):

    code_lines = []

    for code_line in vuln_code:
        if "//" in code_line:
            code_line = code_line[: code_line.find("//")]
        elif "/*" in code_line and "*/" in code_line:
            start_comment_index = code_line.find("/*")
            end_comment_index = code_line.find("*/")

            code_line = (
                code_line[:start_comment_index] + code_line[end_comment_index + 2 :]
            )

        code_lines.append(code_line)

    return "\n".join(code_lines)


def extract_vuln_code(row):
    code = np.asarray(row["code_before"].splitlines())
    vuln_lines = np.asarray(row["vuln_lines"]) - 1

    if len(vuln_lines) == 0:
        return ""

    vuln_code = code[vuln_lines]

    return filter_code(vuln_code)


def extract_clean_code(row, granularity="file", output="code"):
    if granularity == "file":
        code = np.asarray(row["code_before"].splitlines())

        if output == "code":
            code_lines = (
                np.asarray(list(set(row["vuln_lines"]) - set(row["noisy_lines"]))) - 1
            )
        elif output == "context":
            code_lines = (
                np.asarray(
                    list(
                        set(list(range(1, len(code) + 1)))
                        - set(row["noisy_lines"])
                        - set(row["vuln_lines"])
                    )
                )
                - 1
            )
    elif granularity == "method":
        code = np.asarray(row["code"].splitlines())
        start_line = int(row["start_line"])

        if output == "code":
            code_lines = (
                np.asarray(
                    list(set(row["method_vuln_lines"]) - set(row["noisy_lines"]))
                )
                - start_line
            )
        elif output == "context":
            method_lines = np.asarray(list(range(len(code)))) + start_line
            method_lines = method_lines.tolist()
            code_lines = (
                np.asarray(
                    list(
                        set(method_lines)
                        - set(row["noisy_lines"])
                        - set(row["method_vuln_lines"])
                    )
                )
                - start_line
            )

    if len(code_lines) == 0:
        return ""

    code = code[code_lines]

    return filter_code(code)


def extract_surrounding_context_code(row, granularity):
    if granularity == "file":
        code = np.asarray(row["code_before"].splitlines())

        vuln_lines = (
            np.asarray(
                list(
                    set(row["surrounding_context"])
                    - set(row["vuln_lines"])
                    - set(row["noisy_lines"])
                )
            )
            - 1
        )

    elif granularity == "method":
        code = np.asarray(row["code"].splitlines())
        start_line = int(row["start_line"])
        vuln_lines = (
            np.asarray(
                list(
                    set(row["surrounding_context"])
                    - set(row["method_vuln_lines"])
                    - set(row["noisy_lines"])
                )
            )
            - start_line
        )

    if len(vuln_lines) == 0:
        return ""

    vuln_code = code[vuln_lines]

    return filter_code(vuln_code)


def extract_context_code_file(row, context_dict):
    if len(context_dict[row["file_change_id"]]) == 0:
        return ""

    code = np.asarray(row["code_before"].splitlines())

    vuln_lines = (
        np.asarray(
            list(
                set(context_dict[row["file_change_id"]])
                - set(row["vuln_lines"])
                - set(row["noisy_lines"])
            )
        )
        - 1
    )

    if len(vuln_lines) == 0:
        return ""

    vuln_code = code[vuln_lines]

    return filter_code(vuln_code)


def extract_method_vuln_code(row):
    code = np.asarray(row["code"].splitlines())
    start_line = int(row["start_line"])
    vuln_lines = np.asarray(row["method_vuln_lines"]) - start_line

    if len(vuln_lines) == 0:
        return ""

    vuln_code = code[vuln_lines]

    return filter_code(vuln_code)


# def extract_method_vuln_code(row):
# 	code = np.asarray(row['code'].splitlines())
# 	start_line = int(row['start_line'])
# 	print(start_line)
# 	print(row['method_vuln_lines'])
# 	print(row['method_vuln_lines'].replace('b',''))
# 	vuln_lines = [int(line) - start_line for line in row['method_vuln_lines']]

# 	if len(vuln_lines) == 0:
# 		return ''

# 	vuln_code = code[vuln_lines]

# 	return filter_code(vuln_code)


def extract_context_code_method(row):
    code = np.asarray(row["code"].splitlines())
    start_line = int(row["start_line"])
    context_lines = (
        np.asarray(
            list(
                set(row["context_lines"])
                - set(row["method_vuln_lines"])
                - set(row["noisy_lines"])
            )
        )
        - start_line
    )

    if len(context_lines) == 0:
        return ""

    context_code = code[context_lines]

    return filter_code(context_code)


def extract_left_right_context(vuln_lines, context_lines):
    start_line, end_line = vuln_lines[0], vuln_lines[-1]
    context_lines = np.asarray(list(set(context_lines) - set(vuln_lines)))

    return (
        context_lines[context_lines < end_line],
        context_lines[context_lines > start_line],
    )


def create_fold(df, key, folds):
    sizes = []
    fold_sum = 0

    if type(folds) is list:

        for i in range(len(folds)):
            if i == len(folds) - 1:
                sizes.append(len(df) - 1)
            else:
                sizes.append(int(len(df) * folds[i]) + fold_sum)
                fold_sum += int(len(df) * folds[i])
    else:

        # print("Here")

        size_per_fold = int(len(df) / folds)

        for i in range(folds):
            if i == folds - 1:
                sizes.append(len(df) - 1)
            else:
                sizes.append(size_per_fold + fold_sum)
                fold_sum += size_per_fold

    tmp_df = df.copy()
    tmp_df["row_index"] = list(range(len(df)))
    # tmp_df['row_index'] = tmp_df['row_index'].astype(int)
    tmp_df = tmp_df.rename(columns={key: "key"})

    tmp_df["fold"] = 0

    for i, size in enumerate(sizes):

        if i == 0:
            start_index = 0
        else:
            start_index = sizes[i - 1] + 1

        end_index = size

        # print(start_index, end_index, i)

        tmp_df.loc[
            (start_index <= tmp_df["row_index"]) & (tmp_df["row_index"] <= end_index),
            "fold",
        ] = i

    fold_map = tmp_df[["key", "fold"]].copy()
    fold_map["key"] = fold_map["key"].astype(str)
    fold_map["fold"] = fold_map["fold"].astype(int)

    # print(len(fold_map), fold_map.columns, fold_map.dtypes)
    # print(fold_map.head(10))

    return fold_map


def change_whole_method(row):
    start_line, end_line = int(row["start_line"]), int(row["end_line"])
    cur_vuln_lines = row["method_vuln_lines"]
    noisy_lines = row["noisy_lines"]

    code = row["code"].splitlines()
    line_no = 0

    while not ")" in code[line_no].strip():
        # print(code[line_no])
        line_no += 1

    # print(code[line_no])
    # print(line_no)

    method_lines = list(range(start_line, end_line + 1))

    unchanged_lines = list(
        set(method_lines) - (set(cur_vuln_lines).union(set(noisy_lines)))
    )

    # if len(cur_vuln_lines) >= (end_line - start_line + 1 - 2):
    if len(unchanged_lines) <= 2 + line_no:
        return "True"

    return "False"


def extract_context_scope(row, granularity, scope_size=5):
    context_lines = []

    if granularity == "file":
        vuln_lines = row["vuln_lines"]
        nloc = row["nloc_new"]
        context_lines = vuln_lines.copy()

        for line in vuln_lines:
            start_scope = line - scope_size

            if start_scope < 1:
                start_scope = 1

            end_scope = line + scope_size

            if end_scope > nloc:
                end_scope = nloc

            context_lines.extend(
                [line_index for line_index in range(start_scope, end_scope + 1)]
            )

    elif granularity == "method":
        vuln_lines = row["method_vuln_lines"]
        context_lines = vuln_lines.copy()
        start_line = int(row["start_line"])
        end_line = int(row["end_line"])

        for line in vuln_lines:
            start_scope = line - scope_size

            if start_scope < start_line:
                start_scope = start_line

            end_scope = line + scope_size

            if end_scope > end_line:
                end_scope = end_line

            context_lines.extend(
                [line_index for line_index in range(start_scope, end_scope + 1)]
            )

    return sorted(list(set(context_lines)))


def method_size(row):
    code = np.asarray(row["code"].splitlines())
    start_line = int(row["start_line"])
    method_lines = np.asarray(list(range(len(code)))) + start_line
    method_lines = method_lines.tolist()
    code_lines = (
        np.asarray(list(set(method_lines) - set(row["noisy_lines"]))) - start_line
    )
    return len(code_lines)


def extract_context_code_method_wo_vuln_lines(row):

    start_line = int(row["start_line"])
    context_lines = (
        np.asarray(
            list(
                set(row["context_lines"])
                - set(row["method_vuln_lines"])
                - set(row["noisy_lines"])
            )
        )
        - start_line
    )

    if len(context_lines) == 0:
        return []

    return context_lines


def convert_to_numeric_array(row):
    row["context_lines"] = ast.literal_eval(row["context_lines"])
    row["method_vuln_lines"] = ast.literal_eval(row["method_vuln_lines"])
    row["noisy_lines"] = ast.literal_eval(row["noisy_lines"])
    return row


def convert_to_numeric_array(row):
    
    row["context_lines"] = ast.literal_eval(row["context_lines"])
    row["context_lines"] = row["context_lines"].decode("utf-8")
    row["context_lines"] = ast.literal_eval(row["context_lines"])


    row["method_vuln_lines"] = ast.literal_eval(row["method_vuln_lines"])
    row["method_vuln_lines"] = row["method_vuln_lines"].decode("utf-8")
    row["method_vuln_lines"] = ast.literal_eval(row["method_vuln_lines"])

    row["noisy_lines"] = ast.literal_eval(row["noisy_lines"])
    row["noisy_lines"] = row["noisy_lines"].decode("utf-8")
    row["noisy_lines"] = ast.literal_eval(row["noisy_lines"])
    return row


# df_method = pd.read_parquet("Data/combined_df_method.parquet")
path = "/home/nfs/m2023-zxh/share/DataEval/Le/data/completeness/0.7/"
df_method = pd.read_csv(path + "/train.csv")
df_method = df_method.apply(convert_to_numeric_array, axis=1)



df_method[["author_date", "committer_date"]] = df_method[
    ["author_date", "committer_date"]
].apply(lambda r: pd.to_datetime(r, infer_datetime_format=True))

# Filter the methods that change the whole method body

df_method["nloc_method"] = df_method[["code", "start_line", "noisy_lines"]].apply(
    lambda r: method_size(r), axis=1
)
# print(df_method['method_vuln_lines'])
# print(df_method['nloc_method'])
df_method["method_ratio"] = df_method[["method_vuln_lines", "nloc_method"]].apply(
    lambda r: len(r["method_vuln_lines"]) / r["nloc_method"] * 1.0, axis=1
)

df_method["whole_method_change"] = df_method[
    ["code", "start_line", "end_line", "method_vuln_lines", "noisy_lines"]
].apply(lambda r: change_whole_method(r), axis=1)

# Program slicing w/o vuln lines
df_method["ps_only"] = df_method[
    ["code", "context_lines", "start_line", "method_vuln_lines", "noisy_lines"]
].apply(lambda r: extract_context_code_method_wo_vuln_lines(r), axis=1)

df_method["ps_only_ratio"] = df_method[["ps_only", "nloc_method"]].apply(
    lambda r: len(r["ps_only"]) / r["nloc_method"], axis=1
)

df_method = df_method[
    (df_method["method_ratio"] < 1.0)
    & (df_method["whole_method_change"] == "False")
    & (df_method["ps_only_ratio"] > 0)
]


df_method = df_method.iloc[
    np.random.RandomState(42).permutation(len(df_method))
].reset_index(drop=True)

n_folds = 10
# n_folds = [0.8, 0.1, 0.1]

method_map = create_fold(df_method, "method_change_id", folds=n_folds)
# method_map = create_fold(df_method, 'index', folds=n_folds)

method_map.to_csv(path + "/method_map.csv", index=False)

cvss_cols = [
    "cvss2_confidentiality_impact",
    "cvss2_integrity_impact",
    "cvss2_availability_impact",
    "cvss2_access_vector",
    "cvss2_access_complexity",
    "cvss2_authentication",
    "severity",
]

print("CVSS metric distrbution in each fold in method")
df_method["method_change_id"] = df_method["method_change_id"].astype(str)

# print(method_map)

folds = method_map["fold"].unique()

for i, fold in enumerate(folds):
    cur_keys = method_map[method_map["fold"] == fold]["key"].values
    # print(cur_keys)
    sel_df = df_method[df_method["method_change_id"].isin(cur_keys)].copy()

    print("Fold:", fold)

    if i == len(folds) - 1:
        print(sel_df[["hash", "filename", "name"]])
        print(len(sel_df["hash"].unique()))

    for cvss_col in cvss_cols:
        print(cvss_col)
        print(Counter(sel_df[cvss_col]))


# Vuln lines with context in methods
selected_cols = [
    "method_change_id",
    "code",
    "context_lines",
    "start_line",
    "method_vuln_lines",
    "noisy_lines",
]
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()

df_tmp["vuln_code"] = df_tmp[["code", "method_vuln_lines", "start_line"]].apply(
    lambda r: extract_method_vuln_code(r), axis=1
)

df_tmp["context_code"] = df_tmp[
    ["code", "context_lines", "start_line", "method_vuln_lines", "noisy_lines"]
].apply(lambda r: extract_context_code_method(r), axis=1)
df_tmp = df_tmp.drop(
    columns=["code", "context_lines", "start_line", "method_vuln_lines", "noisy_lines"]
)
df_tmp = df_tmp.rename(
    columns={"vuln_code": "code", "context_code": "context", "method_change_id": "key"}
).reset_index(drop=True)
print(len(df_tmp), df_tmp.columns)
df_tmp.to_parquet(path + "/method_lines_with_all_context_double.parquet", index=False)
