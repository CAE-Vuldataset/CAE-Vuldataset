import json
import pandas as pd
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
import copy
import os
from pandas.core.frame import DataFrame
from sympy import subsets
from unidiff import PatchSet
import copy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import KFold, StratifiedKFold


def csv_to_json(file):
    # set index_col=False for View-1000
    df = pd.read_csv(DATA_PATH + f"{file}.csv", index_col=False)
    df_dict = df.to_dict(orient="records")

    with open(DATA_PATH + f"{file}.json", 'w') as f:
        json.dump(df_dict, f, indent=4)


def build_CWE_tree(view=1000):
    CWE = json.load(open(DATA_PATH + f'{view}.json', 'r'))
    CWE_tree = dict()
    for cwe in CWE:
        cwe_id = int(cwe["CWE-ID"])
        CWE_tree[cwe_id] = cwe
        CWE_tree[cwe_id]["father"] = list()
        CWE_tree[cwe_id]["children"] = list()
        CWE_tree[cwe_id]["peer"] = list()  # PeerOf & CanAlsobe
        CWE_tree[cwe_id]["relate"] = list()  # CanPrecede & CanFollow & Requires

    for cwe_id, cwe in CWE_tree.items():
        if cwe['Related Weaknesses'] == "null":
            continue
        relations = cwe['Related Weaknesses'].split("::")
        for r in relations:
            if f"VIEW ID:{view}" in r:
                rr = r.split(":")
                target_id = int(rr[3])
                if "ChildOf" in rr:
                    cwe["father"].append(target_id)
                    CWE_tree[target_id]["children"].append(cwe_id)
                elif "PeerOf" in rr or "CanAlsoBe" in rr:
                    cwe["peer"].append(target_id)
                    CWE_tree[target_id]["peer"].append(cwe_id)
                elif "CanPrecede" in rr or "Requires" in rr:
                    cwe["relate"].append(target_id)
                    CWE_tree[target_id]["relate"].append(cwe_id)
    
    with open(DATA_PATH + f"cwe_tree_{view}.json", 'w') as f:
        json.dump(CWE_tree, f, indent=4)


def generate_dataset_hunk(directory):
    dir_path = f"{directory}/"

    lang2extent = json.load(open(DATA_PATH + "language_extention.json", 'r'))
    extent2lang = dict()
    extent_not_lang = list()
    for item in lang2extent:
        extensions = item.get("extensions") or list()
        if item["type"] != "programming":
            extent_not_lang.extend(extensions)
            continue
        for ext in extensions:
            extent2lang[ext] = item["name"]

    dataset = list()
    composed_patch = 0
    
    all_files = sorted(os.listdir(dir_path))
    for file in all_files:
        sample_dict = dict()
        info = file.split('.')
        repo = f"{info[1]}/{'.'.join(info[2:-2])}"
        commit_id = info[-2]
        sample_dict["cve_list"] = info[0]
        sample_dict["repo"] = repo
        sample_dict["commit_id"] = commit_id
        sample_dict["user"] = None
        sample_dict["commit_date"] = None
        sample_dict["msg"] = None

        # get commit user, commit_date and msg info
        # commit = get_by_oauth2(f"https://api.github.com/repos/{repo}/git/commits/{commit_id}")
        # if commit is not None:
        #     sample_dict["user"] = commit["committer"]["name"]
        #     sample_dict["commit_date"] = commit["committer"]["date"]
        #     sample_dict["msg"] = commit["message"]

        try:
            patch = PatchSet.from_filename(dir_path + file, encoding="utf-8")
        except Exception as e:
            print(file)
            print(e)
            continue
        sample_dict["Total_LOC_REM"] = patch.removed
        sample_dict["Total_LOC_ADD"] = patch.added
        sample_dict["Total_LOC_MOD"] = sample_dict["Total_LOC_ADD"] + sample_dict["Total_LOC_REM"]
        sample_dict["Total_NUM_FILE"] = len(set(f.path for f in patch))
        if len(set(f.path for f in patch)) < len(patch):
            composed_patch += 1

        sample_dict["Total_NUM_HUNK"] = sum([len(f) for f in patch])

        for f in (patch.modified_files + patch.removed_files + patch.added_files):
            if len(f) == 0:
                if not f.is_rename:
                    raise ValueError("0 change, not renamed")
                continue

            if f.is_binary_file:
                continue

            if '.' not in f.path:
                continue

            extension = '.' + f.path.split('.')[-1]
            if extension not in extent2lang:
                continue

            sample_dict["file_name"] = f.path
            
            if f.is_modified_file:
                sample_dict["file_type"] = "modified"
            elif f.is_added_file:
                sample_dict["file_type"] = "added"
            elif f.is_removed_file:
                sample_dict["file_type"] = "removed"
            
            sample_dict["PL"] = extent2lang[extension]
            
            sample_dict["LOC_REM"] = f.removed
            sample_dict["LOC_ADD"] = f.added
            sample_dict["LOC_MOD"] = sample_dict["LOC_ADD"] + sample_dict["LOC_REM"]
            
            sample_dict["NUM_HUNK"] = len(f)

            rem_lines = list()
            add_lines = list()

            for hunk in f:
                # hunk-level
                l_rem = ''.join([str(l)[1:] for l in hunk.source_lines() if not l.is_context])
                l_add = ''.join([str(l)[1:] for l in hunk.target_lines() if not l.is_context])  
                for s in ['\r\n', '\r', '\n']:
                    # preprocess code exactly the same as the CodeBERT
                    l_rem = l_rem.replace(s, ' ')
                    l_add = l_add.replace(s, ' ')
                l_rem = ' '.join(l_rem.split())
                l_add = ' '.join(l_add.split())
                
                rem_lines.append(l_rem)
                add_lines.append(l_add)

            # hunk-level
            sample_dict["REM_DIFF"] = rem_lines
            sample_dict["ADD_DIFF"] = add_lines

            dataset.append(copy.deepcopy(sample_dict))
    
    print("#total files:", len(dataset))
    print("#composed patches:", composed_patch)
    with open(DATA_PATH + "dataset_hunk.json", 'w', encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)


def agg_cve(df):
    # remove duplicate cves
    return set(df["cve_list"].to_list())


def clean_dataset(file):
    cve_dict = json.load(open(DATA_PATH + "cve_data.json", 'r'))
    cwe_tree = json.load(open(DATA_PATH + "cwe_tree.json", 'r'))
    
    sample_file = json.load(open(DATA_PATH + f"{file}.json", 'r'))

    sample_file = DataFrame(sample_file)
    print("file-level:", len(sample_file))

    commit_cve_dict = sample_file[["commit_id", "cve_list"]].groupby("commit_id").apply(agg_cve).to_dict()  # checked
    commit_cwe_dict = dict()

    num_commit_with_n_cve = 0
    num_commit_with_n_cwe = 0
    commit_with_n_cwe = []
    for commit, cve_set in commit_cve_dict.items():
        cwe_set = set([cwe for cve in cve_set for cwe in cve_dict[cve]["cwe_id"]])
        valid_cwe_set = []
        for cwe in cwe_set:
            try:
                id_ = cwe.split("-")[1]
                cwe_tree[id_]
            except BaseException:
                # print(f"MISS [{id_}]")  # CWE-264 CWE-19 CWE-320 CWE-361 CWE-310 CWE-417
                continue
            valid_cwe_set.append(cwe)
        
        if len(cve_set) > 1:
            num_commit_with_n_cve += 1
        if len(valid_cwe_set) > 1:
            num_commit_with_n_cwe += 1
            # print(f"[{commit}] CVE:{cve_set} CWE:{valid_cwe_set}")
            commit_with_n_cwe.append({"commit": commit, "CVE": list(cve_set), "CWE": list(valid_cwe_set)})

        commit_cve_dict[commit] = ','.join(cve_set)
        commit_cwe_dict[commit] = ','.join(valid_cwe_set)

    print("total commit:", len(commit_cve_dict))
    print("commit with n cve:", num_commit_with_n_cve)
    print("commit with n cwe:", num_commit_with_n_cwe)

    # remove duplicate commit
    sample_file = sample_file.drop_duplicates(subset=["commit_id", "file_name"], keep="first")
    print("remove duplicate:", len(sample_file))
    print("#commit:", len(sample_file.drop_duplicates(subset=["commit_id"], keep="first")))
    del sample_file["cve_list"]
    sample_file["cve_list"] = sample_file["commit_id"].map(commit_cve_dict)
    sample_file["cwe_list"] = sample_file["commit_id"].map(commit_cwe_dict)

    # remove invalid cwe
    sample_file = sample_file[sample_file["cwe_list"] != ""]
    print("remove commit with invalid cwe:", len(sample_file))
    print("#commit:", len(sample_file.drop_duplicates(subset=["commit_id"], keep="first")))

    # remove not commonly used PL
    # ATTENTION: perform at the file-level
    # print("PL distribution")
    PL_distribution = sample_file["PL"].value_counts().to_dict()  # file-level
    # with open("hunk-level_PL_distribution.json", 'w') as f:
    #     json.dump(PL_distribution, f, indent=4)
    # print(PL_distribution)
    selected_PL = ["PHP", "C", "Java", "JavaScript", "Go", "Python", "Objective-C", "C++", "Ruby", "TypeScript"]
    sample_file["is_selected_PL"] = sample_file["PL"].apply(is_selected, selected_list=selected_PL)
    sample_file = sample_file[sample_file.is_selected_PL == True]
    del sample_file["is_selected_PL"]
    print("remove other PL:", len(sample_file))
    print("#commit:", len(sample_file.drop_duplicates(subset=["commit_id"], keep="first")))

    # remove large commits (follow DeepCVA)
    # ATTENTION: perform at the commit-level
    thres_file_num = 100
    thres_LOC = 10000
    print(f"thres_file_num: {thres_file_num}, thres_LOC: {thres_LOC}")
    sample_file = sample_file[(sample_file.Total_NUM_FILE <= thres_file_num) & (sample_file.Total_LOC_MOD <= thres_LOC)]
    print("remove large commits:", len(sample_file))
    print("#commit:", len(sample_file.drop_duplicates(subset=["commit_id"], keep="first")))

    # reordering
    sample_file = sample_file[["cve_list", "cwe_list", "repo", "commit_id", "user", "commit_date", "msg", "Total_LOC_REM", "Total_LOC_ADD", "Total_LOC_MOD", "Total_NUM_FILE", "Total_NUM_HUNK", 
                            "file_name", "file_type", "PL", "LOC_REM", "LOC_ADD", "LOC_MOD", "NUM_HUNK", "REM_DIFF", "ADD_DIFF"]]

    print("total files:", len(sample_file))
    sample_commit = sample_file.drop_duplicates(subset="commit_id", keep="first").copy()
    print("total commit:", len(sample_commit))

    sample_file = sample_file.to_dict(orient="records")
    with open(DATA_PATH + f"dataset_cleaned.json", 'w') as f:
        json.dump(sample_file, f, indent=4)


def DFS(cwe_id, tmp_path, complete_path):
    # depth-first recursion
    global cwe_tree

    # stop condition
    if cwe_tree[cwe_id]["Weakness Abstraction"] == "Pillar":
        tmp_path.append(f"CWE-{cwe_id}")
        path = copy.deepcopy(tmp_path)
        path.reverse()
        complete_path.append(path)  # record the reversed path
        tmp_path.pop(-1)
        return
    
    # access the current cwe_id
    tmp_path.append(f"CWE-{cwe_id}")
    for father in cwe_tree[cwe_id]["father"]:
        DFS(str(father), tmp_path, complete_path)
    
    # leave the current cwe_id
    tmp_path.pop(-1)


def generate_CWE_mapping(file):
    global cwe_tree
    cwe_tree = json.load(open(DATA_PATH + "cwe_tree.json", 'r'))
    
    dataset = json.load(open(DATA_PATH + f"{file}.json", 'r'))
    dataset = DataFrame(dataset)

    dataset = dataset.drop_duplicates(subset="commit_id")

    cwe_list = dataset["cwe_list"].to_list()
    cwe_list = [[_.strip() for _ in cwe_.split(',')] for cwe_ in cwe_list]
    cwe_list = list(set([_ for cwe_ in cwe_list for _ in cwe_]))  # double loop, including all cwes in dataset
    cwe_list.sort()

    print("total CWEs in dataset: ", len(cwe_list))

    missing_cwe = []  # invalid CWE
    cwe_path = {}

    for id_ in cwe_list:

        if id_ in cwe_path:
            continue

        try:
            id_father = id_.split("-")[1]
            cwe_tree[id_father]["Weakness Abstraction"]
        except BaseException:
            # print(f"MISS [{id_}]")  # CWE-264 CWE-19 CWE-320 CWE-361 CWE-310 CWE-417
            missing_cwe.append(id_)
            continue

        cwe_path[id_] = []
        
        # only support single path
        # while cwe_tree[id_father]["Weakness Abstraction"] != "Pillar":
        #     # bottum up
        #     cwe_path[id_].append(f"CWE-{id_father}")
        #     if len(cwe_tree[id_father]["father"]) > 1:
        #         multiple_father.add(id_)
        #     id_father = str(cwe_tree[id_father]["father"][0])
        
        # id_father = "CWE-" + id_father
        # cwe_path[id_].append(id_father)
        # cwe_path[id_].reverse()

        tmp_path = []
        DFS(cwe_id=id_father, tmp_path=tmp_path, complete_path=cwe_path[id_])
        # print(cwe_path[id_])

        # the path for all of its ancestors are determined
        for path in cwe_path[id_]:
            for idx, node in enumerate(path[:-1]):
                new_path = path[:(idx+1)]
                if node not in cwe_path:
                    cwe_path[node] = [new_path]
                else:
                    # need to check if the path has been added
                    existed_path = [','.join(_) for _ in cwe_path[node]]
                    if ','.join(new_path) not in existed_path:
                        cwe_path[node].append(new_path)

    # for id_, paths in cwe_path.items():
    #     cwe_path[id_] = paths[0] if paths else []
    for id_, paths in cwe_path.items():
        selected_path = None
        for path in paths:
            # print(id_)
            # print(path)
            if len(path) >= 3:
                selected_path = path
                break
        
        if selected_path == None:
            cwe_path[id_] = paths[0] if paths else []
        if selected_path:
            cwe_path[id_] = selected_path
   

    print(f"MISSING [{len(missing_cwe)}]")  # we make sure these samples are removed in clean_dataset

    with open(DATA_PATH + 'cwe_path.json', 'w') as f:
        json.dump(cwe_path, f, indent=4)


def is_selected(x, selected_list):
    if x in selected_list:
        return True
    return False


def get_cwe(x):
    global cve_dict
    cve_id = x.strip().split(',')[0]
    return cve_dict[cve_id]["cwe_id"]


def select_cwe(x, target_cwe):
    cwe_list = [cwe.strip() for cwe in x.split(",")]
    if len(cwe_list) > 1:
        return False
    
    if target_cwe not in x:
        return False
    
    return True


def get_path_len(x):
    global cwe_path
    cwe_id = x.split(",")[0].strip()
    if cwe_id in cwe_path:
        return len(cwe_path[cwe_id])

    return -1


def sample_dataset(file, selected_depth=3):
    # sample samples with CWE categories whose depth is larger than 3

    samples = json.load(open(DATA_PATH + f"{file}.json", 'r'))
    samples = DataFrame(samples)

    samples["path_len"] = samples["path_list"].apply(lambda x:len(x[0]))
    samples = samples[samples["path_len"] >= selected_depth]

    print("#files:", len(samples))
    print("#commit:", len(samples.drop_duplicates(subset="commit_id")))

    del samples["path_len"]

    samples = samples.to_dict(orient="records")

    with open(DATA_PATH + f"{file}_level{selected_depth}.json", 'w') as f:
        json.dump(samples, f, indent=4)


def divide_dataset_stratified(file, selected_depth=3, seed=2022):
    data = json.load(open(DATA_PATH + f"{file}.json", 'r'))  # file-level

    df = DataFrame(data)
    print("file-level", len(df))

    label_index = (selected_depth - 1)  # use cwe at the selected depth as label
    df["label"] = df["path_list"].apply(lambda x:x[0][label_index])

    # must split dataset at commit-level
    # first get the commit ids, then filter at file-level
    df_commit = df.drop_duplicates(subset="commit_id", keep="first").copy()
    print("commit-level", len(df_commit))

    label_distribution = df_commit["label"].value_counts().to_dict()
    with open("label_distribution.json", 'w') as f:
        json.dump(label_distribution, f, indent=4)

    cid_label = df_commit[["commit_id", "label"]].to_dict(orient="records")

    X = np.zeros(len(cid_label))
    y = [_["label"] for _ in cid_label]

    # stratified random sampling
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    k_fold_index = [test_index for train_index, test_index in skf.split(X, y)]

    # 不再划分测试集，按照8:2的比例划分训练集和验证集
    validation_index = list(k_fold_index[4])
    # test_index = list(k_fold_index[9])

    print([len(_) for _ in k_fold_index])
        
    validation_cid = [cid_label[idx]["commit_id"] for idx in validation_index]
    # test_cid = [cid_label[idx]["commit_id"] for idx in test_index]

    # df["is_test"] = df["commit_id"].apply(is_selected, selected_list=test_cid)
    df["is_val"] = df["commit_id"].apply(is_selected, selected_list=validation_cid)

    # df_test = df[df["is_test"] == True]
    df_val = df[df["is_val"] == True]
    # df_train = df[(df["is_test"] == False) & (df["is_val"] == False)]
    df_train = df[(df["is_val"] == False)]

    # del df_test["is_test"], df_test["is_val"]
    # del df_val["is_test"], df_val["is_val"]
    # del df_train["is_test"], df_train["is_val"]
    del df_val["is_val"]
    del df_train["is_val"]

    print("train file-level", len(df_train))
    print("train commit-level", len(df_train.drop_duplicates(subset="commit_id")))
    with open(DATA_PATH + "train_set.json", 'w') as f:
        json.dump(df_train.to_dict(orient="records"), f, indent=4)
    
    print("validation file-level", len(df_val))
    print("validation commit-level", len(df_val.drop_duplicates(subset="commit_id")))
    with open(DATA_PATH + "validation_set.json", 'w') as f:
        json.dump(df_val.to_dict(orient="records"), f, indent=4)
    
    # print("test file-level", len(df_test))
    # print("test commit-level", len(df_test.drop_duplicates(subset="commit_id")))
    # with open(DATA_PATH + "test_set.json", 'w') as f:
    #     json.dump(df_test.to_dict(orient="records"), f, indent=4)


def valid_cwe_tree():
    # only include cwe nodes appear in the dataset 

    train_set = json.load(open(DATA_PATH + "train_set.json", 'r'))
    validation_set = json.load(open(DATA_PATH + "validation_set.json", 'r'))
    dataset = train_set + validation_set

    dataset = DataFrame(dataset)
    dataset = dataset.drop_duplicates(subset="commit_id")

    path_list = [_[0] for _ in dataset["path_list"].to_list()]
    
    cwe_tree = json.load(open(DATA_PATH + "cwe_tree.json", 'r'))
    valid_cwe_tree = dict()
    valid_cwe_tree["CWE-1000"] = {"description": "",
                                "children": set()}  # root node
    
    for path in path_list:
        for i, node in enumerate(path):
            if node not in valid_cwe_tree:
                description = cwe_tree[node.split('-')[1]]['Name']
                description = description.lower()
                valid_cwe_tree[node] = {"description": description,
                                    "children": set()}
            if i == 0:
                valid_cwe_tree["CWE-1000"]["children"].add(node)
            
            if i + 1 < len(path):
                valid_cwe_tree[node]["children"].add(path[i + 1])

    print("# total node:", len(valid_cwe_tree))
    not_leaf_node = 0
    for node in valid_cwe_tree.values():
        node["children"] = list(node["children"])
        node["children"].sort()  # make sure the order is same every time run the code
        if len(node["children"]) > 0:
            not_leaf_node += 1
    print("# no leaf node:", not_leaf_node)

    with open(DATA_PATH + "valid_cwe_tree.json", 'w', encoding="utf-8") as f:
        json.dump(valid_cwe_tree, f, indent=4)


def get_valid_cwes():
    # mapping CWE catefories (only those appeared in the train set and validation set) into indexes

    validation_set = json.load(open(DATA_PATH + "validation_set.json", 'r'))
    train_set = json.load(open(DATA_PATH + "train_set.json", 'r'))
    dataset = train_set + validation_set
    
    dataset = DataFrame(dataset)
    dataset = dataset.drop_duplicates(subset="commit_id")

    path_list = [_[0] for _ in dataset["path_list"].to_list()]
    max_path_len = max(len(_) for _ in path_list)

    count_cwes = [dict() for _ in range(max_path_len)]

    for path in path_list:
        for depth, cwe_id in enumerate(path):
            if cwe_id not in count_cwes[depth]:
                count_cwes[depth][cwe_id] = 0
            count_cwes[depth][cwe_id] += 1

    valid_cwes = []  # cwe labels
    for depth in range(max_path_len):
        count_cwes[depth] = sorted(count_cwes[depth].items(), key=lambda x:x[1], reverse=True)
        
        label_cwes = [_[0] for _ in count_cwes[depth]]  # order is decided by the support
        # label_cwes.sort()
        valid_cwes.append(label_cwes)
        print(f"depth: {depth}, #CWE: {len(label_cwes)}")

    with open(DATA_PATH + "valid_cwes.json", 'w', encoding="utf-8") as f:
        json.dump(valid_cwes, f, indent=4)


def dataset_statistics(file):
    sample_file = json.load(open(DATA_PATH + f"{file}.json", 'r'))

    sample_file = DataFrame(sample_file)

    print("# file-level:", len(sample_file))
    
    sample_commit = sample_file.drop_duplicates(subset=["commit_id"])  # commit-level
    print("# commit-level", len(sample_commit))

    repo_distribution = sample_commit["repo"].value_counts().to_dict()
    print("#repo:", len(repo_distribution))
    print(list(repo_distribution.items())[:5])

    all_cves = [_ for cve in sample_commit["cve_list"].to_list() for _ in cve.split(',')]
    u, c = np.unique(all_cves, return_counts=True)
    cve_distribution = [(uu, cc) for uu, cc in zip(u, c)]
    cve_distribution.sort(key=lambda x:x[1], reverse=True)
    print("#cve:", len(cve_distribution))
    print(cve_distribution[:5])

    all_cwes = [_ for cwe in sample_commit["cwe_list"].to_list() for _ in cwe.split(',')]  # can alse use path list
    u, c = np.unique(all_cwes, return_counts=True)
    cwe_distribution = [(uu, cc) for uu, cc in zip(u, c)]
    cwe_distribution.sort(key=lambda x:x[1], reverse=True)
    print("#cwe:", len(cwe_distribution))
    print(cwe_distribution[:5])

    all_pathes = [_[0] for _ in sample_commit["path_list"].to_list()]  # make sure each commit only has one sample
    max_depth = max(len(_) for _ in all_pathes)
    
    depth_distribution = [0]*max_depth
    for path in all_pathes:
        depth_distribution[len(path) - 1] += 1
    
    print(depth_distribution)

    for i in range(1, max_depth):
        depth_distribution[-1-i] += depth_distribution[-i]
    
    print(depth_distribution)

    x = range(max_depth)

    plt.figure(figsize=(25, 15))
    plt.ylim(0, 9500)
    plt.yticks(fontsize=50)
    plt.bar(x, depth_distribution, width=0.7, color="gray", edgecolor='gray')
    plt.xticks(x, ["≥1", "≥2", "≥3", "≥4", "=5"], fontsize=50)
    for a, b in zip(x, depth_distribution):
        plt.text(a, b+100, b, ha="center", fontsize=50)
    plt.xlabel("Depth of CWE Category", fontsize=50)
    plt.ylabel("# Commits", rotation="vertical", fontsize=50)
    plt.savefig("cwe_depth_distribution.pdf")


def num_file_hunk_per_commit(file, p=95):
    dataset = json.load(open(DATA_PATH + f"{file}.json", 'r'))
    dataset = pd.DataFrame(dataset)

    num_hunk_per_commit = dataset.groupby("commit_id")["NUM_HUNK"].agg("sum").to_list()
    num_file_per_commit = dataset.groupby("commit_id")["file_name"].agg("count").to_list()

    print(f"#hunk per commit at percentile={p}", np.percentile(a=num_hunk_per_commit, q=p))
    print(f"#file per commit at percentile={p}", np.percentile(a=num_file_per_commit, q=p))

    
cwe_tree = None
cwe_processed = None

# DATA_PATH = "xxx"

if __name__ == '__main__':
    # generate_dataset_hunk(directory="xxx")
    # clean_dataset(file="dataset_hunk")
    # sample_dataset(file="dataset_cleaned")
    DATASET = 'uniqueness/0.3'
    DATA_PATH = "/home/nfs/zxh2023/TreeVul/TreeVul/zxh_data/" + DATASET + "/"
    # 这个划分数据集的算法每次划分得到的训练集大小不同，不再使用
    # divide_dataset_stratified(file="origin")
    # build_CWE_tree()
    # 这个函数应该是要对整个数据集生成映射
    generate_CWE_mapping("origin")
    get_valid_cwes()
    valid_cwe_tree()
    # dataset_statistics()