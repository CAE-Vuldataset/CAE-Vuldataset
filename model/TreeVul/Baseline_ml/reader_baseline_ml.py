import copy
from tokenizer import tokenize_by_punctuation, tokenize_text


def reader_ml_baselines(samples_file, max_hunk_num=8, target_depth=2, tokenizer_max_length=128):
    samples_commit = dict()  # commit-level sample
    for s in samples_file:
        commit_id = s["commit_id"]
        if commit_id not in samples_commit:
            samples_commit[commit_id] = {"commit_id": commit_id, "path": s["path_list"][0], "model_input_pair": []}

        # hunk-level input
        for rem_diff, add_diff in zip(s["REM_DIFF"], s["ADD_DIFF"]):

            if len(samples_commit[commit_id]["model_input_pair"]) >= max_hunk_num:
                break

            # preprocess code exactly the same as the CodeBERT
            for s in ['\r\n', '\r', '\n']:
                rem_diff = rem_diff.replace(s, ' ')
                add_diff = add_diff.replace(s, ' ')
            rem_diff = ' '.join(rem_diff.strip().split())
            add_diff = ' '.join(add_diff.strip().split())

            # tokenize
            rem_diff = tokenize_by_punctuation(rem_diff)
            add_diff = tokenize_by_punctuation(add_diff)
            rem_diff = tokenize_text(rem_diff)[:tokenizer_max_length]
            add_diff = tokenize_text(add_diff)[:tokenizer_max_length]
            
            diff_pair = rem_diff + ["<SEP>"] + add_diff
            
            samples_commit[commit_id]["model_input_pair"].append(diff_pair)
    
    samples_commit = list(samples_commit.values())

    print(f"sample num (commit-level): {len(samples_commit)}")

    # the input of ml model can not be a list of strs, need further merge
    dataset_commits = []
    dataset_label_path = []
    for s in samples_commit:
        if len(s["path"]) <= target_depth:
            continue

        dataset_label_path.append(s["path"])

        # rem_0 <SEP> add_0 <NL> rem_1 <SEP> add_1
        input_pair_all = [" ".join(hunk) for hunk in s["model_input_pair"]]
        hunk_newline_hunk = " <NL> ".join(input_pair_all)
        dataset_commits.append(hunk_newline_hunk)
        
    return dataset_commits, dataset_label_path