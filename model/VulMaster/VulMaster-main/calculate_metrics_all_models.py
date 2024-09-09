from collections import Counter
import json
import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

df_ = pd.read_csv('CWE_examples_GPT35_generate_fixes_full.csv')
column_names = df_.columns.tolist()
def diff2_before_after_str(diff):
    lines = diff.split('\n')
    lines = [l.strip() for l in lines]

    befores, afters = [],[]
    for l in lines:
        if l.startswith('- '):
            befores.append(l[1:].strip())
        elif l.startswith('+ '):
            afters.append(l[1:].strip())
        else:
            befores.append(l.strip())
            afters.append(l.strip())

    new_befores, new_afters = [],[]
    for l in befores:
        if len(l.strip()) == 0:
            continue
        else:
            new_befores.append(l.strip())

    for l in afters:
        if len(l.strip()) == 0:
            continue
        else:
            new_afters.append(l.strip())
    befores = new_befores
    afters  = new_afters

    return '\n'.join(befores), '\n'.join(afters)
all_befores, all_afters, all_vulnerable_types = [],[],[]
for i in range(len(df_)):
    one_item = df_.iloc[i, :]
    cwe_id = one_item[0]
    for name in column_names[1:]:
        try:
            assert(len(one_item[name])>0)
            # print(one_item[name])
            before, after = diff2_before_after_str(one_item[name])
            all_befores.append(before.replace('\n', ' '))
            all_afters.append(after.replace('\n', ' '))
            all_vulnerable_types.append(cwe_id)
        except:
            continue
type_dict_example = dict(Counter(all_vulnerable_types))
type_dict_example = dict(sorted(type_dict_example.items(), key=lambda item: item[1], reverse=True))



def load_data(data_path=None):
    golds, preds = [],[]
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
            try:
                with open(data_path, 'r', encoding='utf-8') as fin:
                    data = json.load(fin)
            except:
                with open(data_path, 'r', encoding='utf-8') as fin:
                    text = fin.read()
                    # text = text.split('][')[0] +']'
                    text = "[" + text.split('][')[1]
                    data = json.loads(text)
                    print('data samples:', len(data))

    examples = []
    for k, example in enumerate(data):
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        examples.append(example)
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()
    for ex in examples:
        gold_ = ex["golden_answers"][0].replace('<S2SV_StartBug>', '<vul-start>').replace('<S2SV_EndBug>', '<vul-end>').replace('<S2SV_blank>', ' ').replace('<S2SV_null>', ' ').replace('<S2SV>', ' ')
        pred_ = ex["generated_answer"].replace('<S2SV_StartBug>', '<vul-start>').replace('<S2SV_EndBug>', '<vul-end>').replace('<S2SV_blank>', ' ').replace('<S2SV_null>', ' ').replace('<S2SV>', ' ')
        golds.append(gold_.strip())
        preds.append(pred_.strip())
    return golds, preds

def load_data_text(data_path=None):
    golds, preds = [],[]
    assert data_path
    with open(data_path+'/test_predictions.txt') as f:
        preds = f.readlines()
    with open(data_path+'/test_references.txt') as f:
        golds = f.readlines()


    preds = [l.replace('<S2SV_StartBug>', '<vul-start>').replace('<S2SV_EndBug>', '<vul-end>').replace('<S2SV_blank>', ' ').replace('<S2SV_null>', ' ').replace('<S2SV>', ' ') for l in preds]
    golds = [l.replace('<S2SV_StartBug>', '<vul-start>').replace('<S2SV_EndBug>', '<vul-end>').replace('<S2SV_blank>', ' ').replace('<S2SV_null>', ' ').replace('<S2SV>', ' ') for l in golds]

    return golds, preds


golds, preds = load_data("checkpoint_final/final_model/test_results/final_model.json")


def simple_em(golds, preds):
    assert(len(golds) == len(preds))
    count = []
    num = 0


    for i in range(len(golds)):
        if src.evaluation.ems(preds[i], [golds[i]]):
            count.append(1)
        else:
            count.append(0)

        if src.evaluation.ems(preds[i], [golds[i]]) == True and (' '.join(golds[i].split()).strip().lower() != ' '.join(preds[i].split()).strip().lower()):
            num += 1


    # print(Counter(count))
    print(sum(count)/len(count))
    return sum(count)/len(count)

print("Overall Simple EM:")
simple_em(golds, preds)

from codebleu import calc_codebleu
result = calc_codebleu(golds, preds, lang="c", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
print(result)




