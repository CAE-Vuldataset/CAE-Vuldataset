# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import pickle
import random
import json
import numpy as np
import torch
import functools

from tqdm import tqdm
from tree_sitter import Language, Parser
from simple_file_checksum import get_checksum
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


from model import *
from attribution import *


cpu_cont = 16
logger = logging.getLogger(__name__)

def append_to_path(path):
    if path not in sys.path:
        sys.path.append(path)

root_dir = os.getcwd().split("/Causality")[0]
data_dir = os.path.join(root_dir, 'data')
saved_model_dir = os.path.join(root_dir, 'saved_models')
project_dir = os.path.join(root_dir, 'Causality')
append_to_path(f"{project_dir}/GraphCodeBERT")
append_to_path(project_dir)
append_to_path(f"{project_dir}/NatGen")

from NatGen.src.data_preprocessors.transformations import *
language = 'c'
parser_path = project_dir + '/NatGen/parser/languages.so'
transform = VarReplacer(parser_path, language, "", 1)

from gcb_parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript, DFG_c
from gcb_parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token)

dfg_function = {
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript,
    'c': DFG_c,
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language(f'{project_dir}/GraphCodeBERT/gcb_parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
    
#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
             source,
             input_tokens,
             input_ids,
             position_idx,
             dfg_to_code,
             dfg_to_dfg,
             label,
             xp_idx,
    ):
        #The code function
        self.source = source
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        
        #label
        self.label=label
        # Maximum Spurious feature shared data
        self.xp_idx = xp_idx
        

def convert_examples_to_features(js, tokenizer, args):
    parser=parsers[args.parser_lang]
    code = js["func"]
    if args.tokenize_ast_token:
        code = ' '.join(js['ast_tokens'])
    else:
        code = ' '.join(code.split())
    label = js["target"]
    
    #extract data flow
    code_tokens,dfg=extract_dataflow(code, parser, args.parser_lang)
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]  
    
    #truncating
    code_tokens=code_tokens[:args.code_length+args.data_flow_length-3-min(len(dfg),args.data_flow_length)][:512-3]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg=dfg[:args.code_length+args.data_flow_length-len(source_tokens)]
    source_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    source_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=args.code_length+args.data_flow_length-len(source_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    source_ids+=[tokenizer.pad_token_id]*padding_length      
    
    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
        
    return InputFeatures(js['func'], source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg,label, js.get('xp_idx', []))

def get_data(tokenizer, args, file_path):

    file_Path_base = '/'.join(file_path.split('/')[:-1])
    filename_wo_ext = '.'.join((file_path.split('/')[-1].split('.')[:-1]))
    #load index
    cache_filepath = f"{file_Path_base}/cached_{filename_wo_ext}_tat_{args.tokenize_ast_token}_mname_{args.model_name}.pkl"  # do not use os.path.join here - if file_path is absolute, then cache_filepath is set to file_path and it will get zonked
    examples = None
    checksum = get_checksum(file_path)
    # load cache
    if os.path.exists(cache_filepath):
        with open(cache_filepath, "rb") as of:
            cache_data = pickle.load(of)
            if cache_data['checksum'] == checksum:
                print(f"Loading cached data from {cache_filepath} with checksum {checksum}...")
                examples = cache_data["examples"]
            else:
                print(f"Checksum mismatch {cache_data['checksum']} != {checksum} -- reloading cached data...")
    if examples is None:
        with open(file_path) as f:
            num_lines = sum(1 for _ in f)
        with open(file_path) as f:
            data = [json.loads(line.strip()) for line in f]
            examples = list(tqdm(map(functools.partial(convert_examples_to_features, tokenizer=tokenizer, args=args), data), desc=f"Loading examples from {file_path}...", total=num_lines))
            # save to cache
            os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
            with open(cache_filepath, "wb") as of:
                pickle.dump({
                    "checksum": checksum,
                    "examples": examples,
                }, of)
    return examples

def get_item(args, example):
    #calculate graph-guided masked function
    attn_mask= np.zeros((args.code_length+args.data_flow_length,
                            args.code_length+args.data_flow_length),dtype=bool)
    #calculate begin index of node and max length of input
    node_index=sum([i>1 for i in example.position_idx])
    max_length=sum([i!=1 for i in example.position_idx])
    #sequence can attend to sequence
    attn_mask[:node_index,:node_index]=True
    #special tokens attend to all tokens
    for idx,i in enumerate(example.input_ids):
        if i in [0,2]:
            attn_mask[idx,:max_length]=True
    #nodes attend to code tokens that are identified from
    for idx,(a,b) in enumerate(example.dfg_to_code):
        if a<node_index and b<node_index:
            attn_mask[idx+node_index,a:b]=True
            attn_mask[a:b,idx+node_index]=True
    #nodes attend to adjacent nodes 
    for idx,nodes in enumerate(example.dfg_to_dfg):
        for a in nodes:
            if a+node_index<len(example.position_idx):
                attn_mask[idx+node_index,a+node_index]=True
                
    return (torch.tensor(example.input_ids),
            torch.tensor(example.position_idx),
            torch.tensor(attn_mask), 
            torch.tensor(example.label))


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', split='train'):
        self.args=args
        self.tokenizer = tokenizer
        self.examples = get_data(tokenizer, args, file_path)
        self.split=split

        self.examples_train = {}
        self.examples_train['all'] = get_data(tokenizer, args, args.train_data_file)
        self.train_ids_by_label = [[], []]
        for i, example in enumerate(self.examples_train['all']):
            if example.label not in self.examples_train:
                self.examples_train[example.label] = [example]
            else:
                self.examples_train[example.label].append(example)
            self.train_ids_by_label[example.label].append(i)

        if 'V3' in args.sp_setting:
            train_data = self.examples_train['all']
            var_names = [{}, {}] 
            for example in train_data:
                vnames = transform.get_var_names(example.source)
                for nm in vnames:
                    var_names[example.label][nm] = var_names[example.label].get(nm, 0) + 1
            def top_k_names(_names):
                frq_name = sorted([(_names[ns], ns) for ns in _names if _names[ns] > 1], reverse=True)
                top_k_index = (len(frq_name) * 10) // 100
                return [nm for _, nm in frq_name[:top_k_index]]
            var_names[0] = top_k_names(var_names[0])
            var_names[1] = top_k_names(var_names[1])
            common_names = list(set(var_names[0]) & set(var_names[1]))
            
            self.top_k_vulnerable_names = {
                0:  [nm for nm in var_names[0] if nm not in common_names],
                1:  [nm for nm in var_names[1] if nm not in common_names],
            }
            print(f"Top K non vul names {len(self.top_k_vulnerable_names[0])}")
            print(f"Top K vul names {len(self.top_k_vulnerable_names[1])}")
        if split == 'train':
            examples_ = self.examples
            vul_num = 0
            vul_examples, nvul_examples = [], []
            for ex in examples_:
                if ex.label == 0:
                    nvul_examples.append(ex)
                else:
                    vul_examples.append(ex)
                    vul_num += 1
            nvul_examples = random.sample(nvul_examples, min(len(nvul_examples), 2 * vul_num))
            self.examples = vul_examples + nvul_examples   
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))       
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))


    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        x = get_item(self.args, self.examples[item])
        if self.split == 'train':
            if 'V1' in self.args.sp_setting:
                xps = self.examples[item].xp_idx
                xp_i = random.choice(xps)
                x +=  get_item(self.args, self.examples_train['all'][xp_i[0]])
                return x
            else:
                ind_ = random.randint(0, len(self.examples_train[self.examples[item].label]) - 1)
                src = self.examples[item].source
                des = self.examples_train[self.examples[item].label][ind_].source
                random_names = []
                if "V2" in self.args.sp_setting:
                    random_names = transform.get_var_names(src)
                else:
                    vnames = self.top_k_vulnerable_names[self.examples[item].label]
                    random_names += random.sample(vnames, min(self.args.max_var_num, len(vnames)))
                try:
                    con_code = transform.transform_code(des, random_names)[0]
                except Exception as e:
                    con_code = des
                input_feature_obj = convert_examples_to_features(
                    {'func': con_code, 'target': self.examples[item].label},
                    self.tokenizer,
                    self.args,
                )
                x +=  get_item(self.args, input_feature_obj)

        elif self.split == 'valid':
            idxp = random.randint(0, len(self.examples_train['all']) - 1)
            x +=  get_item(self.args, self.examples_train['all'][idxp])
        else:
            xp = random.sample(self.train_ids_by_label[0], self.args.samples//2)
            xp += random.sample(self.train_ids_by_label[1], self.args.samples//2)       
            for idxp in xp:
                x += get_item(self.args, self.examples_train['all'][idxp])
            # xp = torch.tensor(self.examples_train['all'][random.randint(0, len(self.examples_train['all']) - 1)].input_ids)
        return x


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, classifier, tokenizer):
    """ Train the model """
    
    #build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    args.max_steps=args.epochs*len( train_dataloader)
    args.warmup_steps=args.max_steps//5
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_perf=0

    classifier.zero_grad()
 
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            batch_input = [x.to(args.device)  for x in batch]

            assert len(batch_input) == 8

            model.eval()
            classifier.train()

            with torch.no_grad():
                feature = model.get_representation(input_ids=batch_input[0], position_idx=batch_input[1], attn_mask=batch_input[2])
            loss, logits = classifier(feature, input_ids=batch_input[4], position_idx=batch_input[5], attn_mask=batch_input[6], labels=batch_input[3])
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
                
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

        results = evaluate(args, model, classifier, tokenizer, eval_when_training=True)    
        
        # Save model checkpoint
        if results[f'eval_{args.validation_metric}']>best_perf:
            best_perf=results[f'eval_{args.validation_metric}']
            logger.info("  "+"*"*20)  
            logger.info("  Best %s:%s", args.validation_metric, round(best_perf,4))
            logger.info("  "+"*"*20)                          
            
            def save_model(model_, name_):                       
                output_dir = os.path.join(args.causal_output_dir, f'{name_}.bin')                      
                model_to_save = model_.module if hasattr(model_, 'module') else model_
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)

            if not os.path.exists(args.causal_output_dir):
                os.makedirs(args.causal_output_dir)
            save_model(model, 'model')
            save_model(classifier, 'classifier')


def evaluate(args, model, classifier, tokenizer, eval_when_training=False):
    #build dataloader
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file, split='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=4)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    classifier.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        batch_input = [x.to(args.device)  for x in batch]
        assert len(batch_input) == 8
        with torch.no_grad():
            feature = model.get_representation(input_ids=batch_input[0], position_idx=batch_input[1], attn_mask=batch_input[2])
            lm_loss, logit = classifier(feature, input_ids=batch_input[4], position_idx=batch_input[5], attn_mask=batch_input[6], labels=batch_input[3])
            eval_loss += lm_loss.mean().item()
            logit = logit.max(1)[1]
            logits.append(logit.cpu().numpy())
            y_trues.append(batch_input[3].cpu().numpy())
        nb_eval_steps += 1
    
    #calculate scores
    y_preds=np.concatenate(logits, 0)
    y_trues=np.concatenate(y_trues, 0)
    accuracy=accuracy_score(y_trues, y_preds)
    recall=recall_score(y_trues, y_preds)
    precision=precision_score(y_trues, y_preds)   
    f1=f1_score(y_trues, y_preds)             
    result = {
        "eval_accuracy": float(accuracy),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, classifier, tokenizer, best_threshold=0):
    #build dataloader
    eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file, split='test')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)


    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    classifier.eval()
    logits=[]  
    y_trues=[]
    representations = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        batch_input = [x.to(args.device)  for x in batch]
        assert len(batch_input) == 41 * 4
        with torch.no_grad():
            feature = model.get_representation(input_ids=batch_input[0], position_idx=batch_input[1], attn_mask=batch_input[2])

            if args.inference:
                loss, logit_compose, rp= classifier(feature, input_ids=batch_input[0], position_idx=batch_input[1], attn_mask=batch_input[2], labels=batch_input[3], representation=True)
            else:
                loss, logit_compose, rp = None, None, None
                for i in range(4, (args.samples + 1) * 4, 4):
                    _loss, _logit_compose, _rp = classifier(feature, input_ids=batch_input[i], position_idx=batch_input[i+1], attn_mask=batch_input[i+2], labels=batch_input[3], representation=True)
                    _logit_compose = _logit_compose.cpu().numpy()
                    max_indices = np.argmax(_logit_compose, axis=1)
                    result = np.zeros_like(_logit_compose)
                    result[np.arange(result.shape[0]), max_indices] = 1

                    if loss == None and logit_compose == None:
                        loss, logit_compose, rp = _loss, _logit_compose, _rp
                    else:
                        loss += _loss
                        logit_compose += _logit_compose
                        rp += _rp
                logit_compose = torch.tensor(logit_compose)
                loss /= args.samples

            logit = logit_compose.max(1)[1]
            logits.append(logit.cpu().numpy())
            y_trues.append(batch_input[3].cpu().numpy())
            representations.append(rp)

        nb_eval_steps += 1
    
    #output result
    y_preds=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    representations=np.concatenate(representations, 0)

    accuracy=accuracy_score(y_trues, y_preds)
    recall=recall_score(y_trues, y_preds)
    precision=precision_score(y_trues, y_preds)   
    f1=f1_score(y_trues, y_preds)             
    result = {
        "eval_accuracy": float(accuracy),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),        
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
    
    file_name = args.test_data_file.split("/")[-1].split('.')[0]
    np.savez(f'{args.result_dir}/representations_{file_name}_{args.inference}.npz', representation=representations, labels=y_trues)
    return result


def attribution(args, model, classifier, tokenizer, data_path):
    # Note that DistributedSampler samples randomly
    eval_dataset = TextDataset(tokenizer, args, data_path, split='test')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    model.eval()
    classifier.eval()

    wrapper = ModelClassifierWreapper(model, classifier, args)
    deeplift = DeepLift(wrapper)
    
    attributions_result = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        batch_input = [x.to(args.device)  for x in batch]
        assert len(batch_input) == 41 * 4

        if args.inference:
            xx, xxxp = batch_input[0:4], batch_input[4:]
            attribution_score = do_attribution(wrapper, tokenizer, deeplift, xx, xxxp)
            score, tokens, pred = attribution_score['attributions'].numpy(), attribution_score['tokens'], attribution_score['pred']
        else:
            xx = batch_input[:4]
            for i in range(4, (args.samples + 1) * 4, 4):
                xxxp = batch_input[i: i + 4]
                attribution_score = do_attribution(wrapper, tokenizer, deeplift, xx, xxxp)
                if i == 4:
                    score, tokens, pred = attribution_score['attributions'], attribution_score['tokens'], attribution_score['pred']
                else:
                    score += attribution_score['attributions']
                    pred += attribution_score['pred']
                    assert tokens == attribution_score['tokens']
            score /= args.samples
            score = score / torch.norm(score)
            score = score.numpy()
        # print(pred)
        score, tokens = score.flatten(), tokens[0]
        assert len(score) == len(tokens)
        attributions_result.append({
            'attributions': score,
            'tokens': tokens,
            'pred': pred.numpy().flatten(),
        })
    
    file_name = data_path.split("/")[-1].split(".")[0]
    if not os.path.exists(f"{args.result_dir}/attributions/"):
        os.makedirs(f"{args.result_dir}/attributions/")
    with open(f"{args.result_dir}/attributions/{file_name}_{args.inference}.pkl", "wb") as f:
        pickle.dump(attributions_result, f)


def get_model_1(args):
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels=2
    # args.block_size = min(args.b, tokenizer.max_len_single_sentence)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)
    model = Model(model, config, tokenizer, args)
    return model, tokenizer


classifiers = {
    'B4': ClassifierB4,
    'B6': ClassifierB6,
}
def get_model_2(args):
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 2
    # args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    classifier = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)
    classifier = classifiers[args.classifier](classifier, config, args)
    return classifier, tokenizer

                                                
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=128, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--do_attribution", action='store_true',
                        help="Whether to run attribution")
    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=10,
                        help="training epochs")
    
    parser.add_argument('--validation_metric', type=str, default='f1',
                        help="metric to use to for model selection based on the validation set")
    parser.add_argument('--dataloader_nprocs', type=int, default=6,
                        help="how many processes to use to load data")
    parser.add_argument('--do_preload_datasets', action='store_true',
                        help="whether to go on past loading the datasets")
    parser.add_argument('--parser_lang', type=str, default='java', help="Enter the parser language")
    parser.add_argument('--tokenize_ast_token', type=int, default=0)
    parser.add_argument('--node', type=str, default='node')
    parser.add_argument('--dataset', type=str, default="Devign")
    parser.add_argument('--model_name', type=str, default="causal_spurious")
    parser.add_argument('--classifier', type=str, default='A')
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument("--inference", action='store_true',
                        help="Whether to run the inference.")
    parser.add_argument("--result_dir", default=os.path.join(project_dir, "results/GraphCodeBERT"), type=str,
                        help="To store the result in per epoch")
    parser.add_argument("--vanilla_output_dir", default=None, type=str, required=True,
                        help="The output directory where the vanilla model predictions and checkpoints will be written.")
    parser.add_argument("--causal_output_dir", default=None, type=str, required=True,
                        help="The output directory where the causal model predictions and checkpoints will be written.")
    parser.add_argument("--save_representation", action='store_true',
                        help="Whether to run the inference.")
    parser.add_argument('--sp_setting', type=str, default="V3")
    parser.add_argument('--max_var_num', type=int, default=50)
    parser.add_argument('--early_layer', type=int, default=4)

    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)


    # Set seed
    set_seed(args)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


    model, tokenizer = get_model_1(args)
    classifier, tokenizer = get_model_2(args)
    
    model.to(args.device)
    classifier.to(args.device)
    
    logger.info("Training/evaluation parameters %s", args)

    if args.do_preload_datasets:
        TextDataset(tokenizer, args, file_path=args.train_data_file)
        TextDataset(tokenizer, args, file_path=args.eval_data_file)
        TextDataset(tokenizer, args, file_path=args.test_data_file)

    # Training
    if args.do_train:
        output_dir = os.path.join(args.vanilla_output_dir, 'model.bin')
        model.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file, split='train')
        train(args, train_dataset, model, classifier, tokenizer)

    def load_model(model_, name_):
        output_dir = os.path.join(args.causal_output_dir, f'{name_}.bin')
        model_.load_state_dict(torch.load(output_dir))      
        model_.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        load_model(model, 'model')
        load_model(classifier, 'classifier')
        evaluate(args, model, classifier, tokenizer)

    if args.do_test:
        print("** Calculate Inference **")
        print("*************************")
        args.inference = True
        load_model(model, 'model')
        load_model(classifier, 'classifier')
        test(args, model,classifier, tokenizer, best_threshold=0.5)

        print("** Calculate Causal **")
        print("*************************")
        args.inference = False
        load_model(model, 'model')
        load_model(classifier, 'classifier')
        test(args, model,classifier, tokenizer, best_threshold=0.5)
    
    if args.do_attribution:
        args.inference = True
        load_model(model, 'model')
        load_model(classifier, 'classifier')
        attribution(args, model, classifier, tokenizer, args.test_data_file)

        args.inference = False
        load_model(model, 'model')
        load_model(classifier, 'classifier')
        attribution(args, model, classifier, tokenizer, args.test_data_file)

    return results


if __name__ == "__main__":
    main()
