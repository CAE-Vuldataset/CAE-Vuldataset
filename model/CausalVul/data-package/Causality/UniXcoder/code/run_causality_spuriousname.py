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
import random
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np
import pandas as pd
import pickle

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm

from model import *
from attribution import *
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
import sys

logger = logging.getLogger(__name__)

def append_to_path(path):
    if path not in sys.path:
        sys.path.append(path)

root_dir = os.getcwd().split("/Causality")[0]
data_dir = os.path.join(root_dir, 'data')
saved_model_dir = os.path.join(root_dir, 'saved_models')
project_dir = os.path.join(root_dir, 'Causality')

append_to_path(project_dir)
append_to_path(f"{project_dir}/NatGen")

from NatGen.src.data_preprocessors.transformations import *
language = 'c'
parser_path = project_dir + '/NatGen/parser/languages.so'
transform = VarReplacer(parser_path, language, "", 1)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 source,
                 input_tokens,
                 input_ids,
                 idx,
                 label,
                 xp_idx,
    ):
        self.source = source
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=idx
        self.label=label
        self.xp_idx = xp_idx

 
def convert_examples_to_features(js,tokenizer,args):
    """convert examples to token ids"""
    code = js[args.code_key]
    code = ' '.join(code.split())
    if args.model_name_or_path == "microsoft/unixcoder-base":
        code_tokens = tokenizer.tokenize(code)[:args.block_size-4]
        source_tokens = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    else:
        code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return InputFeatures(js[args.code_key], source_tokens,source_ids,js[args.index_key],int(js[args.label_key]),  js.get('xp_idx', []))

def convert_code_to_features(args, code, tokenizer):
    #source
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-4]
    source_tokens = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return (source_tokens, source_ids)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, split='train'):
        self.file_path = file_path
        self.split = split
        self.args = args
        self.tokenizer = tokenizer
        print(args.train_data_file, self.file_path)

        def get_data(file_path_):
            ex = []
            with open(file_path_) as f:
                for line in f:
                    js=json.loads(line.strip())
                    input_features = convert_examples_to_features(js,tokenizer,args)
                    ex.append(input_features)
            return ex
        
        self.examples_train = {}
        self.examples_train['all'] = get_data(args.train_data_file)
        self.train_ids_by_label = [[], []]
        for i, example in enumerate(self.examples_train['all']):
            if example.label not in self.examples_train:
                self.examples_train[example.label] = [example]
            else:
                self.examples_train[example.label].append(example)
            self.train_ids_by_label[example.label].append(i)
        self.examples = get_data(self.file_path)
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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        x = torch.tensor(self.examples[i].input_ids)
        label = torch.tensor(self.examples[i].label)
        if 'train' == self.split:
            if self.args.sp_setting == 'V1':
                xps = self.examples[i].xp_idx
                xp_i = random.choice(xps)
                xp = torch.tensor(self.examples_train['all'][xp_i[0]].input_ids)
                return x, xp, label
            else:
                ind_ = random.randint(0, len(self.examples_train[self.examples[i].label]) - 1)
                src = self.examples[i].source
                des = self.examples_train[self.examples[i].label][ind_].source
                random_names = []
                if "V2" in self.args.sp_setting:
                    random_names = transform.get_var_names(src)
                else:
                    vnames = self.top_k_vulnerable_names[self.examples[i].label]
                    random_names += random.sample(vnames, min(self.args.max_var_num, len(vnames)))
                try:
                    con_code = transform.transform_code(des, random_names)[0]
                except Exception as e:
                    con_code = des
                tok_, ids_ = convert_code_to_features(self.args, con_code, self.tokenizer)
                xp = torch.tensor(ids_)

        elif 'valid' == self.split:
            xp = torch.tensor(self.examples_train['all']
                [random.randint(0, len(self.examples_train['all']) - 1)].input_ids
            )
        else:
            xp = random.sample(self.train_ids_by_label[0], self.args.samples//2)
            xp += random.sample(self.train_ids_by_label[1], self.args.samples//2)
        return x, xp, label

    def get_xp(self, ids):
        batch_ids = []
        for i in ids:
            batch_ids.append(self.examples_train['all'][i].input_ids)
        return torch.tensor(batch_ids)
 

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, classifier, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    
    args.max_steps = args.num_train_epochs*len( train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu )
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Effective train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    losses, best_perf, acc_loss = [], 0, []
    
    classifier.zero_grad()
    for idx in range(args.num_train_epochs):
        logits_, labels_ = [], []
        for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)
            x_p    = batch[1].to(args.device)
            labels = batch[2].to(args.device)

            model.eval()
            classifier.train()
            
            with torch.no_grad():
                feature = model.get_representation(inputs)
                if args.classifier[0] == 'A':
                    x_p = model.get_embedding(x_p)
            
            loss, logits = classifier(feature, x_p, labels)
            logit_ = logits.max(1)[1]
            logits_.append(logit_.detach().cpu().numpy())
            labels_.append(labels.detach().cpu().numpy())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)

            losses.append(loss.item())

            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(np.mean(losses[-100:]),4)))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  

        logits_=np.concatenate(logits_, 0)
        labels_=np.concatenate(labels_, 0)
        acc = accuracy_score(labels_, logits_)
        f1 = f1_score(labels_, logits_)

        ## Evaluation to save the model
        results = evaluate(args, model, classifier, tokenizer)
        acc_loss.append([acc, f1, np.mean(losses[-100:]).item(), results['eval_accuracy'], results['eval_f1'], results['eval_loss']])           
        print(acc_loss)
        if results[f'eval_{args.validation_metric}'] > best_perf:
            best_perf = results[f'eval_{args.validation_metric}']
            logger.info("  "+"*"*20)  
            logger.info(f"  Best {args.validation_metric}:%s",round(best_perf,4))
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

        df = pd.DataFrame(acc_loss, columns=[
            'Train Acc', 'Train F1', 'Train Loss',
            'Val Acc', 'Val F1', 'Val Loss',
        ]) 
        df.to_csv(f"{args.result_dir}/acc_loss.csv")   


def evaluate(args, model, classifier, tokenizer, split='valid'):
    """ Evaluate the model """
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, split=split)
    if 'test' in split:
        eval_dataset = TextDataset(tokenizer, args, args.test_data_file, split=split)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Effective batch size = %d", args.eval_batch_size * args.gradient_accumulation_steps)
    
    model.eval()
    classifier.eval()
    logits, labels =  [], []
    losses = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)  
        x_p     = batch[1].to(args.device)
        label  = batch[2].to(args.device)

        with torch.no_grad():
            feature = model.get_representation(inputs)
            if args.classifier[0] == 'A':
                x_p = model.get_embedding(x_p)
            loss, logit_compose = classifier(feature, x_p, label)
            losses.append(loss.item())
            logit = logit_compose.max(1)[1]
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    preds=np.concatenate(logits, 0)
    labels=np.concatenate(labels, 0)
          
    result = {
        "eval_accuracy": accuracy_score(labels, preds),
        "eval_precision": precision_score(labels, preds),
        "eval_recall": recall_score(labels, preds),
        "eval_f1": f1_score(labels, preds),
        "eval_loss": sum(losses) / len(losses)
    }

    return result


def test(args, model, classifier, tokenizer, split='test'):
    # Note that DistributedSampler samples randomly
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, split=split)
    if 'test' in split:
        eval_dataset = TextDataset(tokenizer, args, args.test_data_file, split=split)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    classifier.eval()
    logits, labels =  [], []
    losses = []
    representations = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)  
        xps     = batch[1]  
        label  = batch[2].to(args.device)

        with torch.no_grad():
            feature = model.get_representation(inputs)
            if args.inference:
                # x_p_embeddings = model.get_embedding(inputs)
                input_ = inputs
                if args.classifier[0] == 'A':
                    input_ = model.get_embedding(inputs)
                loss, logit_compose, rp = classifier(feature, input_, label, representation=args.save_representation)
            else:
                for j, id_ in enumerate(xps):
                    xp = eval_dataset.get_xp(id_).to(args.device)
                    if args.classifier[0] == 'A':
                        xp = model.get_embedding(xp)
                    if j == 0:
                        loss, logit_compose, rp = classifier(feature, xp, label, representation=args.save_representation)
                        logit_compose = logit_compose.cpu().numpy()
                        max_indices = np.argmax(logit_compose, axis=1)
                        result = np.zeros_like(logit_compose)
                        result[np.arange(result.shape[0]), max_indices] = 1
                        logit_compose = result
                    else:
                        _loss, _logit_compose, rp_ =  classifier(feature, xp, label, representation=args.save_representation)
                        loss += _loss
                        _logit_compose = _logit_compose.cpu().numpy()
                        max_indices = np.argmax(_logit_compose, axis=1)
                        result = np.zeros_like(_logit_compose)
                        result[np.arange(result.shape[0]), max_indices] = 1
                        logit_compose += result
                        rp += rp_
                logit_compose = torch.tensor(logit_compose)
                rp /= args.samples
            losses.append(loss.item() / args.samples)
            logit = logit_compose.max(1)[1]
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            representations.append(rp)

    logits=np.concatenate(logits, 0)
    labels=np.concatenate(labels, 0)
    representations = np.concatenate(representations, 0)
    y_trues, y_preds = labels, logits

    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)   
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "accuracy": float(acc),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "loss": sum(losses) / len(losses),
    }
    logger.info('Accuracy: {}, Recall: {}, Precision: {}, F1: {}'.format(result['accuracy'], result['recall'], result['precision'], result['f1']))
    file_name = (args.test_data_file if ("test" in split) else args.eval_data_file).split("/")[-1].split(".")[0]
    if args.save_representation:
        np.savez(f'{args.result_dir}/representations_{file_name}_{args.inference}.npz', representation=representations, labels=labels)
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
        inputs = batch[0].to(args.device)  
        xps     = batch[1]  
        label  = batch[2].to(args.device)

        if args.inference:
            attribution_score = do_attribution(wrapper, tokenizer, deeplift, inputs, inputs)
            score, tokens, pred = attribution_score['attributions'].numpy(), attribution_score['tokens'], attribution_score['pred'].numpy()
        else:
            for j, id_ in enumerate(xps):
                xp_ids = eval_dataset.get_xp(id_).to(args.device)
                attribution_score = do_attribution(wrapper, tokenizer, deeplift, inputs, xp_ids)
                if j == 0:
                    score, tokens, pred = attribution_score['attributions'], attribution_score['tokens'], attribution_score['pred']
                else:
                    score += attribution_score['attributions']
                    pred += attribution_score['pred']
                    assert tokens == attribution_score['tokens']
            
            score /= args.samples
            score = score / torch.norm(score)
            score = score.numpy()
            pred = pred.numpy()

        score, tokens, pred = score.flatten(), tokens[0], pred.flatten()
        assert len(score) == len(tokens)
        attributions_result.append({
            'attributions': score,
            'tokens': tokens,
            'pred': pred.flatten()
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
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)
    model = Model(model, config, tokenizer, args)
    return model, tokenizer


classifiers = {
    'A1': ClassifierA1,
    'B4': ClassifierB4,
    'B6': ClassifierB6,
    'B7': ClassifierB7,
}
def get_model_2(args):
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels=2
    if args.classifier[0] == 'A':
        model = classifiers[args.classifier](config, args)
    if args.classifier[0] == 'B':

        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)
        model = classifiers[args.classifier](model, config, args)
    return model, tokenizer

                        
def main():
    parser = argparse.ArgumentParser()

    ## Other parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")    
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default="microsoft/unixcoder-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_attribution", action='store_true',
                        help="Whether to run attribution")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--validation_metric', type=str, default='f1',
                        help="metric to use to for model selection based on the validation set")
    parser.add_argument('--code_key', type=str, default="func",
                        help="dataset key for code")
    parser.add_argument('--label_key', type=str, default="target",
                        help="dataset key for labels")
    parser.add_argument('--index_key', type=str, default="idx",
                        help="dataset key for index")
    parser.add_argument('--head', type=int,
                        help="number of dataset examples to cut off")
    parser.add_argument('--freeze_encoder_and_embeddings', action='store_true',
                        help="whether to freeze the model's encoder and embeddings layers")
    parser.add_argument('--tokenize_ast_token', type=int, default=0)
    parser.add_argument('--classifier', type=str, default='A')
    parser.add_argument('--node', type=str, default='node')
    parser.add_argument('--dataset', type=str, default="Devign")
    parser.add_argument('--model_name', type=str, default="causal_chengzi")
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument("--inference", action='store_true',
                        help="Whether to run the inference.")
    parser.add_argument("--result_dir", default=os.path.join(project_dir, "results/UniXcoder"), type=str,
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

    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model, tokenizer = get_model_1(args)
    classifier, tokenizer = get_model_2(args)

    model.to(args.device)
    classifier.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    def load_model(model_, name_):
        output_dir = os.path.join(args.causal_output_dir, f'{name_}.bin')
        model_.load_state_dict(torch.load(output_dir))      
        model_.to(args.device)

    # Training     
    if args.do_train:
        output_dir = os.path.join(args.vanilla_output_dir, 'model.bin')
        model.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        train_dataset = TextDataset(tokenizer, args, args.train_data_file, split='train')
        train(args, train_dataset, model, classifier, tokenizer)
        
    # Evaluation
    if args.do_eval:
        load_model(model, 'model')
        load_model(classifier, 'classifier')
        result = evaluate(args, model, classifier, tokenizer, split='valid')
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        args.save_representation = True
        args.inference = True
        load_model(model, 'model')
        load_model(classifier, 'classifier')
        result = test(args, model, classifier, tokenizer, split='test')
        logger.info("***** Inference results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
        
        args.inference = False
        load_model(model, 'model')
        load_model(classifier, 'classifier')
        result = test(args, model, classifier, tokenizer, split='test')
        logger.info("***** Inference results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
    
    if args.do_attribution:
        args.inference = True
        load_model(model, 'model')
        load_model(classifier, 'classifier')
        attribution(args, model, classifier, tokenizer, args.test_data_file)

        args.inference = False
        load_model(model, 'model')
        load_model(classifier, 'classifier')
        attribution(args, model, classifier, tokenizer, args.test_data_file)    

if __name__ == "__main__":
    main()
