#!/bin/bash

model=codebert
dataset_dir="$1"

node="amp-1"
root_dir=''

train_data_file=${dataset_dir}/train_no_transform.jsonl
eval_data_file=${dataset_dir}/val_no_transform.jsonl
test_data_file=${dataset_dir}/test_no_transform.jsonl

seed="123456"
vanilla_output_dir=${dataset_dir}/${model}/vanilla_output_dir
mkdir -p ${vanilla_output_dir}

cd ${root_dir}/Causality/${model}/code
batch_size=16

python run_${model}_vanilla.py \
    --node ${node} \
    --train_data_file=${train_data_file} \
    --eval_data_file=${eval_data_file} \
    --test_data_file=${test_data_file} \
    --output_dir ${vanilla_output_dir} \
    --tokenize_ast_token 0 \
    --result_dir ${vanilla_output_dir} \
    --train_batch_size ${batch_size} \
    --eval_batch_size ${batch_size} \
    --do_train --do_eval --do_test \
    2>&1 | tee ${vanilla_output_dir}/train.log
