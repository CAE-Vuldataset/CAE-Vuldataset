#!/bin/bash

model=codebert
classifier=B6
deadcode_setting=V3
early_layer=4

root_dir=''
node="amp-1"
dataset_dir="$1"

train_data_file=${dataset_dir}/train_no_transform_with_xp.jsonl
eval_data_file=${dataset_dir}/val_no_transform.jsonl
test_data_file=${dataset_dir}/test_no_transform.jsonl

seed="123456"
saved_model_dir=${root_dir}/Causality/jobs/saved_models
vanilla_output_dir=${dataset_dir}/${model}/vanilla_output_dir
causal_output_dir=${dataset_dir}/${model}/causal_output_dir

mkdir -p ${causal_output_dir}

cd ${root_dir}/Causality/${model}/code
batch_size=16

python run_causality_combine.py  \
    --classifier ${classifier} \
    --node ${node} \
    --train_data_file=${train_data_file} \
    --eval_data_file=${eval_data_file} \
    --test_data_file=${test_data_file} \
    --vanilla_output_dir ${vanilla_output_dir} \
    --causal_output_dir ${causal_output_dir} \
    --deadcode_setting ${deadcode_setting} \
    --result_dir ${causal_output_dir} \
    --early_layer ${early_layer} \
    --train_batch_size ${batch_size} \
    --eval_batch_size ${batch_size} \
    --do_train --do_eval --do_test \
    2>&1 | tee ${causal_output_dir}/train.log

