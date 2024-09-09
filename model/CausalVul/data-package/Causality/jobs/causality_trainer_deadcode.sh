#!/bin/bash

model="$1"
classifier="$2"
deadcode_setting="$3"
train="$4"
dataset="$5"
test_data_file="$6"
early_layer="$7"
extra_agrs="${@:8}"

node="amp-1"
root_dir=$(echo $(pwd) | awk -F"Causality" '{print $1}')


data_dir="${root_dir}data"
if [ "$deadcode_setting" = "V1" ] 
then
    train_data_file="${data_dir}/${dataset}/train_no_transform_dead_code_with_xp.jsonl"
else
    train_data_file="${data_dir}/${dataset}/train_no_transform.jsonl"
fi
eval_data_file="${data_dir}/${dataset}/valid_no_transform.jsonl"
test_data_file="${data_dir}/${dataset}/${test_data_file}"

seed="123456"
for arg in "$@"
do
    if [[ $arg == --seed=* ]]; then
        seed="${arg#*=}"
    fi
done
saved_model_dir="${root_dir}saved_models_07_23"
result_dir="${root_dir}/Causality/results/${model}/${dataset}/deadcode_${deadcode_setting}/${classifier}/seed_${seed}${early_layer}"
vanilla_output_dir="${saved_model_dir}/${model}/${dataset}-checkpoint-best-f1/causality/seed_${seed}/node_${node}/causal_vanilla/tokenize_ast_tokens_0/"
causal_output_dir="${saved_model_dir}/${model}/${dataset}-checkpoint-best-f1/causality/seed_${seed}/node_${node}/causal_deadcode_${deadcode_setting}/classifier_${classifier}/tokenize_ast_tokens_0/"


case "$train" in
    0)
        extra_agrs+=" --do_test"
        ;;
    1)
        extra_agrs+="  --do_train --do_eval --do_test"
        ;;
    2)
        extra_agrs+="  --do_test --inference"
        ;;
    3)
        extra_agrs+="  --do_attribution"
        ;;
    *)
        echo "Invalid Train, Test, Validation code"
        exit 1
        ;;
esac

extra_args_slug="$(echo $extra_agrs | sed 's/ //g')"
echo "${extra_agrs}"
echo "${extra_args_slug}"
echo ${train_data_file}

cd ${root_dir}/Causality/${model}/code
mkdir -p ${root_dir}/Causality/logs/${model}/deadcode_${deadcode_setting}_${classifier}
python run_causality_deadcode.py  \
    --classifier ${classifier} \
    --node ${node} \
    --dataset ${dataset} \
    --train_data_file=${train_data_file} \
    --eval_data_file=${eval_data_file} \
    --test_data_file=${test_data_file} \
    --vanilla_output_dir ${vanilla_output_dir} \
    --causal_output_dir ${causal_output_dir} \
    --deadcode_setting ${deadcode_setting} \
    --result_dir ${result_dir} \
    --early_layer ${early_layer} \
    ${extra_agrs} > ${root_dir}/Causality/logs/${model}/deadcode_${deadcode_setting}_${classifier}/$1_$2_$3_$4_$5_$6_$7_${extra_args_slug}.log 2>&1
# fi
