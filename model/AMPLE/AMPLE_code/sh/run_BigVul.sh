data_dir=''

# "benchmark" "accuracy" "completeness" "size" "uniqueness"
list_names=("uniqueness")

for item1 in "${list_names[@]}"; do

sub_names=("0.3" "0.5" "0.7")

for item2 in "${sub_names[@]}"; do

if [ "$item1" == "benchmark" ]; then
    dataset_dir=${data_dir}/${item1}
else
    dataset_dir=${data_dir}/${item1}/${item2}
fi

output_dir=${data_dir}/${item1}/${item2}/saved_model
mkdir -p ${output_dir}

# cd ${word_dir}/data_processing
# python process.py \
#     --data_dir ${dataset_dir}
# wait

# python cpg_GS.py \
#     --data_dir ${dataset_dir}
# wait

cd ${word_dir}/AMPLE_code

batch_size=128
python main.py \
    --model_type devign \
    --input_dir ${data_dir}/${item1}/${item2} \
    --batch_size ${batch_size} \
    --output_dir ${output_dir}
wait

if [ "$item1" == "benchmark" ]; then
    break
fi



done
done
