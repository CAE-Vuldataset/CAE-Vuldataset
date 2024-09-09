data_dir=''
# accuracy completeness2 size uniqueness
list_names=("uniqueness")

for item1 in "${list_names[@]}"; do

if [ "$item1" == "accuracy" ]; then
    sub_names=("0.5" "0.7")
else
    sub_names=("0.3" "0.5" "0.7")
fi

for item2 in "${sub_names[@]}"; do

if [ "$item1" == "benchmark" ]; then
    dataset_dir=${root_dir}/data/Detection/Devign/${item1}
else
    dataset_dir=${root_dir}/data/Detection/Devign/${item1}/${item2}
fi

output_dir=${data_dir}/${item1}/${item2}/sinked_timeout_5_saved_model
mkdir -p ${output_dir}

# cd ${work_dir}/data_processing
# python process.py \
#     --data_dir ${data_dir}/${item1}/${item2}
# wait

# python cpg_GS.py \
#     --data_dir ${data_dir}/${item1}/${item2}
# wait

cd ${work_dir}/AMPLE_code

batch_size=128
python main.py \
    --model_type devign \
    --input_dir ${data_dir}/${item1}/${item2} \
    --batch_size ${batch_size} \
    --output_dir ${output_dir}
wait

done
done
