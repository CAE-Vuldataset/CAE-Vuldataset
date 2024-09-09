data_dir=''

# "balance" "fidelity"
list_names=("fidelity")

for item1 in "${list_names[@]}"; do

if [ "$item1" == "balance" ]; then
    sub_names=("0.1" "0.01" "0.001")
else
    sub_names=("0.7")
fi

for item2 in "${sub_names[@]}"; do

dataset_dir=${root_dir}/data/Detection/BigVul+Devign/${item1}/${item2}

output_dir=${data_dir}/${item1}/${item2}/with_data_process_saved_model
mkdir -p ${output_dir}

cd ${work_dir}/data_processing
# python process.py \
#     --data_dir ${data_dir}/${item1}/${item2}
# wait

python cpg_GS.py \
    --data_dir ${data_dir}/${item1}/${item2}
wait

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
