root_dir=''

# "completeness2"
# "benchmark" "accuracy" "completeness" "size" "uniqueness"
list_names=("completeness")

for item1 in "${list_names[@]}"; do

sub_names=("0.3" "0.5" "0.7")

for item2 in "${sub_names[@]}"; do

if [ "$item1" == "benchmark" ]; then
    dataset_dir=${root_dir}/data/Detection/BigVul/${item1}
else
    dataset_dir=${root_dir}/data/Detection/BigVul/${item1}/${item2}
fi

cd ${root_dir}/Causality/data_preprocess
python create_spurious_dataset.py \
	--dataset_dir ${dataset_dir}
wait

python create_no_transform_data.py \
	--dataset_dir ${dataset_dir}
wait

cd ${root_dir}/Causality/jobs
bash causality_trainer_vanilla.sh ${dataset_dir}
wait

bash causality_trainer_combine.sh ${dataset_dir}
wait

if [ "$item1" == "benchmark" ]; then
    break
fi

done
done

