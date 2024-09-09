# Training and testing
Preparing the environment
```python
source /root/env_VulRepair.sh
```
```c
#Need to be able to connect to bugface to download the pre-training model codet5, no please download in advance
# Train
CUDA_VISIBLE_DEVICES=1 python vulrepair_changed.py \
    --dataset=BigVul/diversity/non-uniform \
    --model_name=model.bin \
    --output_dir=./data/ \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_train \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ./data/BigVul/diversity/non-uniform/model/train.log

# Test
CUDA_VISIBLE_DEVICES=1 python vulrepair_changed.py \
    --dataset=BigVul/diversity/uniform \
    --model_name=model.bin_11 \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_test \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --num_beams=50 \
    --eval_batch_size 1
```

> With vulrepair_changed.py and vulrepair_origin.py for comparison, the main modifications are to the way the dataset is entered and the way the model is entered

```python
train_data_whole = datasets.load_dataset("csv",data_files="./data/train.csv",split="train")

val_data_whole = datasets.load_dataset("csv",data_files="./data/valid.csv",split="train")

model_file="/root/model/vulrepair.model"
model.load_state_dict(torch.load(model_file, map_location=args.device))

test_dataset = TextDataset(tokenizer, args, file_type='test',input=args.inference_file)
```
# Dataset preparation
Preparing the environment
```python
source /root/env_VRepair.sh
```
> - Code modified, the current version, can only handle a single file pair, produce a csv, contains ['cve_id', 'cwe_id', 'source', 'target'] four attributes, you need to batch generate repeated use of the script, and the resulting csv can be merged to be used in training or testing
preprocess.py
```python
python3 preprocess.py  {old_file_path} {new_file_path} {cwe_id} {cve_name}
```
> - cve_name can be taken arbitrarily, just a tag does not affect the result, old file is vulnerable file, new file is patched file, do not get the order reversed!
> - This code may not be able to handle all c or c++ languages and cause errors, the processing code comes from VRepair's original data processing code.

# docker Image
```python
docker pull vul4c/vulrepair:2.0
docker run -itd --gpus all --privileged --name {name} vul4c/vulrepair:2.0 bash
```


