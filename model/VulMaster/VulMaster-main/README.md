# Code Execution
git_repo : [https://github.com/soarsmu/VulMaster_](https://github.com/soarsmu/VulMaster_)

1. Preparing the code environment
```python
docker pull pytorch/pytorch:latest
docker run -itd --gpus all --name VulMaster -v /home/nfs/zxh2023/DataEval/VulMaster:/root/VulMaster --privileged -p 9999:22 pytorch/pytorch:latest
docker run -itd --gpus all --name VulMaster -v /home/nfs/m2023-zxh/share/VulMaster:/root/VulMaster --privileged -p 9999:22 pytorch/pytorch:latest
docker exec -it VulMaster bash

apt-get update
apt-get install build-essential wget -y

cd /root/VulMaster
git clone https://github.com/soarsmu/VulMaster_.git
unzip VulMaster-main.zip
cd VulMaster-main
conda create -n vulmaster python=3.8 
conda activate vulmaster
pip install -r requirements.txt
pip install datasets

#pip install tree-sitter==0.2.1
#Don't use the latest version of tree-sitter it will report an error, after trying 0.2.1 it works!
```

2. Preparing pre-trained models
```python
# Preparing CodeT5 model after adaptation 
cd VulMaster-main
mkdir bugfix_pretrain_with_ast
cd bugfix_pretrain_with_ast
wget -O pytorch_model.bin "https://drive.usercontent.google.com/download?id=1057u16sqSf14w51CA0fZt-WJ6FjS2X6I&export=download&authuser=0&confirm=t&uuid=dd9ba052-5fe6-4485-bb24-8c62dc2d9ae6&at=APZUnTU1Ttp61xfoHk1Gl8SEWnq4:1711447936835"

cd VulMaster-main
mkdir c_dataset

# The server can't go over the wall, please download the codet5_base model in advance, and change the model_name in train_model.py to the specific path.
# Original model address : https://huggingface.co/Salesforce/codet5-base/tree/main      


```

3. Training dataset preparation
```python
# This part of the code is not open source, ask the author for it, the code is contained in the attached data_construction folder.
# The required environment is the same as the one prepared in 1, put the data_construction folder in the folder mounted on the host computer.
# Input: csv files processed by VRepair, [deduplicated_data/valid.csv,deduplicated_data/train.csv,deduplicated_data/test.csv].
# Output: generated in vulmaster_data folder [vulmaster_data/dev.json,vulmaster_data/test.json,vulmaster_data/train.json]
# In the attached folder, there are already results in deduplicated_data and vulmaster_data, just replace the inputs with the same name.
# If you need to modify the input and output locations, please query modify prepare_vulmasters_data.py.
conda activate vulmaster
python3 prepare_vulmasters_data.py
```

4. Train
```python
# Modify src/model.py.
# Please replace the generate function in the src/model.py function with a parameter that does not support multi-beam_size, i.e., generate multiple results for a single input.
def generate(self, input_ids, attention_mask, add_loss, max_length):
    self.encoder.n_passages = input_ids.size(1)
    if self.split_psg_subset and input_ids is not None:
        for i in range(self.n_context):
            input_ids = torch.cat([input_ids, input_ids], dim=0)
    print("beam size:",self.beam_size)
    return super().generate(
        input_ids=input_ids.view(input_ids.size(0), -1),
        attention_mask=attention_mask.view(attention_mask.size(0), -1),
        max_length=max_length,
        output_attentions=self.output_attentions,
        return_dict_in_generate=self.output_attentions, num_beams=self.beam_size,
        do_sample=False,
        num_return_sequences=self.beam_size,
    )

# Modify the codet5-base model import statements in train.py and test.py to pre-downloaded code-t5 model paths
# Example
#model_name = 'Salesforce/codet5-base'
model_name = '/root/VulMaster/codet5' 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```
Copy the [dev.json, test.json, train.json] files generated in step 3 to the folder vulnfix_data.
```python
# Train
bash 0a_train.sh
# Test, change beam_size in sh if needed, it was 1.
bash 0b_infer.sh
# The extrapolated results are locatedVulMaster-main/checkpoint_final/final_model/test_results
```

