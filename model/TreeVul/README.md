# TreeVul

## Project Description
This replication package contains the dataset and code for our paper `Fine-grained Commit-level Vulnerability Type Prediction By CWE Tree Structure`.

We introduce the task of categorizing the detected security patches into fine-grained vulnerability types.

We collect a large and up-to-date security patch dataset from NVD, consisting of 6,541 patches (i.e., commit in our study) from 1,560 GitHub OSS repositories. We label patches with categories at the third level of the CWE tree.

We propose an approach named TreeVul, which incorporates the structure information of CWE tree as prior knowledge of the classification task.


## Environments

1. OS: Ubuntu

   GPU: NVIDIA GTX 3090.

2. Language: Python (v3.8)

3. CUDA: 11.2

4. Python packages:
   * [PyTorch 1.8.1+cu11](https://pytorch.org/)
   * [AllenNLP 2.4.0](https://allennlp.org/)
   * [Transformers 4.5.1](https://huggingface.co/)
   
   Please refer the official docs for the use of these packages (especially **AllenNLP**).

5. Setup:

   We modified the approach proposed by Liu *et al.* ([Just-In-Time Obsolete Comment Detection and Update, TSE 2021](https://ieeexplore.ieee.org/abstract/document/9664004)) to extract token-level code change information. The replication package of their work is archived at [link](https://github.com/Tbabm/CUP2). You can find the modified code [here](./TreeVul/process_edit.py). **Note that we propose an entirely new approach to encode the code changes.**
   
   We adopt CodeBERT and Bi-LSTM as neural baselines. You can find the code [here](./Baseline/). For the training of Bi-LSTM, we use glove embedding. Please download [Glove](http://nlp.stanford.edu/data/glove.6B.zip) first, then unzip this file and put `glove.6B.300d.txt` into the folder.

   We use [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base) from HuggingFaces Transformer Libarary. You don't need to download the pretrained model by yourself as it will be downloaded the first time you run the code.

   According to the [official docs of PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM), there are known non-determinism issues for RNN functions (i.e., Bi-LSTM component used in our model). Our own experiments also verify this tip. Thus, following the guidance, we enforce deterministic behavior by setting CUDA environment variables `CUBLAS_WORKSPACE_CONFIG=:16:8`. For a fair comparison (setting CUDA environment will also affect other components of nueral networks), all nueral networks in our experiment are trained and evaluated using this CUDA environment.

## Dataset

* `dataset_cleaned.json`: the collected 10,037 security patches (i.e., commit in our study) from 2,260 GitHub OSS repositories. **Note that this dataset has already been cleaned** (e.g., duplicate patches, patches with invalid CWEs, patches without source code, large patches). We have already combined the corresponding CVE ID and CWE ID (also its path) with the collected security patches. **Note that each item in this dataset is a file**, while the classification task should be performed at the commit-level.

 Below, shows an example:

   ```
    {
        "cve_list": "CVE-2010-5332",
        "cwe_list": "CWE-119",
        "path_list": [
            [
                "CWE-664",
                "CWE-118",
                "CWE-119"
            ]
        ],
        "repo": "torvalds/linux",
        "commit_id": "0926f91083f34d047abc74f1ca4fa6a9c161f7db",
        "user": "David S. Miller",
        "commit_date": "2010-10-25T19:14:11Z",
        "msg": "mlx4_en: Fix out of bounds array access\n\nWhen searching for a free entry in either mlx4_register_vlan() or\nmlx4_register_mac(), and there is no free entry, the loop terminates without\nupdating the local variable free thus causing out of array bounds access. Fix\nthis by adding a proper check outside the loop.\n\nSigned-off-by: Eli Cohen <eli@mellanox.co.il>\nSigned-off-by: David S. Miller <davem@davemloft.net>",
        "Total_LOC_REM": 0,
        "Total_LOC_ADD": 11,
        "Total_LOC_MOD": 11,
        "Total_NUM_FILE": 1,
        "Total_NUM_HUNK": 2,
        "file_name": "drivers/net/mlx4/port.c",
        "file_type": "modified",
        "PL": "C",
        "LOC_REM": 0,
        "LOC_ADD": 11,
        "LOC_MOD": 11,
        "NUM_HUNK": 2,
        "REM_DIFF": [
            "",
            ""
        ],
        "ADD_DIFF": [
            "if (free < 0) { err = -ENOMEM; goto out; }",
            "if (free < 0) { err = -ENOMEM; goto out; }"
        ]
    }
   ```
   *cve_list* is the correspoding CVE ID.

   *cwe_list* is the correspoding CWE ID.

   *path_list* is the *path* (from the root node of the CWE tree to the target node) of the corresponding CWE.

   *repo* is GitHub repository name.

   *commit_id* is the commit id of the security patch.

   *user* is the the name of the committer.

   *commit_date* is the date of the commit.

   *msg* is the commit message.

   *Total_LOC_ADD*, *Total_LOC_REM*, *Total_LOC_MOD* are the number of code lines that are removed, added or modified (sum of removed and added) in the entire commit, respectively.

   *Total_NUM_FILE*, *Total_NUM_HUNK* are the number of files and hunks within this commit, respectively.

   **the following features are related to the sepcific files changed in the commit:**

   *file_name* is the name of the changed file.

   *PL* is the corresponding programming language inferred using file extension.

   *LOC_ADD*, *LOC_REM*, *LOC_MOD* is the number of code lines that are removed, added or modified (sum of removed and added) in the file, respectively.

   *ADD_DIFF*, *REM_DIFF* is the list of removed and added code segments of each hunk within the file, respectively.

* `dataset_cleaned_level3.json`: Our task is to classify security patches into fine-grained categories at the third level of the CWE tree. We remove patches whose CWE categories are at level 1 or 2 from `dataset_cleaned.json`. This dataset contains 6,541 security patches (i.e., commit in our study) from 1,560 GitHub OSS repositories

* `test_set.json`: test set used in the experiments. We split the `dataset_cleaned_level3.json` using **stratified random sampling** with a ratio of 8:1:1 (for train set, validation set and test set, respectively). 
   
* `train_set.json`: train set used in the experiments.

* `validation_set.json`: validation set used in the experiments.

## File Organization
There are several files and three directoris (`Baseline_ml` - machine learning baselines, `Baseline` - deep learning baselines, `TreeVul` - our proposed approach and its variants used in ablation studies, `data` - all the data used in the experiments).



### Files

* `predict.py`: scripts for evaluations of neural models. You can use it for evaluations of TreeVul, variants used in the ablations (i.e., TreeVul-t and TreeVul-h), and neural baselines (i.e., Bi-LSTM and CodeBERT). We have already provided logics to handle the differences of these approaches
* `cal_metrics.py`: scripts for calculating the metrics. We implement three wrappers for machine learning models,  neural models without tree structure, and neural models with tree structure, respectively.
* `test_config.json`: config for test of neural baselins (Bi-LSTM and CodeBERT) and TreeVul-h. These models directly map an input patch to its CWE category.
* `test_config_tree.json`: config for test of TreeVul and TreeVul-t. These two models incoporate the knowledge of CWE tree structure with a hierarchical and chained model architecture. Their inferences are based on our proposed tree structure aware and beam searched based inference algorithm.
* `utils.py`: all the util fuctions: e.g., divide the dataset into training set and testing set, build CWE tree, generate the CWE path, preprocess the dataset


### Directories

* `TreeVul/`: code of TreeVul, together with two variants, i.e., TreeVul-t (remove token-level code change information) and TreeVul-h (remove the design of hierarchical and chained model architecture, which is used to incoporate the CWE tree structure information).

   * `reader_treevul_hunk.py`: dataset reader for TreeVul
   * `model_treevul.py`: model architecture for TreeVul and TreeVul-t
   * `config_treevul.json`: config for the training of TreeVul

   * `reader_ablation_noedit_hunk.py`: dataset reader for TreeVul-t
   * `config_ablation_noedit.json`: config for the training of TreeVul-t

   * `reader_ablation_notree_hunk.py`: dataset reader for TreeVul-h
   * `model_ablation_notree.py`: model architecture for TreeVul-h
   * `config_abaltion_notree.json`: config for the training of TreeVul-h

   * `reader_cwe.py`: dataset reader for loading CWE descriptions, which are used to generate label embeddings
   * `custom_PTM_embedder.py`: custom embedder, where we add support for incoporating token-level code change information
   * `custom_trainer.py`: custom trainer for supporting custom callbacks
   * `custom_metric.py`: custom validation, where we calculate our self-defined metrics. Same to the `cal_metrics.py`
   * `custom_modules.py`: custom modules used in model construction
   * `callbacks.py`: callbacks used in training, e.g., preparing CWE tree before model training
   * `process_edit.py`: extract token-level code change information
  

* `Baseline/`: code for two neural baselines (Bi-LSTM and CodeBERT).

   * `tokenizer.py`: tokenizer used for Bi-LSTM

   * `reader_baseline_bilstm_hunk.py`: dataset reader for Bi-LSTM
   * `model_baseline_bilstm.py`: model architecture for Bi-LSTM
   * `config_baseline_bilstm.json`: config for the training of Bi-LSTM
   
   * `reader_baseline_codebert_hunk.py`: dataset reader for CodeBERT
   * `model_baseline_codebert.py`: model architecture for CodeBERT
   * `config_baseline_codebert.json`: config for the training of CodeBERT
  
   * `custom_metric.py`: custom validation, where we calculate our self-defined metrics. Same to the one under `TreeVul/`
   * `custom_modules.py`: custom modules used in model construction


* `Baseline_ml/`: code for five machine learning baselines.
  
  * `tokenizer.py`: tokenizer used for machine learning baselines. Same to the one used for Bi-LSTM
  * `reader_baseline_ml.py`: dataset reader for machine learning baselines
  * `ml_baseline.py`: implementation of five machine learning baselines, i.e., Random Forest (RF), LR (Linear Regression), SVM (Support Vector Machine), XGB (XGBoost), KNN (K-Nearest Neighbour).
  * `cal_metrics.py`: evaluations of machine learning baselines. Same to the one under the parent directory.


* `data/`: all the data (in json format) used in the experiments. You can build them using the correspoding fuctions in [utils.py](utils.py).

   * `cve_data.json`: all the CVE records
   * `cwe_tree.json`: We orgnize the CWE entries in the [Research View](https://cwe.mitre.org/data/definitions/1000.html) into a tree-like structure
   * `cwe_path.json`: map each CWE category into the corresponding CWE path, i.e., a path start from the root of the CWE tree to the target category
   * `valid_cwe_tree.json`: CWE tree that only contains the categories appear in our train set and validation set
   * `valid_cwes.json`: map categories at each level of the CWE tree into indexes (only includes those appear in our train set and validation set)

## Train & Test

For running the machine learning baselines, enter the directory `Baseline` and run `python ml_baselines.py` (which also handles the evaluation)

For training of all the neural models, i.e, TreeVul, its variants (TreeVul-t and TreeVul-h) and neural baselins (i.e., Bi-LSTM and CodeBERT) (we implement these models using AllenNLP package),

Open terminal in the parent folder and run
`CUBLAS_WORKSPACE_CONFIG=:16:8 allennlp train <config file> -s <serialization path> --include-package <package name>`. Please refer to official docs of [AllenNLP](https://allennlp.org/) for more details. The reason of setting CUDA environment is explained in Section Environments.

For example, with `CUBLAS_WORKSPACE_CONFIG=:16:8 allennlp train TreeVul/config_treevul.json -s TreeVul/out_treevul/ --include-package TreeVul`, you can get the output folder at `TreeVul/out_treevul/` and log information showed in the console.

For test of neural models, please follow the comments in [predict.py](predict.py). We have already provided logics to handle the differences between these approaches. You can run the test function to get the detailed results of each sample (saved in file `<model>_result.json`) and the metrics (saved in file `<model>_metric.json`). You don't need to use `cal_metrics.py` to calculate the metrics again for nueral models.
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:16:8 allennlp train TreeVul/config_treevul.json -s /home/nfs/zxh2023/TreeVul/TreeVul/zxh_data/uniqueness/0.5/out_treevul/ --include-package TreeVul