# Function-Level Software Vulnerability Assessment
This is the README file for the reproduction package of the paper: "On the Use of Fine-grained Vulnerable Code Statements for Software Vulnerability Assessment Models".

The package contains the following artefacts:
1. Data: contains the code of the 1,782 vulnerable methods along with the line numbers of cosmetic statements (e.g., comments/empty lines), vulnerable statements and program slicing context of such statements.
2. Code: contains the source code we used in our work. It's noted that we setup the code to run on a supercomputing cluster that runs on Slurm. Therefore, most of the code must be submitted using bash script (.sh) file, but our code can still be run locally by executing the python file directly.

Before running any code, please install all the required Python packages using the following command: `pip install -r requirements.txt`

## Feature Generation and Baseline Models
1. Train and infer the single-input code features for RQ1-RQ2 (Bag-of-Tokens, Bag-of-Subtokens, Word2vec and fastText) by running `python Code/single/extract_features_single.py`.
2. Train and infer the single-input CodeBERT features for RQ1-RQ2 by running `python Code/single/extract_features_single_codebert.py`.
3. Train and infer the double-input code features for RQ3 (Bag-of-Tokens, Bag-of-Subtokens, Word2vec and fastText) by running `python Code/double/extract_features_double.py`.
4. Train and infer the double-input CodeBERT features for RQ3 by running `python Code/double/extract_features_single_codebert.py`.

Note that the generation of CodeBERT features can be accelerated with GPUs (i.e., moving the models to GPUs in Pytorch). The CodeBERT pre-trained model would be downloaded from the Huggingface library on-the-fly.

## Training and Evaluation
1. Train and evaluate single-input models for RQ1-RQ2 by running `evaluate_models_single.sh` (slurm script)
2. Train and evaluate double-input models for RQ3 by running `evaluate_models_double.sh` (slurm script)

Note that after these training/evaluation scripts finish, they will generate output folders, i.e., `Code/ml_results_single/` and `Code/ml_results_double/` containing the results (.csv files) for the single-input and double-input models, respectively.

These .csv result files can then be used for analysis and comparison as described in the paper.
