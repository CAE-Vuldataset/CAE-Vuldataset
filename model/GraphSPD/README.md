# Paimon
Paimon: Patch Identification Monster

## 0. Introduction

`Paimon` is an extended version of `GraphSPD` [1], which is a graph-based security patch detection program.

Compared with the `GraphSPD`, `Paimon` makes the following changes:

* multiple bug fixes, better modular design, and more diverse changable arguments (see [args.py](./args.py)).
* new (statement) node alignment and graph merging algorithm with less overhead and faster speed.
* accelarate graph slicing with a reconstructed algorithm, based on graph theory and matrix operations. 
* generated the node embeddings with advanced methods, e.g., CodeBERT.
* updated the default hyper-parameters of graph learning models.

Citation: 

[1] Shu Wang, Xinda Wang, Kun Sun, Sushil Jajodia, Haining Wang, and Qi Li, “*[GraphSPD: Graph-Based Security Patch Detection with Enriched Code Semantics](https://ieeexplore.ieee.org/document/10179479)*,” 2023 IEEE Symposium on Security and Privacy (S&P 2023), San Francisco, CA, USA, 2023, pp. 2409-2426, doi: 10.1109/SP46215.2023.10179479.

## 1. Dependencies

`Paimon` can run on the `conda` environment by the following setup. The environment is based on GPU with `cuda 11.7`.

```bash
$ conda create -n paimon python=3.9
$ conda activate paimon
$ conda install numpy scipy transformers
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
$ conda install pyg -c pyg
```

If using `pip`, execute the following commands.

```bash
$ pip install numpy scipy transformers
$ pip install torch torchvision torchaudio
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

```

## 2. Model Training

All commands are executed under the root folder: `<PATH_TO_FOLDER>/Paimon/`, which is refered as `<root>` in the following instructions.

### Option 1: Train the model first time.

If you train the model first time, please use the following commands.
```shell
python paimon.py --task train
```
You can find the avaiable arguments by `python paimon.py --help`

### Option 2: Train the model with processed data.

If you have already processed the dataset, you can train the model using `--train_only` flag, which saves a lot of time in data processing.
```shell
python paimon.py --task train --train_only
```

### Option 3: Train the model of twin networks.

If you do not use PatchCPG, you can train the twin network model by using the following commands.
```shell
python paimon.py --task train --twin 
```
The flag `--train_only` is also avaiable if you have processed dataset.

## 3. Model Testing.

Test the model using the following command.
```shell
python paimon.py --task test
```
If you test the twin network model, please also include `--twin` flag.

## Appendix

The old version of GraphSPD also included in this repo. Please see the [Old_ReadMe](./src/README.md) for instructions.