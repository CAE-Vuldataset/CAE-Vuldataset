#!/bin/bash

## old version - [not recommended].
# python src/get_dataset.py
# python src/parse_graphs.py
# python src/embed_graphs.py
# python src/gnn_patch.py

## PGCN.
## train phase.
python paimon.py --task train --max_epoch 1000 --train_only
## test phase.
# python paimon.py --task test

## Twin PGCN.
## train phase.
# python paimon.py --task train --twin --max_epoch 100
## test phase.
# python paimon.py --task test --twin 