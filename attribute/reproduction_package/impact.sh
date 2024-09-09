#!/bin/bash

echo "Running $1 experiment on $2"
python dq_analysis/svp/experiments.py $1 $2 linevul
