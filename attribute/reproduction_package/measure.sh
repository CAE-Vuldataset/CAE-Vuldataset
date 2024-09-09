#!/bin/bash

if [ $1 = "accuracy" ]
then
  python dq_analysis/attributes/accuracy.py prepare $2
  echo "Computing $1 measurement"
  python dq_analysis/attributes/accuracy.py measure $2
elif [ $1 = "completeness" ]
then
  echo "Checking for truncated files..."
  python dq_analysis/attributes/completeness.py prepare $2
  echo "Computing $1 measurement"
  python dq_analysis/attributes/completeness.py measure $2
elif [ $1 = "consistency" ]
then
  echo "Identifying exact duplicates..."
  python dq_analysis/attributes/consistency.py prepare $2
  echo "Computing $1 measurement"
  python dq_analysis/attributes/consistency.py measure $2
elif [ $1 = "immediacy" ]
then
  echo "Tokenizing dataset..."
  python dq_analysis/attributes/immediacy.py prepare $2
  echo "Computing $1 measurement"
  python dq_analysis/attributes/immediacy.py measure $2
elif [ $1 = "uniqueness" ]
then
  echo "Identifying code clones..."
  python dq_analysis/attributes/uniqueness.py prepare $2
  echo "Computing $1 measurement"
  python dq_analysis/attributes/uniqueness.py measure $2
else
  echo "Unsupported command line argument: $1"
fi
