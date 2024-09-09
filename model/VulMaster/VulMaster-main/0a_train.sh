#!/bin/bash
name=final_model
dataset=BigVul/diversity/non-uniform
CUDA_VISIBLE_DEVICES=0 SLURM_NTASKS=1	python train_model.py \
        --dataset ${dataset} \
        --train_data  vulfix_data/${dataset}/train.json \
        --eval_data   vulfix_data/${dataset}/dev.json \
	      --model_size base \
        --per_gpu_train_batch_size 2 \
        --per_gpu_eval_batch_size 2 \
        --accumulation_steps 32 \
        --total_steps 80000 \
        --eval_freq 4000 \
        --save_freq 4000 \
        --n_context 10 \
	      --beam_size 1 \
	      --use_adapted_model \
	      --adapted_model_path bugfix_pretrain_with_ast/pytorch_model.bin \
	      --text_maxlength 512 \
	      --answer_maxlength 512 \
        --add_loss binary \
        --cat_emb \
        --name ${name} \
        --checkpoint_dir checkpoint_final
