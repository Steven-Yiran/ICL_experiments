#! /bin/bash


# python evaluate.py --model meta-llama/Llama-3.2-3B-Instruct --valid_token " Yes" --invalid_token " No" --device cuda

# for gemma use "Yes" and "No" as the valid and invalid tokens

python finetune.py \
    --model openai-community/gpt2 \
    --train_data data/train-00000-of-00001.parquet \
    --eval_data data/eval_prompts.csv \
    --output_dir results \
    --device cpu
