#! /bin/bash


# python evaluate.py --model meta-llama/Llama-3.2-3B-Instruct --valid_token " Yes" --invalid_token " No" --device cuda

# for gemma use "Yes" and "No" as the valid and invalid tokens
# for llama use " Yes" and " No" as the valid and invalid tokens
python finetune.py \
    --model google/gemma-2-2b-it \
    --train_data wikitext \
    --eval_data data/eval_prompts.csv \
    --output_dir results \
    --num_steps 2000 \
    --forget_interval 200 \
    --batch_size 4 \
    --valid_token "Yes" \
    --invalid_token "No" \
    --device cuda
