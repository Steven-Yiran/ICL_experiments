#!/bin/bash

# options: meta-llama/Llama-3.2-3B-Instruct, google/gemma-2-2b-it meta-llama/Llama-3.2-1B
MODEL="meta-llama/Llama-3.2-3B-Instruct"
TOTAL_STEPS=50
NUM_FORGET_STEPS=0
FORGET_INTERVAL=0
EVAL_INTERVAL=2
VALID_TOKEN=" Yes" # "Yes"
INVALID_TOKEN=" No" # "No"
LEARNING_RATE=5e-6
MASK_PROB=0.05

echo "Model: $MODEL"
echo "Valid token: $VALID_TOKEN"
echo "Invalid token: $INVALID_TOKEN"
echo "N: $NUM_FORGET_STEPS"
echo "k: $FORGET_INTERVAL"
echo "Total steps: $TOTAL_STEPS"

python finetune.py \
    --model $MODEL \
    --train_data wikitext \
    --eval_data data/eval_prompts.csv \
    --output_dir results \
    --num_forget_steps $NUM_FORGET_STEPS \
    --forget_interval $FORGET_INTERVAL \
    --total_steps $TOTAL_STEPS \
    --eval_interval $EVAL_INTERVAL \
    --batch_size 2 \
    --valid_token $VALID_TOKEN \
    --invalid_token $INVALID_TOKEN \
    --learning_rate $LEARNING_RATE \
    --mask_prob $MASK_PROB \
    --finetune \
    --device cuda
