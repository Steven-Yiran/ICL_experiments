#!/bin/bash

# options: meta-llama/Llama-3.2-3B-Instruct, google/gemma-2-2b-it meta-llama/Llama-3.2-1B
MODEL="meta-llama/Llama-3.2-1B"
TOTAL_STEPS=400
NUM_FORGET_STEPS=600
FORGET_INTERVAL=50
EVAL_INTERVAL=25
LEARNING_RATE=5e-6
MASK_PROB=0.1

echo "Model: $MODEL"
echo "N: $NUM_FORGET_STEPS"
echo "k: $FORGET_INTERVAL"
echo "Total steps: $TOTAL_STEPS"
echo "Eval interval: $EVAL_INTERVAL"
echo "Learning rate: $LEARNING_RATE"
echo "Mask prob: $MASK_PROB"

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
    --learning_rate $LEARNING_RATE \
    --mask_prob $MASK_PROB \
    --device cuda
