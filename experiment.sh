#! /bin/bash

MODEL="google/gemma-2-2b-it"
NUM_STEPS=1000
FORGET_INTERVAL=50
TOTAL_STEPS=2000
if [[ "$MODEL" =~ "gemma" ]]; then
    VALID_TOKEN=" Yes"
    INVALID_TOKEN=" No"
elif [[ "$MODEL" =~ "llama" ]]; then
    VALID_TOKEN=" Yes"
    INVALID_TOKEN=" No"
else
    VALID_TOKEN="Unknown"
    INVALID_TOKEN="Unknown"
fi

echo "Model: $MODEL"
echo "Valid token: $VALID_TOKEN"
echo "Invalid token: $INVALID_TOKEN"
echo "N: $NUM_STEPS"
echo "k: $FORGET_INTERVAL"
echo "Total steps: $TOTAL_STEPS"

python finetune.py \
    --model $MODEL \
    --train_data wikitext \
    --eval_data data/eval_prompts.csv \
    --output_dir results \
    --num_steps $NUM_STEPS \
    --forget_interval $FORGET_INTERVAL \
    --total_steps $TOTAL_STEPS \
    --eval_interval 100 \
    --batch_size 4 \
    --valid_token $VALID_TOKEN \
    --invalid_token $INVALID_TOKEN \
    --device cuda
