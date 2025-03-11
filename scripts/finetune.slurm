#!/bin/bash
#SBATCH --job-name=prob-temp-fog
#SBATCH --output=outputs/log_%j.out
#SBATCH --error=outputs/log_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --cpus-per-task=1


REPO_PATH=/users/yshi28/dev/ICL_experiments

echo "Activating virtual environment"
source $REPO_PATH/venv/bin/activate

echo "Running pipeline"
MODEL="openai-community/gpt2-large"
TOTAL_STEPS=400
NUM_FORGET_STEPS=600
FORGET_INTERVAL=50
EVAL_INTERVAL=25
EVAL_DATA_PATH=$REPO_PATH/data/eval_prompts.csv
OUTPUT_DIR=$REPO_PATH/results
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
    --eval_data $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_forget_steps $NUM_FORGET_STEPS \
    --forget_interval $FORGET_INTERVAL \
    --total_steps $TOTAL_STEPS \
    --eval_interval $EVAL_INTERVAL \
    --batch_size 2 \
    --learning_rate $LEARNING_RATE \
    --mask_prob $MASK_PROB \
    --device cuda