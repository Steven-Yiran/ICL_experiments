#! /bin/bash


python evaluate.py --model meta-llama/Llama-3.2-3B-Instruct --valid_token " Yes" --invalid_token " No" --device cuda

# for gemma use "Yes" and "No" as the valid and invalid tokens