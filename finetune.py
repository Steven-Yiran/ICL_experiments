import argparse
import sys

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch


def parse_args():
    """Establish arguments for run"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        default="google/gemma-2-2b-it",
        help="transformers model to use",
    )
    
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        help="device",
    )

    parser.add_argument(
        "-N",
        default=5000,
        type=int,
        help="the number of initial steps to perform active forgetting",
    )

    parser.add_argument(
        "-k",
        default=100,
        type=int,
        help="the frequency of performing active forgetting, k << N",
    )

    parser.add_argument(
        "-l",
        default=5e-5,
        type=float,
        help="the learning rate",
    )

    parser.add_argument(
        "--data_path",
        default="data/eval_prompts.csv",
        help="path to the evaluation data",
    )

    parser.add_argument(
        "--output_dir",
        default="results",
        help="path to the output directory",
    )
    
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)

    return args


def get_model_and_tokenizer(model_str, device):
    model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    return model, tokenizer


def setup_dataset(tokenizer, data_path):
    data_baseline = {
        "clean_x": ["Sample prompt 1", "Sample prompt 2"],
        "clean_y": ["Correct answer", "Correct answer"],
        "wrong_y": ["Incorrect answer", "Incorrect answer"],
    }

    return data_baseline

def main():
    args = parse_args()

    model, tokenizer = get_model_and_tokenizer(args.model, args.device)

    train_loader = setup_dataset(tokenizer, args.data_path)
if __name__ == "__main__":
    main()
