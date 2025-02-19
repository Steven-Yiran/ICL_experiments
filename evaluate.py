import argparse
import sys

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
from tqdm import tqdm

def parse_args():
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
        "-p",
        "--data_path",
        default="./data/eval_prompts.csv",
        help="path to data",
    )

    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)

    if "-it" in args.model:
        args.it = True
    else:
        args.it = False

    return args

def eval_prompt(question):
    return f"""
    Answer this question with Yes or No. Question: {question} Answer:\n
    """

def get_model_and_tokenizer(model_str, device):
    """Helper function to get model"""
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.bfloat16)
    model.to(device)
    return model, tokenizer


def inference(model, tokenizer, data, args):
    consistent_correct = 0
    inconsistent_correct = 0
    nonsense_correct = 0

    valid_tok = tokenizer(" Yes")["input_ids"][1]
    invalid_tok = tokenizer(" No")["input_ids"][1]

    for i, row in tqdm(data.iterrows(), total=len(data)):
        question = row["prompt"]
        gold = row["gold"]
        dataset_type = row["dataset_type"]

        prompt = eval_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        outputs = model(inputs["input_ids"]).logits

        if gold == "yes":
            correct_tok = valid_tok
            incorrect_tok = invalid_tok
        else:
            correct_tok = invalid_tok
            incorrect_tok = valid_tok
        
        correct = outputs[0, -1, correct_tok] > outputs[0, -1, incorrect_tok]

        if correct.cpu():
            if dataset_type == "consistent":
                consistent_correct += 1
            elif dataset_type == "inconsistent":
                inconsistent_correct += 1
            else:
                nonsense_correct += 1

    return consistent_correct, inconsistent_correct, nonsense_correct


def evaluate(model, tokenizer, data, args):
    consistent_correct, inconsistent_correct, nonsense_correct = inference(model, tokenizer, data, args)

    consistent_total = len(data[data["dataset_type"] == "consistent"])
    inconsistent_total = len(data[data["dataset_type"] == "inconsistent"])
    nonsense_total = len(data[data["dataset_type"] == "nonsense"])

    consistent_acc = consistent_correct / consistent_total
    inconsistent_acc = inconsistent_correct / inconsistent_total
    nonsense_acc = nonsense_correct / nonsense_total

    return {
        "model": [args.model],
        "consistent_acc": [consistent_acc],
        "inconsistent_acc": [inconsistent_acc],
        "nonsense_acc": [nonsense_acc]
    }


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = get_model_and_tokenizer(args.model, args.device)

    # Load data
    data = pd.read_csv(args.data_path)

    # Evaluate
    results = evaluate(model, tokenizer, data, args)

    # Save results
    output_path = f"./results/{'_'.join(args.model.split('/'))}_eval.csv"
    pd.DataFrame.from_dict(results).to_csv(output_path)
