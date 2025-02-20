import argparse
import sys

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
from tqdm import tqdm

class LogitAccuracy:
    def __init__(self, data_type):
        self.data_type = data_type
        self.correct = 0
        self.total = 0

    def update(self, correct_logit, incorrect_logit):
        self.total += 1
        if correct_logit > incorrect_logit:
            self.correct += 1

    def get_acc(self):
        return self.correct / self.total

class ExactMatchAccuracy:
    def __init__(self, data_type):
        self.data_type = data_type
        self.correct = 0
        self.total = 0
        
    def update(self, generation, gold):
        self.total += 1
        if generation == gold:
            self.correct += 1

    def get_acc(self):
        return self.correct / self.total


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

    parser.add_argument(
        "-v",
        "--valid_token",
        default=" Yes",
        help="token to use for responding ``valid''",
    )

    parser.add_argument(
        "-i",
        "--invalid_token",
        default=" No",
        help="token to use for responding ``invalid''",
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
    Answer this question with Yes or No. Question: {question} Answer:
    """

def get_model_and_tokenizer(model_str, device):
    """Helper function to get model"""
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    #tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.bfloat16)
    model.to(device)
    return model, tokenizer


def evaluate(model, tokenizer, data, args):
    consistent_logit_acc = LogitAccuracy("consistent")
    inconsistent_logit_acc = LogitAccuracy("inconsistent")
    nonsense_logit_acc = LogitAccuracy("nonsense")
    consistent_exact_match_acc = ExactMatchAccuracy("consistent")
    inconsistent_exact_match_acc = ExactMatchAccuracy("inconsistent")
    nonsense_exact_match_acc = ExactMatchAccuracy("nonsense")
    
    valid_tok = tokenizer(args.valid_token)["input_ids"][1]
    invalid_tok = tokenizer(args.invalid_token)["input_ids"][1]

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
        
        max_token = torch.argmax(outputs[0, -1, :]).item()
        max_prob_tok = tokenizer.decode(max_token).lower().strip()
        
        if max_token not in [valid_tok, invalid_tok]:
            print(f"Max token {max_token} ({tokenizer.decode(max_token)}) not in {valid_tok} or {invalid_tok}")
            continue

        if dataset_type == "consistent":
            consistent_logit_acc.update(outputs[0, -1, correct_tok], outputs[0, -1, incorrect_tok])
            consistent_exact_match_acc.update(max_prob_tok, gold)
        elif dataset_type == "inconsistent":
            inconsistent_logit_acc.update(outputs[0, -1, correct_tok], outputs[0, -1, incorrect_tok])
            inconsistent_exact_match_acc.update(max_prob_tok, gold)
        else:
            nonsense_logit_acc.update(outputs[0, -1, correct_tok], outputs[0, -1, incorrect_tok])
            nonsense_exact_match_acc.update(max_prob_tok, gold)

    return {
        "model": [args.model],
        "consistent_logit_acc": [consistent_logit_acc.get_acc()],
        "inconsistent_logit_acc": [inconsistent_logit_acc.get_acc()],
        "nonsense_logit_acc": [nonsense_logit_acc.get_acc()],
        "consistent_exact_match_acc": [consistent_exact_match_acc.get_acc()],
        "inconsistent_exact_match_acc": [inconsistent_exact_match_acc.get_acc()],
        "nonsense_exact_match_acc": [nonsense_exact_match_acc.get_acc()]
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
