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

def eval_prompt(question):
    return f"""
    Answer this question with Yes or No. Question: {question} Answer:
    """

def get_model_and_tokenizer(model_str, device):
    model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def evaluate(model, tokenizer, data, args):
    consistent_logit_acc = LogitAccuracy("consistent")
    inconsistent_logit_acc = LogitAccuracy("inconsistent")
    nonsense_logit_acc = LogitAccuracy("nonsense")
    
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

        if dataset_type == "consistent":
            consistent_logit_acc.update(outputs[0, -1, correct_tok], outputs[0, -1, incorrect_tok])
        elif dataset_type == "inconsistent":
            inconsistent_logit_acc.update(outputs[0, -1, correct_tok], outputs[0, -1, incorrect_tok])
        else:
            nonsense_logit_acc.update(outputs[0, -1, correct_tok], outputs[0, -1, incorrect_tok])

    return {
        "consistent_logit_acc": consistent_logit_acc.get_acc(),
        "inconsistent_logit_acc": inconsistent_logit_acc.get_acc(),
        "nonsense_logit_acc": nonsense_logit_acc.get_acc(),
    }
