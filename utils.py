import argparse
import sys
import matplotlib.pyplot as plt

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


def evaluate_content_effect(model, tokenizer, data, args):
    consistent_logit_acc = LogitAccuracy("consistent")
    inconsistent_logit_acc = LogitAccuracy("inconsistent")
    nonsense_logit_acc = LogitAccuracy("nonsense")

    # TODO: check behavior match between GPT and LLaMA and Gemma
    if "gpt" in args.model:
        valid_tok = tokenizer.encode(args.valid_token)[0]
        invalid_tok = tokenizer.encode(args.invalid_token)[0]
    else:
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

def plot_metrics(metrics, args):
    """
    Create a plot for the training dynamics of the models.
    """
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot each accuracy metric on the first y-axis
    steps = list(metrics["baseline_acc"].keys())
    steps = [int(x) for x in steps]
    
    baseline_acc = list(metrics["baseline_acc"].values())
    inconsistent_acc = list(metrics["inconsistent_acc"].values()) 
    nonsense_acc = list(metrics["nonsense_acc"].values())

    ax1.plot(steps, baseline_acc, label='Consistent', color='#2ecc71', marker='o')
    ax1.plot(steps, inconsistent_acc, label='Inconsistent', color='#e74c3c', marker='s')
    ax1.plot(steps, nonsense_acc, label='Nonsense', color='#3498db', marker='^')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlim(0, max(steps))

    # Plot loss on the second y-axis
    loss_steps = list(metrics["loss"].keys())
    loss_steps = [int(x) for x in loss_steps]
    losses = list(metrics["loss"].values())
    ax2.plot(loss_steps, losses, label='Loss', color='#9b59b6', linestyle='--')
    ax2.set_ylabel('Loss')

    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f"{args.model.split('/')[-1]} Training Dynamics (N={args.num_forget_steps}, k={args.forget_interval})")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save plot
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/{args.model.split("/")[-1]}_N{args.num_forget_steps}k{args.forget_interval}_dynamics.png')
    plt.close()