import argparse
import sys
import json
import os

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
import matplotlib.pyplot as plt

from utils import (
    eval_prompt,
    evaluate,
    LogitAccuracy,
    get_model_and_tokenizer
)


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
        "--num_forget_steps",
        type=int,
        help="the number of initial steps to perform active forgetting (N)",
        required=True,
    )
    parser.add_argument(
        "-k",
        "--forget_interval",
        type=int,
        help="K, interval between two consecutive forgetting operations, K << N",
        required=True,
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        help="the frequency of steps to evaluate the model",
        default=100,
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        help="the total number of training steps to perform",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        default=5e-5,
        type=float,
        help="the learning rate",
    )
    parser.add_argument(
        "--min_length",
        default=4,
        type=int,
        help="the minimum length of the training data",
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="the batch size",
    )
    parser.add_argument(
        "--data_type",
        default="t",
        help="the type of data to use",
    )
    parser.add_argument(
        "--train_data",
        default="wikitext",
        help="path or handler to the training data",
    )
    parser.add_argument(
        "--eval_data",
        default="data/eval_prompts.csv",
        help="path to the evaluation data",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        help="the probability of masking a token in an entry",
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
    parser.add_argument(
        "--output_dir",
        default="results",
        help="path to the output directory",
    )
    
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)

    return args

class LogitDiffAccuracy:
    """
    Compute the accuracy of the model based on the logit difference between the correct and incorrect answers.
    """
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, logit_diff):
        self.total += 1
        if logit_diff > 0:
            self.correct += 1

    def get_acc(self):
        return self.correct / self.total

class TopKAccuracy:
    """
    Compute the accuracy of the model based on the top-k accuracy.
    """
    def __init__(self, k=1):
        self.k = k
        self.correct = 0
        self.total = 0

    def update(self, logits, gold):
        self.total += 1
        # Get top k predictions
        topk_values, topk_indices = torch.topk(logits[0, -1], self.k)
        
        # Check if gold token is in top k predictions
        if gold in topk_indices:
            self.correct += 1

    def get_acc(self):
        return self.correct / self.total


def content_effect_eval(
    model,
    tokenizer,
    data,
    partition,
    valid_idx,
    invalid_idx,
    args,
    device,
):
    """Evaluate the model on the content effect task."""
    assert partition in ["consistent", "inconsistent", "nonsense"]
    data = data[data["dataset_type"] == partition]
    accuracy = []
    skip_count = 0

    for i, row in data.iterrows():
        question = row["prompt"]
        gold = row["gold"]

        prompt = eval_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        logits = model(inputs["input_ids"]).logits

        #Get top 5 tokens
        _, top5_indices = torch.topk(logits[0, -1], 5)
        # convert top5_indices to tokens
        top5_tokens = tokenizer.convert_ids_to_tokens(top5_indices)
        # Skip if neither valid nor invalid token in top 5
        if valid_idx not in top5_indices and invalid_idx not in top5_indices:
            skip_count += 1
            print(f"valid_idx: {valid_idx}, invalid_idx: {invalid_idx}, top5_indices: {top5_indices}")
            continue

        if gold == "yes":
            correct_idx = valid_idx
            incorrect_idx = invalid_idx
        else:
            correct_idx = invalid_idx
            incorrect_idx = valid_idx

        logit_diff = logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]
        accuracy.append(logit_diff > 0)

    if skip_count > 0:
        print(f"Skipped {skip_count} examples")

    return torch.mean(torch.Tensor(accuracy)).item()


def training_loop_temporary_forgetting(
        model,
        tokenizer,
        train_loader,
        eval_data,
        optimizer,
        scheduler,
        num_forget_steps,
        forget_interval,
        mask_prob,
        initializer_range=None,
        device=None,
        args=None
    ):
    """Run the training loop with probabilistic temporary forgetting"""
    # if model name contains "gemma"
    if "gemma" in args.model:
        embedding_layer = model.model.embed_tokens
    elif "gpt" in args.model:
        embedding_layer = model.transformer.wte
    elif "llama" in args.model:
        embedding_layer = model.model.embed_tokens
    else:
        raise ValueError(f"Invalid model: {args.model}")

    metrics = {
        "baseline_acc": {},
        "inconsistent_acc": {},
        "nonsense_acc": {},
        "loss": {}
    }
    
    initializer_range = model.config.initializer_range

    for i, batch in enumerate(tqdm(train_loader)):
        if i > args.total_steps:
            break

        model.train()
        batch = {k: v.to(device) for k, v in batch.items()}

        do_forgetting = i > 0 and i < num_forget_steps and i % forget_interval == 0
        if do_forgetting:
            # Clone the original embeddings
            original_embeddings = embedding_layer.weight.data.clone()

            # Find unique tokens in this batch (ignoring pad token)
            unique_tokens = torch.unique(batch["input_ids"])
            unique_tokens = unique_tokens[unique_tokens != tokenizer.pad_token_id]

            # Randomly select tokens to replace
            mask = torch.rand(len(unique_tokens), device=device) < mask_prob
            tokens_to_replace = unique_tokens[mask]

            # Replace selected token embeddings with new random embeddings
            with torch.no_grad():
                random_embeddings = torch.normal(
                    mean=0.0,
                    std=initializer_range,
                    size=(tokens_to_replace.size(0), embedding_layer.weight.shape[1]),
                    device=device,
                    dtype=embedding_layer.weight.dtype
                )
                embedding_layer.weight.data[tokens_to_replace] = random_embeddings

            # Assert that the LM head weights match the transformer embeddings
            assert torch.all(embedding_layer.weight.data == model.lm_head.weight.data).item()

        outputs = model(**batch)
        loss = outputs.loss
        metrics["loss"][i] = loss.item()
        loss.backward()

        if do_forgetting:
            # Restore original embeddings for the replaced tokens
            with torch.no_grad():
                embedding_layer.weight.data[tokens_to_replace] = original_embeddings[tokens_to_replace]

            assert torch.all(embedding_layer.weight.data == model.lm_head.weight.data).item()
            assert torch.all(embedding_layer.weight.data == original_embeddings).item()

            # Prevent any update to the embedding weights
            embedding_layer.weight.grad = None

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i % args.eval_interval == 0:
            model.eval()
            results = evaluate(model, tokenizer, eval_data, args)
            metrics["baseline_acc"][i] = results["consistent_logit_acc"]
            metrics["inconsistent_acc"][i] = results["inconsistent_logit_acc"]
            metrics["nonsense_acc"][i] = results["nonsense_logit_acc"]
            print(f"Step {i}, Baseline Acc: {metrics['baseline_acc'][i]:.3f}, Inconsistent Acc: {metrics['inconsistent_acc'][i]:.3f}, Nonsense Acc: {metrics['nonsense_acc'][i]:.3f}")

    return metrics


def setup_dataset(tokenizer, args):
    if args.train_data == "wikitext":
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    else:
        raise ValueError(f"Invalid train data: {args.train_data}")

    def filter_empty_examples(example):
        return example["text"].strip() != ""

    filtered_dataset = dataset.filter(filter_empty_examples)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_datasets = filtered_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    def filter_short_examples(example):
        return len(example["input_ids"]) > args.min_length
    
    tokenized_datasets = tokenized_datasets.filter(filter_short_examples)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        tokenized_datasets,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    return train_loader

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


def main():
    args = parse_args()
    model, tokenizer = get_model_and_tokenizer(args.model, args.device)

    eval_data = pd.read_csv(args.eval_data)
    print("Loaded evaluation data with {} examples".format(len(eval_data)))

    train_loader = setup_dataset(tokenizer, args)
    print("Loaded train dataset with {} examples".format(len(train_loader.dataset)))

    total_steps = len(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

    metrics = training_loop_temporary_forgetting(
        model,
        tokenizer,
        train_loader,
        eval_data,
        optimizer,
        scheduler,
        num_forget_steps=args.num_forget_steps,
        forget_interval=args.forget_interval,
        device=args.device,
        args=args,
        mask_prob=args.mask_prob
    )
    
    plot_metrics(metrics, args)

if __name__ == "__main__":
    main()
