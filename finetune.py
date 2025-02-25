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

from evaluate import eval_prompt


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
        "--num_steps",
        default=5000,
        type=int,
        help="the number of initial steps to perform active forgetting",
    )
    parser.add_argument(
        "-k",
        "--forget_interval",
        default=100,
        type=int,
        help="the frequency of performing active forgetting, k << N",
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


def get_model_and_tokenizer(model_str, device):
    model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def content_effect_eval(
    model,
    tokenizer,
    data,
    partition,
    args,
    device,
):
    """Evaluate the model on the content effect task."""
    assert partition in ["consistent", "inconsistent", "nonsense"]
    if partition == "consistent":
        data = data[data["dataset_type"] == "consistent"]
    elif partition == "inconsistent":
        data = data[data["dataset_type"] == "inconsistent"]
    elif partition == "nonsense":
        data = data[data["dataset_type"] == "nonsense"]
    
    accuracy = []
    
    valid_index = tokenizer(args.valid_token)["input_ids"][1]
    invalid_index = tokenizer(args.invalid_token)["input_ids"][1]
    for i, row in data.iterrows():
        question = row["prompt"]
        gold = row["gold"]

        prompt = eval_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        logits = model(inputs["input_ids"]).logits

        if gold == "yes":
            correct_idx = valid_index
            incorrect_idx = invalid_index
        else:
            correct_idx = invalid_index
            incorrect_idx = valid_index
        
        max_token = torch.argmax(logits[0, -1, :]).item()
        max_prob_tok = tokenizer.decode(max_token).lower().strip()

        if max_token not in [valid_index, invalid_index]:
            print(f"Max token {max_token} ({tokenizer.decode(max_token)}) not in {valid_index} or {invalid_index}")
            continue    

        logit_diff = logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]
        accuracy.append(logit_diff > 0)

    return torch.mean(torch.Tensor(accuracy)).item()


def training_loop_temporary_forgetting(
        model,
        tokenizer,
        train_loader,
        eval_data,
        optimizer,
        scheduler,
        num_steps,
        forget_interval,
        mask_prob=0.1,
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

    metrics = {"baseline_acc": {}, "inconsistent_acc": {}, "nonsense_acc": {}}
    
    initializer_range = model.config.initializer_range

    for i, batch in enumerate(tqdm(train_loader)):
        if i > num_steps:
            break

        model.train()
        batch = {k: v.to(device) for k, v in batch.items()}

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

        loss.backward()

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

        if i % forget_interval == 0:
            model.eval()
            baseline_acc = content_effect_eval(
                model, tokenizer, eval_data, "consistent", args, device
            )
            inconsistent_acc = content_effect_eval(
                model, tokenizer, eval_data, "inconsistent", args, device
            )
            nonsense_acc = content_effect_eval(
                model, tokenizer, eval_data, "nonsense", args, device
            )
            metrics["baseline_acc"][i] = baseline_acc
            metrics["inconsistent_acc"][i] = inconsistent_acc
            metrics["nonsense_acc"][i] = nonsense_acc
            print(f"Step {i}, Baseline Acc: {baseline_acc}, Inconsistent Acc: {inconsistent_acc}, Nonsense Acc: {nonsense_acc}")

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

def main():
    args = parse_args()

    model, tokenizer = get_model_and_tokenizer(args.model, args.device)
    train_loader = setup_dataset(tokenizer, args)
    print("Loaded train dataset with {} examples".format(len(train_loader.dataset)))

    eval_data = pd.read_csv(args.eval_data)
    print("Loaded evaluation data with {} examples".format(len(eval_data)))

    total_steps = len(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

    metrics = training_loop_temporary_forgetting(
        model,
        tokenizer,
        train_loader,
        eval_data,
        optimizer,
        scheduler,
        num_steps=args.num_steps,
        forget_interval=args.forget_interval,
        device=args.device,
        args=args
    )
    
    model_name = args.model.split("/")[-1]
    hyperparams = f"N{args.num_steps}k{args.forget_interval}"
    output_path = f"{args.output_dir}/{'_'.join(args.model.split('/'))}_{hyperparams}_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {output_path}")

if __name__ == "__main__":
    main()
