#!/usr/bin/env python
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "gpt2-large"
MIN_LENGTH=4

## stuff change
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy behavioral analysis data (replace with your actual data)
data_baseline = {
    "clean_x": ["Sample prompt 1", "Sample prompt 2"],
    "clean_y": ["Correct answer", "Correct answer"],
    "wrong_y": ["Incorrect answer", "Incorrect answer"],
}
data_nonce = data_baseline  # or define differently if needed

# Dummy vocabulary placeholder (replace with your actual vocabulary if needed)
vocabulary = None


def logits_to_logit_diff_t(tokenizer, logits, correct_answer, incorrect_answer):
    """Compute logit difference between correct and incorrect answers."""
    correct_idx = tokenizer.encode(correct_answer)[0]
    incorrect_idx = tokenizer.encode(incorrect_answer)[0]
    return logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]


def behavioral_analysis_t(
    model, tokenizer, vocabulary, q_type, correct_label, incorrect_label, device, nonce=False, verbose=True
):
    """Evaluate the model on a behavioral task.

    Uses either baseline or nonce data depending on the `nonce` flag.
    """
    data = data_nonce if nonce else data_baseline
    accuracy = []
    total_samples = len(data["clean_x"])
    for i in range(total_samples):
        if i % 50 == 0 and verbose:
            print(f"Progress: {round(i / total_samples, 3)}")
        clean_prompt = data[q_type][i]
        clean_label = data[correct_label][i]
        cf_label = data[incorrect_label][i]
        clean_tokens = tokenizer.encode(clean_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(clean_tokens).logits
        accuracy.append(logits_to_logit_diff_t(tokenizer, logits, clean_label, cf_label) > 0)
    if verbose:
        print(f"Model Accuracy: {torch.mean(torch.Tensor(accuracy))}")
    return torch.mean(torch.Tensor(accuracy))


def setup_model(num_random_tokens=2000):
    """Set up the model and tokenizer."""
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Add new tokens and resize embeddings
    tokenizer.add_tokens([f" [NEW_TOKEN{i}]" for i in range(num_random_tokens)])
    model.resize_token_embeddings(len(tokenizer))
    print(f"Updated vocab size: {len(tokenizer)}, Embedding matrix size: {model.transformer.wte.weight.shape}")
    # Set pad token to eos
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def setup_dataset(tokenizer, data_path="../../train-00000-of-00001.parquet"):
    """Load and tokenize the dataset from a parquet file."""
    dataset = load_dataset("parquet", data_files=data_path, split="train")

    def filter_empty_examples(example):
        return example["text"].strip() != ""

    filtered_dataset = dataset.filter(filter_empty_examples)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = filtered_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    def filter_short_examples(example):
        return len(example["input_ids"]) > MIN_LENGTH

    tokenized_datasets = tokenized_datasets.filter(filter_short_examples)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        tokenized_datasets,
        batch_size=2,  # Adjust batch size as needed
        shuffle=False,
        collate_fn=data_collator,
    )
    return train_loader


def training_loop_temporary_forgetting(model, tokenizer, train_loader, optimizer, scheduler, num_steps=2000, mask_prob=0.1, initializer_range=None, vocabulary=None):
    """Run the training loop with probabilistic temporary forgetting"""
    metrics = {"baseline_acc": {}, "nonce_acc": {}}
    if initializer_range is None:
        initializer_range = model.config.initializer_range

    for i, batch in enumerate(tqdm(train_loader)):
        if i > num_steps:
            break

        model.train()
        batch = {k: v.to(device) for k, v in batch.items()}

        # Clone the original embeddings
        original_embeddings = model.transformer.wte.weight.data.clone()

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
                size=(tokens_to_replace.size(0), model.transformer.wte.weight.shape[1]),
                device=device,
            )
            model.transformer.wte.weight.data[tokens_to_replace] = random_embeddings

        # Assert that the LM head weights match the transformer embeddings
        assert torch.all(model.transformer.wte.weight.data == model.lm_head.weight.data).item()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()

        # Restore original embeddings for the replaced tokens
        with torch.no_grad():
            model.transformer.wte.weight.data[tokens_to_replace] = original_embeddings[tokens_to_replace]

        assert torch.all(model.transformer.wte.weight.data == model.lm_head.weight.data).item()
        assert torch.all(model.transformer.wte.weight.data == original_embeddings).item()

        # Prevent any update to the embedding weights
        model.transformer.wte.weight.grad = None

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            model.eval()
            baseline_acc = behavioral_analysis_t(
                model, tokenizer, vocabulary, "clean_x", "clean_y", "wrong_y", device, nonce=False, verbose=False
            ).item()
            nonce_acc = behavioral_analysis_t(
                model, tokenizer, vocabulary, "clean_x", "clean_y", "wrong_y", device, nonce=True, verbose=False
            ).item()
            metrics["baseline_acc"][i] = baseline_acc
            metrics["nonce_acc"][i] = nonce_acc
            print(f"Step {i}, Baseline Acc: {baseline_acc}, Nonce Acc: {nonce_acc}")

    return metrics


def main():
    model, tokenizer = setup_model()
    train_loader = setup_dataset(tokenizer)

    total_steps = len(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

    initializer_range = model.config.initializer_range

    metrics = training_loop_temporary_forgetting(
        model,
        tokenizer,
        train_loader,
        optimizer,
        scheduler,
        num_steps=2000,
        initializer_range=initializer_range,
        vocabulary=vocabulary,
    )

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
