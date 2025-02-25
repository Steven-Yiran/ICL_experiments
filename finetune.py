import argparse
import sys
import json
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
        "--num_steps_between_forgets",
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
        default=16,
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
        default="data/train-00000-of-00001.parquet",
        help="path or handler to the training data",
    )
    parser.add_argument(
        "--eval_data",
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
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def logits_to_logit_diff_t(tokenizer, logits, correct_answer, incorrect_answer):
    """Compute logit difference between correct and incorrect answers."""
    correct_idx = tokenizer.encode(correct_answer)[0]
    incorrect_idx = tokenizer.encode(incorrect_answer)[0]
    return logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]


def behavioral_analysis_t(
    model,
    tokenizer,
    data,
    vocabulary,
    q_type,
    correct_label,
    incorrect_label,
    device,
    nonce=False,
    verbose=True
):
    """Evaluate the model on a behavioral task.

    Uses either baseline or nonce data depending on the `nonce` flag.
    """
    #data = data_nonce if nonce else data_baseline
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


def training_loop_temporary_forgetting(
        model,
        tokenizer,
        train_loader,
        optimizer,
        scheduler,
        num_steps,
        forget_interval,
        mask_prob=0.1,
        initializer_range=None,
        vocabulary=None,
        device=None
    ):
    """Run the training loop with probabilistic temporary forgetting"""
    metrics = {"baseline_acc": {}, "inconsistent_acc": {}, "nonce_acc": {}}
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
                dtype=model.transformer.wte.weight.dtype
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

        # if i % 100 == 0:
        #     model.eval()
        #     baseline_acc = behavioral_analysis_t(
        #         model, tokenizer, vocabulary, "clean_x", "clean_y", "wrong_y", device, nonce=False, verbose=False
        #     ).item()
        #     nonce_acc = behavioral_analysis_t(
        #         model, tokenizer, vocabulary, "clean_x", "clean_y", "wrong_y", device, nonce=True, verbose=False
        #     ).item()
        #     metrics["baseline_acc"][i] = baseline_acc
        #     metrics["nonce_acc"][i] = nonce_acc
        #     print(f"Step {i}, Baseline Acc: {baseline_acc}, Nonce Acc: {nonce_acc}")

    return metrics


def setup_dataset(tokenizer, args):
    dataset = load_dataset("parquet", data_files=args.train_data, split="train")

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
    print("Loaded dataset with {} examples".format(len(train_loader.dataset)))

    total_steps = len(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

    initializer_range = model.config.initializer_range

    metrics = training_loop_temporary_forgetting(
        model,
        tokenizer,
        train_loader,
        optimizer,
        scheduler,
        num_steps=args.N,
        forget_interval=args.k,
        initializer_range=initializer_range,
        vocabulary=None,
    )
    
    model_name = args.model.split("/")[-1]
    output_path = f"{args.output_dir}/{model_name}_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {output_path}")

if __name__ == "__main__":
    main()
