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


def create_dataset(data, args):
    """Split dataset into labelled valid and invalid premise-hypothesis strings"""

    if args.it:
        suffix = "?"
    else:
        suffix = "?"
    x = []
    y = []
    for i, row in data.iterrows():
        premise = row["input"]
        continuations = extract_answer(row["continuations"])
        valid_answer = continuations[0]
        invalid_answer = continuations[1]

        if args.it:
            if "gemma" in args.model:
                B_TURN = "<start_of_turn>"
                E_TURN = "<end_of_turn>"
                user_name, model_name = "user", "model"

                premise = premise.replace("then", "then is it true that")

                val= f"{B_TURN}{user_name}\n{premise + valid_answer + suffix}{E_TURN}\n{B_TURN}{model_name}\nAnswer:"
                inval = f"{B_TURN}{user_name}\n{premise + invalid_answer + suffix}{E_TURN}\n{B_TURN}{model_name}\nAnswer:"
                x.append((val, inval))
        else:
            premise = premise.replace("then", "then is it true that")
            val = premise + valid_answer + suffix
            inval = premise + invalid_answer + suffix
            x.append((val, inval))

        y.append((1, 0))

    return x, y


def preprocess_nli_data(path, args):
    """Create datasets from nli_results.csv"""
    data = pd.read_csv(path)
    data = data[data["prompt_name"] == "Human-like"]

    consistent = data[data["is_consistent"] == True]
    consistent = consistent[consistent["is_nonsense"] == False]

    inconsistent = data[data["is_consistent"] == False]
    inconsistent = inconsistent[inconsistent["is_nonsense"] == False]

    nonsense = data[data["is_nonsense"] == True]

    return (
        create_dataset(consistent, args),
        create_dataset(inconsistent, args),
        create_dataset(nonsense, args),
    )


def get_model_and_tokenizer(model_str, device):
    """Helper function to get model"""
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.bfloat16)
    model.to(device)
    return model, tokenizer


def evaluate(model, tokenizer, x, y, args):
    """Run evaluation on pairs of items, must get both valid and invalid correct to count"""
    item_accuracy = []

    valid_tok = tokenizer(args.valid_token)["input_ids"][1]
    invalid_tok = tokenizer(args.invalid_token)["input_ids"][1]

    for i in range(len(x)):

        pair_correct = 0
        for j in range(2):
            ipt = tokenizer.encode_plus(x[i][j], return_tensors="pt").to(args.device)
            logits = model(ipt["input_ids"]).logits

            if y[i][j] == 1:
                correct_idx = valid_tok
                incorrect_idx = invalid_tok
            else:
                correct_idx = invalid_tok
                incorrect_idx = valid_tok

            correct = logits[0, -1, correct_idx] > logits[0, -1, incorrect_idx]
            pair_correct += correct.cpu()
        
        item_accuracy.append(pair_correct == 2)

    return item_accuracy


def format_interp_dataset(pairs, accs, label, args):

    dataset_dict = {
        "valid_stimuli": [],
        "invalid_stimuli": [],
        "valid_word": [],
        "invalid_word": [],
        "correct": []
    }

    for i in range(len(pairs)):
        valid = pairs[i][0]
        invalid = pairs[i][1]
        valid_word = args.valid_token
        invalid_word = args.invalid_token
        correct = accs[i].int().item()

        dataset_dict["valid_stimuli"].append(valid)
        dataset_dict["invalid_stimuli"].append(invalid)
        dataset_dict["valid_word"].append(valid_word)
        dataset_dict["invalid_word"].append(invalid_word)
        dataset_dict["correct"].append(correct)

    pd.DataFrame.from_dict(dataset_dict).to_csv(
        f"./data/{'_'.join(args.model.split('/'))}_{label}.csv"
    )

if __name__ == "__main__":

    args = parse_args()
    consistent, inconsistent, nonsense = preprocess_nli_data(
        "./data/nli_results.csv", args
    )
    model, tokenizer = get_model_and_tokenizer(args.model, args.device)

    consistent_accs = evaluate(model, tokenizer, consistent[0], consistent[1], args)
    inconsistent_accs = evaluate(
        model, tokenizer, inconsistent[0], inconsistent[1], args
    )
    nonsense_accs = evaluate(model, tokenizer, nonsense[0], nonsense[1], args)

    results = {
        "model": [args.model],
        "consistent": [np.mean(consistent_accs)],
        "inconsistent": [np.mean(inconsistent_accs)],
        "nonsense": [np.mean(nonsense_accs)],
    }

    format_interp_dataset(consistent[0], consistent_accs, "consistent", args)
    format_interp_dataset(inconsistent[0], inconsistent_accs, "inconsistent", args)
    format_interp_dataset(nonsense[0], nonsense_accs, "nonsense", args)

    pd.DataFrame.from_dict(results).to_csv(
        f"./results/{'_'.join(args.model.split('/'))}_behavior.csv"
    )
