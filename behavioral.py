import random
import numpy as np
import torch

vocabulary = [
    " A",
    " B",
    " C",
    " D",
    " E",
    " F",
    " G",
    " H",
    " I",
    " J",
    " K",
    " L",
    " M",
    " N",
    " O",
    " P",
    " Q",
    " R",
    " S",
    " T",
    " U",
    " V",
    " W",
    " X",
    " Y",
    " Z",
]

def logits_to_probability(tokenizer, logits, correct_answer):
    """Compute the probability of the correct answer"""
    correct_idx = tokenizer.encode(correct_answer)[0]
    
    logits_last = logits[0, -1, :]  # Shape: [vocab_size]
    probs = torch.softmax(logits_last, dim=-1)  # Shape: [vocab_size]
    prob_correct = probs[correct_idx]
    return prob_correct.item()

def logits_to_logit_diff_t(tokenizer, logits, correct_answer, incorrect_answer):
    """Compute logit difference"""
    correct_idx = tokenizer.encode(correct_answer)[0]
    incorrect_idx = tokenizer.encode(incorrect_answer)[0]
    # print(logits[0, -1, correct_idx], logits[0, -1, incorrect_idx])
    # print(logits_to_probability(tokenizer, logits, correct_answer), logits_to_probability(tokenizer, logits, incorrect_answer))
    # print(logits[0, -1].argmax().item() == correct_idx, logits[0, -1].argmax().item() == incorrect_idx)
    # print(logits[0, -1].argmax().item(), correct_idx, incorrect_idx)
    return logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]

def behavioral_analysis_t(
    model,
    tokenizer,
    vocabulary,
    q_type,
    correct_label,
    incorrect_label,
    device,
    data,
    verbose=True
):
    """Main function"""
    accuracy = []
    total_samples = len(data["clean_x"])
    for i in range(total_samples):
        if i % 50 == 0 and verbose:
            print(f"Progress: {round(i/total_samples, 3)}")
        clean_prompt = data[q_type][i]
        clean_label = data[correct_label][i]
        cf_label = data[incorrect_label][i]
        clean_tokens = tokenizer.encode(clean_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(clean_tokens).logits
        # Get the top prediction token
        top_token_id = logits[0, -1].argmax().item()
        top_token = tokenizer.decode([top_token_id])
        
        if verbose:
            print(f"{clean_prompt}")
            print(f"{top_token} {clean_label}")
        accuracy.append(logits_to_logit_diff_t(tokenizer, logits, clean_label, cf_label) > 0)
    if verbose:
        print(f"Model Accuracy: {torch.mean(torch.Tensor(accuracy))}")
    return torch.mean(torch.Tensor(accuracy))

def generate_abstract_example_t(vocab_set, idx_1, idx_2, idx_3, nonce=False):
    """Helper function to generate abstract example"""
    item_1 = random.choice([f' [NEW_TOKEN{i}]' for i in range(50)]) if nonce else vocab_set[idx_1]
    item_2 = vocab_set[idx_2]
    item_3 = vocab_set[idx_3]
    
    prompt = (
        f"All{item_1} are{item_2}. "
        + f"All{item_2} are{item_3}. "
        + f"Therefore, all{item_1} are"
    )
    label = item_3
    wrong_label = item_2
    #print(prompt, label, wrong_label)
    return prompt, label, wrong_label


def setup_behavior_dataset(
    vocabulary,
    data_size):

    data_baseline = {
    "clean_x": [],
    "clean_y": [],
    "wrong_y": [],
    }
    for _ in range(data_size):
        vocab_set = np.random.choice(vocabulary, 8, replace=False)
        vocab_set_1 = vocab_set[:4]
        clean_x, clean_y, wrong_y = generate_abstract_example_t(vocab_set_1, 0, 1, 2, nonce=False)
        data_baseline["clean_x"].append(clean_x)
        data_baseline["clean_y"].append(clean_y)
        data_baseline["wrong_y"].append(wrong_y)

    data_nonce = {
    "clean_x": [],
    "clean_y": [],
    "wrong_y": [],
    }
    for _ in range(data_size):
        vocab_set = np.random.choice(vocabulary, 8, replace=False)
        vocab_set_1 = vocab_set[:4]
        clean_x, clean_y, wrong_y = generate_abstract_example_t(vocab_set_1, 0, 1, 2, nonce=True)
        data_nonce["clean_x"].append(clean_x)
        data_nonce["clean_y"].append(clean_y)
        data_nonce["wrong_y"].append(wrong_y)

    return data_baseline, data_nonce