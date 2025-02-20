from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# Get the token ID 3553
token_id = 1718

# Decode the token ID to see what it represents
token = tokenizer.decode([token_id])
print(f"Token ID {token_id} represents: {token}")

# You can also see the token in the vocabulary
vocab = tokenizer.get_vocab()
for word, id in vocab.items():
    if id == token_id:
        print(f"Found in vocabulary: {word}")
