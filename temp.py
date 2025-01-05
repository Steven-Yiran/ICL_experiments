for question in questions:
  prompt = f"Answer with true or false. Q: {question} \n A:\n"
  input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
  outputs = model.generate(**input_ids, max_new_tokens=16)
  print(tokenizer.decode(outputs[0]))


def generate_with_model(model, tokenizer, prompt, question):
    inputs = prompt.format(question)
    input_ids = tokenizer(inputs, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=16)
    # find the first token that is not part of the input
    new_tokens = outputs[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens)
