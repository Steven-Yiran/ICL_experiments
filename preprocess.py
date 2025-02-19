import json
import os
from pydantic import BaseModel
import pandas as pd
from tqdm import tqdm

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

class SubjectVerb(BaseModel):
    subjects: list[str]
    verb: str

def system_prompt():
    return """
You are a helpful assistant that preprocesses text data. Your task is to identify the subjects, their plurality, and the verb in a sentence.
There are strictly 2 subjects and 1 verb in the sentence."""

def assistant_prompt():
    return """
Text: birds have more legs than octopuses
{
    "subjects": ["birds", "octopuses"],
    "verb": "have",
    "is_plural": true,
    "is_definite": false
}
Text: the moon is bigger than the earth
{
    "subjects": ["the moon", "the earth"],
    "verb": "is",
    "is_plural": false,
    "is_definite": true
}
"""

def user_prompt(text: str):
    return f"""
Task:
    Text: {text}
"""

def run_openai_task(text: str):
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "assistant", "content": assistant_prompt()},
        {"role": "user", "content": user_prompt(text)},
    ]
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
    )

    try:
        result = json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        print(completion.choices[0].message.content)
        raise Exception("Failed to parse JSON")
    
    # check all the keys are present
    if "subjects" not in result or "verb" not in result or "is_plural" not in result or "is_definite" not in result:
        raise Exception("Missing keys in result")
    
    return result["subjects"], result["verb"], result["is_plural"], result["is_definite"]

def extract_comparatives(text):
    # Remove the brackets and quotes
    cleaned = text.strip("[]'")
    # Split into individual terms
    terms = [term.strip().strip("'") for term in cleaned.split(",")]
    
    # Extract just the comparative words
    comparatives = []
    for term in terms:
        # Split on "than" and take the first part
        comparative = term.split(" than ")[0].strip()
        comparatives.append(comparative)
    
    return comparatives


def create_dataset(data, args, dataset_type):
    """
    dataset_type: "consistent", "inconsistent", "nonsense"
    """
    entries = []
    for i, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {dataset_type} data"):
        ori_idx = row["index"]
        text = row["input"]
        # remove the first word 'if' and split the rest by , and only take the first split
        text = text.split(',')[0]
        text = text.split('If')[1].strip()
        try:
            subjects, verb, is_plural, is_definite = run_openai_task(text)
        except Exception as e:
            print(f"Error processing {text}: {e}")
            continue
        is_be_verb = verb in ["is", "are"]
        # extract comparitor_noun phrase from continuations
        # by taking the words before 'than'
        comparatives = extract_comparatives(row["continuations"])

        if len(subjects) != 2 or len(comparatives) != 2:
            print(f"Skipping {text} because it has {len(subjects)} subjects, and {len(comparatives)} comparatives")
            continue

        # swap the order of the comparitives for the generation to make sense
        comparatives = comparatives[::-1]

        premises = []
        main_subject, comparitor_subject = subjects
        premises.extend([
            f"{main_subject} {verb} {comparatives[0]} than {comparitor_subject}.",
            f"{comparitor_subject} {verb} {comparatives[1]} than {main_subject}.",
        ])
        
        # create question sentence
        questions = []
        golds = []
        for i in range(len(comparatives)):
            for j in range(len(subjects)):
                main_subject = subjects[j]
                comparitor_subject = subjects[1 - j]
                
                question = ""
                if is_be_verb:
                    if is_plural:
                        question = f"Are {main_subject} {comparatives[i]} than {comparitor_subject}?"
                    else:
                        question = f"Is {main_subject} {comparatives[i]} than {comparitor_subject}?"
                else:
                    if is_plural:
                        question = f"Do {main_subject} {verb} {comparatives[i]} than {comparitor_subject}?"
                    else:
                        question = f"Does {main_subject} {verb} {comparatives[i]} than {comparitor_subject}?"

                questions.append(question)
                if i == j:
                    golds.append("yes")
                else:
                    golds.append("no")

        # mix every premise with every question
        for premise in premises:
            for question, gold in zip(questions, golds):
                entries.append({
                    "ori_idx": ori_idx,
                    "prompt": premise + " " + question,
                    "gold": gold,
                    "dataset_type": dataset_type,
                })                

    # create a new dataframe with the entries
    return pd.DataFrame(entries)

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
        create_dataset(consistent, args, "consistent"),
        create_dataset(inconsistent, args, "inconsistent"),
        create_dataset(nonsense, args, "nonsense"),
    )


if __name__ == "__main__":
    args = {}
    #consistent, inconsistent, nonsense = preprocess_nli_data(
    #    "./data/nli_results.csv", args
    #)
    consistent, inconsistent, nonsense = preprocess_nli_data(
        "./data/nli_results.csv", args
    )
    
    # merge the datasets
    dataset = pd.concat([consistent, inconsistent, nonsense])
    dataset.to_csv("./data/eval_prompts.csv", index=False)

