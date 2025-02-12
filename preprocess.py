import json
import os
from pydantic import BaseModel
import pandas as pd

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
    "plurality": [true, true],
    "verb": "have"
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
    
    return result["subjects"], result["plurality"], result["verb"]

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


def create_dataset(data, args):
    entries = []
    for i, row in data.iterrows():
        ori_idx = row["index"]
        text = row["input"]
        # remove the first word 'if' and split the rest by , and only take the first split
        text = text.split(',')[0]
        text = text.split('If')[1].strip()
        subjects, plurality, verb = run_openai_task(text)
        is_be_verb = verb in ["is", "are"]
    
        assert len(subjects) == 2
        assert len(plurality) == 2
        
        # extract comparitor_noun phrase from continuations
        # by taking the words before 'than'
        comparatives = extract_comparatives(row["continuations"])
        
        assert len(comparatives) == 2
        
        # create question sentence
        questions = []
        for i in range(2):
            for j in range(2):
                main_subject = subjects[i]
                comparitor_subject = subjects[1 - i]
                is_plural = plurality[i]

                question = ""
                if is_be_verb:
                    if is_plural:
                        question = f"Are {main_subject} {comparatives[j]} than {comparitor_subject}?"
                    else:
                        question = f"Is {main_subject} {comparatives[j]} than {comparitor_subject}?"
                else:
                    if is_plural:
                        question = f"Do {main_subject} {verb} {comparatives[j]} than {comparitor_subject}?"
                    else:
                        question = f"Does {main_subject} {verb} {comparatives[j]} than {comparitor_subject}?"

                questions.append(question)

        premises = []
        for i in range(2):
            for j in range(2):
                main_subject = subjects[i]
                comparitor_subject = subjects[1 - i]
                is_plural = plurality[i]

                premise = ""
                if is_be_verb and is_plural:
                    premise = f"{main_subject} {verb} {comparatives[j]} than {comparitor_subject}."
                elif is_be_verb and not is_plural:
                    premise = f"{main_subject} {verb} {comparatives[j]} than {comparitor_subject}."
                elif not is_be_verb and is_plural:
                    premise = f"{main_subject} {verb} {comparatives[j]} than {comparitor_subject}."
                else:
                    premise = f"{main_subject} {verb} {comparatives[j]} than {comparitor_subject}."

                premises.append(premise)

        # mix every premise with every question
        for premise in premises:
            for question in questions:
                entries.append({
                    "ori_idx": ori_idx,
                    "prompt": premise + " " + question,
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

    return create_dataset(consistent, args)
    # return (
    #     create_dataset(consistent, args),
    #     #create_dataset(inconsistent, args),
    #     #create_dataset(nonsense, args),
    # )


if __name__ == "__main__":
    args = {}
    #consistent, inconsistent, nonsense = preprocess_nli_data(
    #    "./data/nli_results.csv", args
    #)
    consistent = preprocess_nli_data(
        "./data/nli_results.csv", args
    )
    consistent.to_csv("./data/consistent_results.csv", index=False)

