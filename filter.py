import pandas as pd
import numpy as np

def filter(df, is_consistent=True, is_nonsense=False):
    df = df[df["prompt_name"] == "Human-like"]

    consistent = df[df["is_consistent"] == is_consistent]
    consistent = consistent[consistent["is_nonsense"] == is_nonsense]

    return consistent

def main():
    data_path = "data/nli_results.csv"

    df = pd.read_csv(data_path)
    consistent_df = filter(df, is_consistent=True, is_nonsense=False)
    inconsistent_df = filter(df, is_consistent=False, is_nonsense=False)

    consistent_df.to_csv("data/consistent_results.csv")
    inconsistent_df.to_csv("data/inconsistent_results.csv")

if __name__ == "__main__":
    main()

