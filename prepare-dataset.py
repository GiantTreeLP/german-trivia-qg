from operator import itemgetter

import pandas as pd

from datasets import load_dataset


def main():
    dataset = load_dataset("deepset/germanquad")
    for sub_dataset in dataset:
        df = pd.DataFrame.from_dict(dataset[sub_dataset])
        df["label"] = "Kontext: " + df["context"] + "\nAntwort: " + df["answers"].map(itemgetter("text")).map(
            itemgetter(0))
        df["question"].apply(str.strip)
        df.drop(columns=["context", "answers", "id"], inplace=True)
        df = df.filter(["label", "question"])
        df.to_csv(f"./datasets/germanquad-jeopardy-prepared/{sub_dataset}.csv", index=False)


if __name__ == "__main__":
    main()
