import pandas as pd

from datasets import load_dataset


def main():
    dataset = load_dataset("deepset/germanquad")
    for sub_dataset in dataset:
        df = pd.DataFrame.from_dict(dataset[sub_dataset])
        df.drop(columns=["id", "answers"], inplace=True)
        df["question"].apply(str.strip)
        df = df.groupby("context").aggregate({"question": " <sep> ".join})
        df.reset_index(inplace=True)
        df.to_csv(f"./datasets/germanquad-e2e-prepared/{sub_dataset}.csv", index=False)


if __name__ == "__main__":
    main()
