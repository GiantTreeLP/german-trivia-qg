import sys
from pathlib import Path
from typing import Optional

import evaluate


def load_data(input_path: Path) -> (list, list):
    labels_path = input_path / "labels.txt"
    predictions_path = input_path / "predictions.txt"

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = f.readlines()
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = f.readlines()

    return labels, predictions


def calculate_bleu_score(predictions: list[str], labels: list[str]):
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=labels)


def main():
    path: Optional[Path] = None
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} [path]")
        sys.exit(1)
    if len(sys.argv) == 2:
        arg_path = sys.argv[1]
        path = Path(arg_path)
        if not path.exists():
            print(f"Path {arg_path} does not exist!")
            sys.exit(1)
    if path is None:
        print("Something went wrong!")
        sys.exit(1)

    labels, predictions = load_data(path)
    print(f"Loaded {len(labels)} labels and {len(predictions)} predictions.")

    score = calculate_bleu_score(predictions, labels)
    print(f"BLEU scores:")
    print(f" BLEU: {score['bleu']}")
    print(f" Precisions:")
    for i, precision in enumerate(score["precisions"]):
        print(f"  {i}: {precision}")
    print(f"Brevity penalty: {score['brevity_penalty']}")
    print(f"Length ratio: {score['length_ratio']}")
    print(f"Translation length: {score['translation_length']}")
    print(f"Reference length: {score['reference_length']}")

    print("README.md format:")
    # The following lines of code generate the YAML config file for the
    # HuggingFace Hub.
    # The config.json file is used to store the evaluation metrics.
    # The metrics are stored in a dictionary with the metric name as key.
    print("results:")
    print("- task:")
    print("    type: text2text-generation")
    print("    name: Question Generation")
    print("  dataset:")
    print("    type: deepset/germanquad")
    print("    name: GermanQuAD")
    print("  metrics:")
    print("    - type: bleu")
    print(f"      value: {score['bleu']:0.4f}")
    print("      name: BLEU")
    print("      verified: false")


if __name__ == '__main__':
    main()
