import json
import sys
from pathlib import Path

import evaluate
import numpy as np
from transformers.data.metrics.squad_metrics import compute_f1


def load_data(input_path: Path) -> (list, list):
    labels_path = input_path / "labels.txt"
    predictions_path = input_path / "predictions.txt"

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = f.readlines()
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = f.readlines()
    return labels, predictions


def calculate_scores(predictions: list[str], labels: list[str]):
    bleu = evaluate.combine(["sacrebleu", "rouge", "exact_match"])
    scores = bleu.compute(predictions=predictions, references=labels)

    # Compute F1 score
    f1_scores: list[float] = []
    for label, prediction in zip(labels, predictions):
        f1_scores.append(compute_f1(a_gold=label, a_pred=prediction))
    scores["f1"] = np.mean(f1_scores)
    return scores


def main():
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    else:
        sys.exit("Usage: python calc-bleu.py config.json")

    labels, predictions = load_data(Path(config["output_dir"]))
    print(f"Loaded {len(labels)} labels and {len(predictions)} predictions.")

    score = calculate_scores(predictions, labels)
    print(f"BLEU scores:")
    print(f" BLEU: {score['score']}")
    print(f" Precisions:")
    for i, precision in enumerate(score["precisions"]):
        print(f"  {i}: {precision}")
    print(f"Brevity penalty: {score['bp']}")
    print(f"Length ratio: {score['sys_len'] / score['ref_len']}")
    print(f"Translation length: {score['sys_len']}")
    print(f"Reference length: {score['ref_len']}")

    print(f"ROUGE scores:")
    print(f" ROUGE-1: {score['rouge1']}")
    print(f" ROUGE-2: {score['rouge2']}")
    print(f" ROUGE-L: {score['rougeL']}")
    print(f" ROUGE-Lsum: {score['rougeLsum']}")

    print(f"Exact match: {score['exact_match']}")
    print(f"F1 score: {score['f1']}")

    print("README.md format:")
    # The following lines of code generate the YAML config file for the
    # HuggingFace Hub.
    # The config.json file is used to store the evaluation metrics.
    print("  results:")
    print("  - task:")
    print("      type: text2text-generation")
    print("      name: Question Generation")
    print("    dataset:")
    print("      type: deepset/germanquad")
    print("      name: GermanQuAD")
    print("    metrics:")
    print("      - type: sacrebleu")
    print(f"        value: {score['score']:0.4f}")
    print("        name: SacreBLEU")
    print("        verified: false")
    print("      - type: rouge")
    print(f"        value: {score['rouge1']:0.4f}")
    print("        name: ROUGE-1")
    print("        verified: false")
    print("      - type: rouge")
    print(f"        value: {score['rouge2']:0.4f}")
    print("        name: ROUGE-2")
    print("        verified: false")
    print("      - type: rouge")
    print(f"        value: {score['rougeL']:0.4f}")
    print("        name: ROUGE-L")
    print("        verified: false")
    print("      - type: rouge")
    print(f"        value: {score['rougeLsum']:0.4f}")
    print("        name: ROUGE-Lsum")
    print("        verified: false")
    print("      - type: exact_match")
    print(f"        value: {score['exact_match']:0.4f}")
    print("        name: Exact Match")
    print("        verified: false")
    print("      - type: f1")
    print(f"        value: {score['f1']:0.4f}")
    print("        name: F1")
    print("        verified: false")


if __name__ == '__main__':
    main()
