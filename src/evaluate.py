"""Итоговая оценка трёх моделей: метрики, матрицы ошибок, примеры."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

from data import ROOT, get_loaders
from model import create_model

EXPERIMENTS = ("pretrained-adam", "pretrained-adabound", "scratch-adam")
RESULTS = ROOT / "results"


def save_examples(examples, classes):
    correct = [item for item in examples if item[1] == item[2]][:6]
    wrong = [item for item in examples if item[1] != item[2]][:6]
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (path, truth, pred) in zip(axes.flat, correct + wrong):
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        true_name = classes[truth].split("-", 1)[-1]
        pred_name = classes[pred].split("-", 1)[-1]
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=9,
                     color="green" if truth == pred else "red")
    fig.tight_layout()
    fig.savefig(RESULTS / "examples.png", dpi=200)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, loader = get_loaders(32)
    dataset = loader.dataset
    classes = dataset.dataset.classes
    RESULTS.mkdir(exist_ok=True)
    metrics = {}

    for name in EXPERIMENTS:
        checkpoint = ROOT / "checkpoints" / f"{name}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Нет весов: {checkpoint}")

        model = create_model(pretrained=False).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device,
                                         weights_only=True))
        model.eval()
        true, pred, examples = [], [], []
        position = 0
        print("Тест:", name)

        with torch.inference_mode():
            for images, labels in tqdm(loader, leave=False):
                outputs = model(images.to(device))
                predictions = outputs.argmax(1).cpu().tolist()
                actual = labels.tolist()
                true.extend(actual)
                pred.extend(predictions)
                if name == "pretrained-adam":
                    for truth, guess in zip(actual, predictions):
                        idx = dataset.indices[position]
                        path = dataset.dataset.samples[idx][0]
                        examples.append((path, truth, guess))
                        position += 1

        metrics[name] = {
            "accuracy": accuracy_score(true, pred),
            "precision_macro": precision_score(true, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(true, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(true, pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(true, pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(true, pred, average="weighted", zero_division=0),
        }

        matrix = confusion_matrix(true, pred, labels=range(len(classes)))
        with open(RESULTS / f"{name}_confusion.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["True / Predicted", *classes])
            for breed, row in zip(classes, matrix):
                writer.writerow([breed, *row])

        fig, ax = plt.subplots(figsize=(10, 9))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax)
        ax.set(title=f"Confusion matrix: {name}", xlabel="Predicted class ID",
               ylabel="True class ID")
        fig.tight_layout()
        fig.savefig(RESULTS / f"{name}_confusion.png", dpi=180)
        plt.close(fig)

        if name == "pretrained-adam":
            save_examples(examples, classes)

        result = metrics[name]
        print(f"Accuracy {result['accuracy']:.2%} | Precision macro "
              f"{result['precision_macro']:.2%} | Recall macro {result['recall_macro']:.2%} "
              f"| F1 macro {result['f1_macro']:.2%}")
        del model

    with open(RESULTS / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print("Тестирование завершено, результаты в results/")


if __name__ == "__main__":
    main()
