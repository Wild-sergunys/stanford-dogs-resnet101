"""Графики обучения всех трёх экспериментов."""
import json

import matplotlib.pyplot as plt

from data import ROOT

NAMES = {
    "pretrained-adam": "Pretrained + Adam",
    "pretrained-adabound": "Pretrained + AdaBound",
    "scratch-adam": "Scratch + Adam",
}


def plot(metric):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, label in NAMES.items():
        with open(ROOT / "results" / f"{name}.json", encoding="utf-8") as file:
            history = json.load(file)["history"]
        epochs = [item["epoch"] for item in history]
        factor = 100 if metric == "acc" else 1
        train = [item[f"train_{metric}"] * factor for item in history]
        val = [item[f"val_{metric}"] * factor for item in history]
        ax.plot(epochs, train, linestyle="--", label=f"{label} / train")
        ax.plot(epochs, val, label=f"{label} / val")
    ax.set(xlabel="Epoch", ylabel="Accuracy (%)" if metric == "acc" else "Loss",
           title="Training and validation " + ("accuracy" if metric == "acc" else "loss"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"{'accuracy' if metric == 'acc' else 'loss'}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot("acc")
    plot("loss")
    print("Графики сохранены в results/")
