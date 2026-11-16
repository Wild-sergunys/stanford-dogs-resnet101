"""Обучение трёх конфигураций ResNet-101 без заморозки слоёв."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from pytorch_optimizer import AdaBound
from tqdm import tqdm

from data import ROOT, get_loaders
from model import create_model

# Два сопоставимых эксперимента с предобучением; scratch учим дольше.
EXPERIMENTS = {
    "pretrained-adam": {"epochs": 30, "lr": 1e-4, "patience": 10, "min_epochs": 12},
    "pretrained-adabound": {"epochs": 30, "lr": 1e-4, "patience": 10, "min_epochs": 12},
    "scratch-adam": {"epochs": 150, "lr": 3e-4, "patience": 15, "min_epochs": 30},
}
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        loss_sum += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += len(labels)

    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    args = parser.parse_args()

    name = args.experiment
    config = EXPERIMENTS[name]
    checkpoint = ROOT / "checkpoints" / f"{name}.pt"
    report = ROOT / "results" / f"{name}.json"
    if checkpoint.exists() or report.exists():
        parser.error("Результаты уже существуют. Не удаляй их случайно.")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_loaders(BATCH_SIZE)
    model = create_model(pretrained=name != "scratch-adam").to(device)
    criterion = nn.CrossEntropyLoss()

    if name == "pretrained-adabound":
        optimizer = AdaBound(model.parameters(), lr=config["lr"],
                             final_lr=0.1, weight_decay=WEIGHT_DECAY,
                             weight_decouple=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                     weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_accuracy = -1.0
    best_epoch = 0
    bad_epochs = 0
    print(f"ResNet-101 | {name} | {device} | batch={BATCH_SIZE} | "
          f"max_epochs={config['epochs']}")

    for epoch in range(1, config["epochs"] + 1):
        start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "learning_rate": current_lr, "seconds": time.perf_counter() - start,
        })

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1

        print(f"{epoch:3d}/{config['epochs']} | "
              f"train: {train_acc:.2%}, loss={train_loss:.3f} | "
              f"val: {val_acc:.2%}, loss={val_loss:.3f} | "
              f"best: {best_accuracy:.2%} ({best_epoch})")

        with open(report, "w", encoding="utf-8") as file:
            json.dump({
                "experiment": name,
                "pretrained_weights": "IMAGENET1K_V1" if name != "scratch-adam" else None,
                "optimizer": "AdaBound" if name == "pretrained-adabound" else "Adam",
                "batch_size": BATCH_SIZE, "learning_rate": config["lr"],
                "weight_decay": WEIGHT_DECAY, "scheduler": "CosineAnnealingLR",
                "epochs_max": config["epochs"], "patience": config["patience"],
                "min_epochs": config["min_epochs"], "best_epoch": best_epoch,
                "best_val_accuracy": best_accuracy,
                "device": str(device), "loss": "CrossEntropyLoss",
                "augmentation": "RandomCrop, HorizontalFlip, RandomAffine",
                "history": history,
            }, file, indent=2, ensure_ascii=False)

        if epoch >= config["min_epochs"] and bad_epochs >= config["patience"]:
            print("Early stopping: улучшений нет", bad_epochs, "эпох")
            break

    print(f"Готово. Лучшая эпоха: {best_epoch}, Val Accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
