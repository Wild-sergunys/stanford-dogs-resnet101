"""Подготовка Stanford Dogs и загрузчики данных."""
import json
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/stanford-dogs/images/Images"
SPLITS = ROOT / "data/splits.json"


def prepare_data():
    dataset = datasets.ImageFolder(IMAGES)
    if SPLITS.exists():
        with open(SPLITS, encoding="utf-8") as file:
            parts = json.load(file)
        print("Используем сохранённое разбиение:",
              {k: len(v) for k, v in parts.items()})
        return

    # Проверяем изображения и делим одни и те же файлы между тремя выборками.
    for path, _ in dataset.samples:
        with Image.open(path) as image:
            image.verify()

    indices = list(range(len(dataset)))
    train, remaining = train_test_split(
        indices, test_size=0.30, stratify=dataset.targets, random_state=42
    )
    val, test = train_test_split(
        remaining, test_size=0.50,
        stratify=[dataset.targets[i] for i in remaining], random_state=42
    )
    assert len(set(train + val + test)) == len(dataset)

    SPLITS.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS, "w", encoding="utf-8") as file:
        json.dump({"train": train, "val": val, "test": test}, file)
    print(f"Пород: {len(dataset.classes)} | Train: {len(train)} | "
          f"Val: {len(val)} | Test: {len(test)}")


def get_loaders(batch_size=32):
    with open(SPLITS, encoding="utf-8") as file:
        parts = json.load(file)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(), normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), normalize,
    ])

    train_data = datasets.ImageFolder(IMAGES, transform=train_transform)
    eval_data = datasets.ImageFolder(IMAGES, transform=test_transform)

    return (
        DataLoader(Subset(train_data, parts["train"]), batch_size=batch_size,
                   shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(Subset(eval_data, parts["val"]), batch_size=batch_size,
                   num_workers=2, pin_memory=True),
        DataLoader(Subset(eval_data, parts["test"]), batch_size=batch_size,
                   num_workers=2, pin_memory=True),
    )


if __name__ == "__main__":
    prepare_data()
