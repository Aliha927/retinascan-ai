"""
train.py
--------
Fine-tunes the ResNet-50 model on your DR dataset and saves a checkpoint that
model.py will load automatically.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from model import build_architecture, CLASS_NAMES, DEVICE, MODEL_CHECKPOINT_PATH

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 0):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

    # Sanity check: class folder order must match CLASS_NAMES in model.py
    assert set(train_dataset.classes) == set(CLASS_NAMES), (
        f"Class folder names {train_dataset.classes} don't match expected "
        f"{CLASS_NAMES}. Rename your folders to match exactly."
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_dataset.classes


def train(data_dir: str, epochs: int, batch_size: int, lr: float, num_workers: int = 0):
    train_loader, val_loader, class_order = get_dataloaders(data_dir, batch_size, num_workers)

    model = build_architecture(num_classes=len(class_order)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_loss = running_loss / len(train_loader.dataset)
        val_acc, val_report, val_cm = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    # Save final checkpoint
    os.makedirs(os.path.dirname(MODEL_CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
    print(f"\nSaved checkpoint to {MODEL_CHECKPOINT_PATH}")

    print("\nFinal Validation Report:")
    print(val_report)
    print("Confusion Matrix:")
    print(val_cm)


def evaluate(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, report, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset/ folder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0, help="0 for CPU safe compatibility")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.num_workers)