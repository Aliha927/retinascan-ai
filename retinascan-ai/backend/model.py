"""
model.py
--------
Loads the DR classification model and runs inference.

Architecture: ResNet-50 backbone (ImageNet pretrained) with a custom
5-class classification head, fine-tuned for Diabetic Retinopathy severity
grading (Stage 2 of the pipeline).

IMPORTANT: This file expects a fine-tuned checkpoint at MODEL_CHECKPOINT_PATH.
Run train.py first to produce this checkpoint from your dataset (e.g. APTOS 2019).
If no checkpoint is found, it falls back to an ImageNet-pretrained backbone with
an UNTRAINED classification head — this will run without crashing (good for
testing the pipeline plumbing) but predictions will be meaningless until you
train it. Do not present untrained-head output as a real result.
"""

import os
import torch
import torch.nn as nn
from torchvision import models

MODEL_CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "checkpoints/dr_model.pth")

# Class order MUST match the order used during training (see train.py)
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_architecture(num_classes: int = 5) -> nn.Module:
    """Builds a ResNet-50 with the final FC layer replaced for DR grading."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_dr_model() -> nn.Module:
    """
    Loads the model. If a fine-tuned checkpoint exists, loads its weights.
    Otherwise returns the ImageNet-pretrained backbone with a fresh (untrained)
    head, and prints a loud warning so this is never silently mistaken for a
    working clinical model.
    """
    model = build_architecture(num_classes=len(CLASS_NAMES))

    if os.path.exists(MODEL_CHECKPOINT_PATH):
        state_dict = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"[model.py] Loaded fine-tuned checkpoint: {MODEL_CHECKPOINT_PATH}")
    else:
        print(
            "[model.py] WARNING: No fine-tuned checkpoint found at "
            f"'{MODEL_CHECKPOINT_PATH}'. Using ImageNet backbone with an "
            "UNTRAINED classification head. Run train.py before demoing — "
            "predictions right now are not clinically meaningful."
        )

    model.to(DEVICE)
    model.eval()
    return model


def predict_dr(model: nn.Module, input_tensor: torch.Tensor):
    """
    Runs inference on a preprocessed image tensor.
    Returns (predicted_class_name: str, confidence_percent: float)
    """
    input_tensor = input_tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_percent = confidence.item() * 100
    return predicted_class, confidence_percent
