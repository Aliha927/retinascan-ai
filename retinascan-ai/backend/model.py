import os
import torch
import torch.nn as nn
from torchvision import models

MODEL_CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "checkpoints/dr_model.pth")
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_architecture(num_classes: int = 5) -> nn.Module:
    """Builds a ResNet-50 for offline training."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_dr_model():
    """Loads model without downloading weights dynamically on production startup."""
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "dr_model.pth")
    
    # Instantiate architecture locally (No 100MB download on Render startup)
    model = models.resnet50(weights=None)
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, len(CLASS_NAMES)),
    )
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("[INFO] Successfully loaded fine-tuned DR model checkpoint.")
    else:
        print("[WARNING] No checkpoint found. Initialized model with default structure.")
    
    model.eval()
    model.to(DEVICE)
    return model


def predict_dr(model: nn.Module, input_tensor: torch.Tensor):
    """Runs inference with memory guards."""
    input_tensor = input_tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_percent = confidence.item() * 100
    return predicted_class, confidence_percent