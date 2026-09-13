"""
utils.py
--------
Preprocessing and Explainable AI (Grad-CAM) utilities.

Stage 1 (Data Ingestion & Preprocessing): CLAHE contrast enhancement + resize
  + tensor conversion, matching what the model was trained on.
Stage 3 (XAI Module): Grad-CAM heatmap generation showing which retinal
  regions drove the model's classification.
"""

import io
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

IMAGE_SIZE = 224  # standard ResNet-50 input size

# Must match the normalization used during training (ImageNet stats, since
# the backbone is ImageNet-pretrained)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


def _apply_clahe(image_np: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization on the L channel
    (LAB color space) to correct uneven illumination common in fundus photos,
    without distorting color-based lesion signals (hemorrhages, exudates).
    """
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def preprocess_image(image_bytes: bytes):
    """
    Takes raw uploaded image bytes and returns:
      - original_image: np.ndarray (RGB, float 0-1, resized) for Grad-CAM overlay
      - input_tensor: torch.Tensor, shape (1, 3, 224, 224), normalized, ready for the model
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(pil_image)

    # Center crop to square before resize, to avoid squashing the fundus circle
    h, w, _ = image_np.shape
    crop_size = min(h, w)
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    image_np = image_np[top: top + crop_size, left: left + crop_size]

    image_np = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
    image_np = _apply_clahe(image_np)

    # Keep a float [0,1] copy for Grad-CAM overlay rendering
    original_image_float = image_np.astype(np.float32) / 255.0

    input_tensor = _transform(image_np).unsqueeze(0)  # add batch dimension

    return original_image_float, input_tensor


def generate_gradcam_overlay(original_image: np.ndarray, model, input_tensor) -> bytes:
    """
    Generates a Grad-CAM heatmap overlaid on the original fundus image,
    highlighting the regions that most influenced the model's prediction.
    Returns JPEG-encoded bytes ready to be base64-sent to the frontend.
    """
    # Target the last convolutional block of ResNet-50 for Grad-CAM
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # (H, W)

    overlay = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    success, buffer = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode Grad-CAM overlay image")

    return buffer.tobytes()
