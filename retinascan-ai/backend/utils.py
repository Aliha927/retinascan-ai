import io
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

IMAGE_SIZE = 224

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


def _apply_clahe(image_np: np.ndarray) -> np.ndarray:
    """Applies CLAHE on the L channel (LAB color space) to standardize lighting."""
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess_image(image_bytes: bytes):
    """Preprocesses input bytes to (1, 3, 224, 224) tensor and normalized float numpy array."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # --- 512MB RAM GUARD ---
    # Downscale large fundus photos immediately to prevent memory spikes on Render Free Tier
    pil_image.thumbnail((512, 512))
    
    image_np = np.array(pil_image)

    # Square center-crop
    h, w, _ = image_np.shape
    crop_size = min(h, w)
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    image_np = image_np[top: top + crop_size, left: left + crop_size]

    # Resize to exact model input size (224x224)
    image_np = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
    image_np = _apply_clahe(image_np)

    original_image_float = image_np.astype(np.float32) / 255.0
    input_tensor = _transform(image_np).unsqueeze(0)

    return original_image_float, input_tensor


def generate_gradcam_overlay(original_image: np.ndarray, model, input_tensor) -> bytes:
    """Generates Grad-CAM heatmap overlay with a safety fallback."""
    try:
        model.eval()
        target_layers = [model.layer4[-1]]

        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor)[0]

        overlay = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

        success, buffer = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Failed to encode Grad-CAM image")

        return buffer.tobytes()
    except Exception as e:
        print(f"[WARNING] Grad-CAM computation skipped or failed: {e}")
        # Safe fallback: return CLAHE-enhanced fundus image
        img_bgr = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr)
        return buffer.tobytes()