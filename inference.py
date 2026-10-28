"""
Dual Inference Engine for Banana Analysis:
  1. Disease Diagnosis: DenseNet-121 (Version 24 from Kaggle, 384x384) with Grad-CAM
  2. Ripeness Classification: EfficientNet-B0 (Version 18 from Kaggle, 224x224)
"""

import os
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

from classes import (
    BANANA_CLASSES,
    BANANA_VN_MAP,
    DISEASE_INFO,
    RIPENESS_CLASSES,
    RIPENESS_INFO,
    get_pathogen_info
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DISEASE_MODEL_PATH = MODEL_DIR / "best_banana_densenet121.pth"
RIPENESS_MODEL_PATH = MODEL_DIR / "best_banana_ripeness_efficientnet_b0.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Caches
_DISEASE_MODEL: Optional[nn.Module] = None
_RIPENESS_MODEL: Optional[nn.Module] = None

# Transforms
DISEASE_SIZE = (384, 384)
RIPENESS_SIZE = (224, 224)

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_TRANSFORM_DISEASE = transforms.Compose([
    transforms.Resize(DISEASE_SIZE),
    transforms.ToTensor(),
    NORMALIZE
])

_TRANSFORM_RIPENESS = transforms.Compose([
    transforms.Resize(RIPENESS_SIZE),
    transforms.ToTensor(),
    NORMALIZE
])


def get_disease_model() -> Optional[nn.Module]:
    """Load and cache DenseNet-121 for banana disease classification (Version 24)."""
    global _DISEASE_MODEL
    if _DISEASE_MODEL is not None:
        return _DISEASE_MODEL

    if not DISEASE_MODEL_PATH.exists():
        print(f"[Inference] Warning: Disease model not found at {DISEASE_MODEL_PATH}")
        return None

    try:
        print(f"[Inference] Loading DenseNet-121 (Disease - Version 24) on {DEVICE}...")
        model = models.densenet121(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, len(BANANA_CLASSES))
        )
        state_dict = torch.load(str(DISEASE_MODEL_PATH), map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        _DISEASE_MODEL = model
        print("[Inference] DenseNet-121 loaded successfully.")
        return _DISEASE_MODEL
    except Exception as e:
        print(f"[Inference] Error loading disease model: {e}")
        return None


def get_ripeness_model() -> Optional[nn.Module]:
    """Load and cache EfficientNet-B0 for banana ripeness classification (Version 18)."""
    global _RIPENESS_MODEL
    if _RIPENESS_MODEL is not None:
        return _RIPENESS_MODEL

    if not RIPENESS_MODEL_PATH.exists():
        print(f"[Inference] Warning: Ripeness model not found at {RIPENESS_MODEL_PATH}")
        return None

    try:
        print(f"[Inference] Loading EfficientNet-B0 (Ripeness - Version 18) on {DEVICE}...")
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, len(RIPENESS_CLASSES))
        )
        state_dict = torch.load(str(RIPENESS_MODEL_PATH), map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        _RIPENESS_MODEL = model
        print("[Inference] EfficientNet-B0 Ripeness model loaded successfully.")
        return _RIPENESS_MODEL
    except Exception as e:
        print(f"[Inference] Error loading ripeness model: {e}")
        return None


def load_pil_image(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(str(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
        else:
            return Image.fromarray(image)
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def generate_gradcam(model: nn.Module, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    """Generate Grad-CAM heatmap for DenseNet-121 target layer."""
    activations = None
    gradients = None

    target_layer = model.features.denseblock4

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()
        output = model(input_tensor)
        score = output[0, class_idx]
        score.backward(retain_graph=False)

        if activations is None or gradients is None:
            return np.zeros(DISEASE_SIZE, dtype=np.float32)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1)).squeeze(0)
        cam = cam.detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam
    except Exception as e:
        print(f"[Grad-CAM] Error: {e}")
        return np.zeros(DISEASE_SIZE, dtype=np.float32)
    finally:
        h1.remove()
        h2.remove()


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay heatmap on RGB image."""
    h, w, _ = original_img.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(original_img, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)
    return blended


def predict_banana_dual(
    image: Union[str, Path, Image.Image, np.ndarray],
    conf_threshold: float = 0.15,
    generate_cam: bool = True
) -> Dict[str, Any]:
    """
    Simultaneously predict disease and ripeness for a banana fruit image.
    """
    pil_img = load_pil_image(image)
    resized_orig = pil_img.resize(DISEASE_SIZE)
    orig_rgb = np.array(resized_orig, dtype=np.uint8)

    # 1. Run Disease Diagnosis (DenseNet-121)
    disease_model = get_disease_model()
    if disease_model is None:
        best_disease = {"class": "Unknown", "vn_name": "Chưa tải mô hình bệnh", "confidence": 0.0, "percentage": "0.0%"}
        gradcam_img = orig_rgb
        disease_info = {}
        pathogen_info = {}
    else:
        disease_tensor = _TRANSFORM_DISEASE(pil_img).unsqueeze(0).to(DEVICE)
        disease_tensor.requires_grad = True

        out_disease = disease_model(disease_tensor)
        probs_disease = F.softmax(out_disease, dim=1)[0].detach().cpu().numpy()

        sorted_d_idx = np.argsort(probs_disease)[::-1]
        best_d_idx = int(sorted_d_idx[0])
        best_d_class = BANANA_CLASSES[best_d_idx]
        best_d_score = float(probs_disease[best_d_idx])

        best_disease = {
            "class": best_d_class,
            "vn_name": BANANA_VN_MAP.get(best_d_class, best_d_class),
            "confidence": best_d_score,
            "percentage": f"{best_d_score * 100:.2f}%"
        }

        gradcam_img = orig_rgb
        if generate_cam:
            try:
                cam_heatmap = generate_gradcam(disease_model, disease_tensor, best_d_idx)
                gradcam_img = overlay_heatmap(orig_rgb, cam_heatmap, alpha=0.45)
            except Exception as e:
                print(f"[Inference] Heatmap overlay error: {e}")

        disease_info = DISEASE_INFO.get(best_d_class, {})
        pathogen_info = get_pathogen_info(best_d_class)

    # 2. Run Ripeness Classification (EfficientNet-B0)
    ripeness_model = get_ripeness_model()
    if ripeness_model is None:
        best_ripeness = {"class": "Unknown", "vn_name": "Chưa tải mô hình độ chín", "confidence": 0.0, "percentage": "0.0%"}
        ripeness_meta = {}
    else:
        ripeness_tensor = _TRANSFORM_RIPENESS(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out_ripeness = ripeness_model(ripeness_tensor)
            probs_ripeness = F.softmax(out_ripeness, dim=1)[0].cpu().numpy()

        sorted_r_idx = np.argsort(probs_ripeness)[::-1]
        best_r_idx = int(sorted_r_idx[0])
        best_r_class = RIPENESS_CLASSES[best_r_idx]
        best_r_score = float(probs_ripeness[best_r_idx])

        r_meta = RIPENESS_INFO.get(best_r_class, {})
        best_ripeness = {
            "class": best_r_class,
            "vn_name": r_meta.get("vn_name", best_r_class),
            "icon": r_meta.get("icon", "🍌"),
            "confidence": best_r_score,
            "percentage": f"{best_r_score * 100:.2f}%"
        }
        ripeness_meta = r_meta

    return {
        "disease": best_disease,
        "disease_info": disease_info,
        "pathogen_info": pathogen_info,
        "gradcam_image": gradcam_img,
        "ripeness": best_ripeness,
        "ripeness_meta": ripeness_meta
    }
