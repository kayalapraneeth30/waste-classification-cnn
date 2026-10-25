import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from src.config import IMG_WIDTH, IMG_HEIGHT

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def load_and_preprocess_image(image_input, target_size: Tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT)) -> np.ndarray:
    """
    Loads an image from filepath or byte buffer, converts to RGB,
    resizes to target dimensions, and normalizes pixel values to [0.0, 1.0].
    """
    img_rgb = None

    if isinstance(image_input, (str, Path)):
        file_path = str(image_input)
        if CV2_AVAILABLE:
            bgr_img = cv2.imread(file_path)
            if bgr_img is not None:
                img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        if img_rgb is None and PIL_AVAILABLE:
            pil_img = Image.open(file_path).convert('RGB')
            img_rgb = np.array(pil_img)
    elif isinstance(image_input, bytes):
        if CV2_AVAILABLE:
            nparr = np.frombuffer(image_input, np.uint8)
            bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr_img is not None:
                img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        if img_rgb is None and PIL_AVAILABLE:
            import io
            pil_img = Image.open(io.BytesIO(image_input)).convert('RGB')
            img_rgb = np.array(pil_img)
    elif isinstance(image_input, np.ndarray):
        img_rgb = image_input

    if img_rgb is None:
        raise ValueError("Failed to decode image from provided input.")

    # Resize to target size (128, 128)
    if CV2_AVAILABLE:
        resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    elif PIL_AVAILABLE:
        pil_img = Image.fromarray(img_rgb).resize(target_size, Image.Resampling.BILINEAR)
        resized = np.array(pil_img)
    else:
        # Simple nearest-neighbor resize with pure NumPy
        h_orig, w_orig = img_rgb.shape[:2]
        h_target, w_target = target_size
        row_indices = np.linspace(0, h_orig - 1, h_target, dtype=int)
        col_indices = np.linspace(0, w_orig - 1, w_target, dtype=int)
        resized = img_rgb[row_indices[:, None], col_indices]

    # Normalize to [0.0, 1.0]
    normalized = resized.astype(np.float32) / 255.0
    return normalized

def augment_image(img: np.ndarray) -> np.ndarray:
    """
    Data Augmentation:
    Applies random horizontal flip, slight rotation, and brightness jitter.
    """
    aug = img.copy()

    # 1. Random Horizontal Flip (50% probability)
    if np.random.rand() > 0.5:
        aug = np.fliplr(aug)

    # 2. Random Brightness Scaling (factor between 0.85 and 1.15)
    brightness_factor = np.random.uniform(0.85, 1.15)
    aug = np.clip(aug * brightness_factor, 0.0, 1.0)

    # 3. Random Slight Zoom / Crop
    if np.random.rand() > 0.5:
        crop_factor = np.random.uniform(0.90, 0.98)
        h, w = aug.shape[:2]
        ch, cw = int(h * crop_factor), int(w * crop_factor)
        start_y = np.random.randint(0, h - ch)
        start_x = np.random.randint(0, w - cw)
        cropped = aug[start_y:start_y+ch, start_x:start_x+cw]
        
        # Resize back to original
        if CV2_AVAILABLE:
            aug = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        elif PIL_AVAILABLE:
            pil_img = Image.fromarray((cropped * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
            aug = np.array(pil_img).astype(np.float32) / 255.0

    return aug
