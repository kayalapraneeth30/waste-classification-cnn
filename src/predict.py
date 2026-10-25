from pathlib import Path
from typing import Dict, Any, Union
import numpy as np

from src.config import CLASSES, RECYCLING_GUIDELINES, MODEL_PATH
from src.cnn_model import FallbackInferenceEngine, TF_AVAILABLE
from src.data_preprocessing import load_and_preprocess_image

# Global inference engine instance
_engine = None

def get_inference_engine():
    global _engine
    if _engine is None:
        if TF_AVAILABLE and MODEL_PATH.exists():
            import tensorflow as tf
            try:
                _engine = tf.keras.models.load_model(str(MODEL_PATH))
                print(f"[Predict] Loaded trained CNN model from {MODEL_PATH}")
            except Exception as e:
                print(f"[Predict] Error loading H5 model ({e}), using FallbackInferenceEngine.")
                _engine = FallbackInferenceEngine(classes=CLASSES)
        else:
            _engine = FallbackInferenceEngine(classes=CLASSES)
    return _engine

def classify_waste_image(image_input: Union[str, Path, bytes, np.ndarray]) -> Dict[str, Any]:
    """
    End-to-end prediction pipeline:
    1. Preprocesses image (RGB conversion, resize to 128x128, normalization to [0, 1]).
    2. Runs forward pass through CNN model.
    3. Formulates probability distribution and top-1 class prediction.
    4. Attaches designated disposal bin and eco-friendly recycling guidelines.
    """
    # 1. Preprocess
    preprocessed_img = load_and_preprocess_image(image_input)
    
    # 2. Predict
    engine = get_inference_engine()
    if TF_AVAILABLE and hasattr(engine, 'predict') and not isinstance(engine, FallbackInferenceEngine):
        batch = np.expand_dims(preprocessed_img, axis=0)
        probs = engine.predict(batch, verbose=0)[0]
    else:
        probs = engine.predict(preprocessed_img)

    # 3. Format results
    top_idx = int(np.argmax(probs))
    top_class = CLASSES[top_idx]
    top_confidence = round(float(probs[top_idx]) * 100, 2)

    class_distribution = {
        cls_name: round(float(prob) * 100, 2)
        for cls_name, prob in zip(CLASSES, probs)
    }

    recycling_details = RECYCLING_GUIDELINES.get(top_class, {
        "bin": "General Waste Bin",
        "description": "Dispose of in designated municipal collection points.",
        "recyclable": False,
        "hazard_level": "Low"
    })

    return {
        "predicted_class": top_class,
        "confidence": top_confidence,
        "probabilities": class_distribution,
        "recycling_guidelines": recycling_details
    }
