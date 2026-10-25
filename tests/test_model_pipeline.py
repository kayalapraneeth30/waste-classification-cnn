import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import CLASSES, NUM_CLASSES, IMG_SIZE, RESULTS_DIR
from src.data_preprocessing import load_and_preprocess_image, augment_image
from src.predict import classify_waste_image
from src.evaluate import plot_training_curves, plot_confusion_matrix, generate_classification_report
from src.dataset_generator import create_sample_waste_dataset

def test_pipeline():
    print("\n--- [1] Testing Synthetic Waste Dataset Generation ---")
    files = create_sample_waste_dataset(samples_per_class=2)
    assert len(files) == 10, f"Expected 10 sample images, got {len(files)}"
    print(f"[OK] Generated {len(files)} sample benchmark images across {NUM_CLASSES} classes.")

    print("\n--- [2] Testing Image Preprocessing & Normalization ---")
    sample_file = files[0]
    processed_img = load_and_preprocess_image(sample_file)
    assert processed_img.shape == (IMG_SIZE[0], IMG_SIZE[1], 3), f"Unexpected shape {processed_img.shape}"
    assert 0.0 <= processed_img.min() and processed_img.max() <= 1.0, "Normalization out of range [0, 1]"
    print(f"[OK] Preprocessed image validated: shape={processed_img.shape}, min={processed_img.min():.2f}, max={processed_img.max():.2f}")

    print("\n--- [3] Testing Data Augmentation Pipeline ---")
    augmented = augment_image(processed_img)
    assert augmented.shape == processed_img.shape, "Augmentation changed tensor shape"
    print("[OK] Data augmentation pipeline verified.")

    print("\n--- [4] Testing Inference & Classification ---")
    result = classify_waste_image(sample_file)
    assert "predicted_class" in result
    assert result["predicted_class"] in CLASSES
    assert 0.0 <= result["confidence"] <= 100.0
    assert len(result["probabilities"]) == NUM_CLASSES
    print(f"[OK] Prediction Result: {result['predicted_class']} ({result['confidence']}%)")

    print("\n--- [5] Testing Performance Curves & Confusion Matrix ---")
    plot_training_curves()
    plot_confusion_matrix()
    curves_file = RESULTS_DIR / "training_validation_curves.png"
    cm_file = RESULTS_DIR / "confusion_matrix.png"
    assert curves_file.exists(), "Curves plot not found"
    assert cm_file.exists(), "Confusion matrix plot not found"
    print(f"[OK] Verified generated plots: {curves_file.name}, {cm_file.name}")

    report = generate_classification_report()
    assert report["overall_accuracy"] >= 90.0
    print(f"[OK] Classification report verified (Overall Accuracy: {report['overall_accuracy']}%)")

    print("\n[SUCCESS] All Waste Classification CNN tests passed successfully!")

if __name__ == "__main__":
    test_pipeline()
