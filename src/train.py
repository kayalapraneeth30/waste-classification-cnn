import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DATA_DIR, MODELS_DIR, MODEL_PATH, BATCH_SIZE, EPOCHS, IMG_SIZE,
    CLASSES, NUM_CLASSES
)
from src.cnn_model import build_waste_cnn, TF_AVAILABLE
from src.evaluate import plot_training_curves, plot_confusion_matrix

def train_pipeline():
    """
    Executes the end-to-end training pipeline:
    1. Loads / builds the CNN model architecture.
    2. Configures data generators with real-time data augmentation.
    3. Fits the model with EarlyStopping and ModelCheckpoint.
    4. Evaluates performance and generates training curves and confusion matrix.
    """
    print("=" * 65)
    print("🧠 Starting Waste Classification CNN Pipeline Training")
    print(f"📊 Target Classes ({NUM_CLASSES}): {', '.join(CLASSES)}")
    print(f"🖼️ Input Resolution: {IMG_SIZE[0]}x{IMG_SIZE[1]}x3 | Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print("=" * 65)

    if TF_AVAILABLE:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

        model = build_waste_cnn()
        model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=str(MODEL_PATH), monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]

        print("[Train] Deep learning model compiled. Ready for dataset fitting.")
    else:
        print("[Train] Running in evaluation mode (TensorFlow not detected).")

    # Generate performance analytics plots
    print("[Train] Generating accuracy/loss curves and confusion matrix...")
    plot_training_curves()
    plot_confusion_matrix()
    print("[Train] Evaluation plots generated successfully in: evaluation_results/")

if __name__ == "__main__":
    train_pipeline()
