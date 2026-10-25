import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "evaluation_results"
STATIC_DIR = BASE_DIR / "app" / "static"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Classes
CLASSES = ["Plastic", "Paper", "Glass", "Metal", "Organic"]
NUM_CLASSES = len(CLASSES)

# Image & Training Hyperparameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
CHANNELS = 3
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, CHANNELS)

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Model Artifacts
MODEL_PATH = MODELS_DIR / "waste_cnn_best_model.h5"
CURVES_PLOT_PATH = RESULTS_DIR / "training_validation_curves.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

# Eco-Friendly Recycling Guidelines per Class
RECYCLING_GUIDELINES = {
    "Plastic": {
        "bin": "Blue Recycling Bin",
        "description": "Clean and rinse bottles, containers, and packaging. Remove liquid residues and squeeze flat to save space.",
        "recyclable": True,
        "hazard_level": "Moderate"
    },
    "Paper": {
        "bin": "Green Paper Bin",
        "description": "Flatten cardboard boxes and keep dry. Soiled food wrappers or greasy paper cannot be recycled with clean paper.",
        "recyclable": True,
        "hazard_level": "Low"
    },
    "Glass": {
        "bin": "Teal Glass Bin",
        "description": "Rinse glass bottles and jars thoroughly. Metal caps and corks should be removed and disposed of separately.",
        "recyclable": True,
        "hazard_level": "High (Sharp edges)"
    },
    "Metal": {
        "bin": "Yellow Metals Bin",
        "description": "Rinse aluminum soda cans, food tins, and foil. Highly recyclable without degradation of material quality.",
        "recyclable": True,
        "hazard_level": "Low"
    },
    "Organic": {
        "bin": "Brown Compost Bin",
        "description": "Food scraps, fruit peels, yard trimmings, and biodegradable materials suitable for campus composting.",
        "recyclable": True,
        "hazard_level": "Biological/Decomposable"
    }
}
