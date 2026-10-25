import io
import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Dict, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    CLASSES, RESULTS_DIR, STATIC_DIR, DATA_DIR, RECYCLING_GUIDELINES,
    CURVES_PLOT_PATH, CONFUSION_MATRIX_PATH
)
from src.predict import classify_waste_image
from src.evaluate import generate_classification_report, plot_training_curves, plot_confusion_matrix
from src.dataset_generator import create_sample_waste_dataset

app = FastAPI(
    title="Waste Classification CNN Web API",
    description="Deep learning image classification system for automated categorization of waste into plastic, paper, glass, metal, and organic waste.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# On startup, ensure evaluation plots & sample images exist
@app.on_event("startup")
def startup_event():
    if not CURVES_PLOT_PATH.exists():
        plot_training_curves()
    if not CONFUSION_MATRIX_PATH.exists():
        plot_confusion_matrix()
    sample_dir = DATA_DIR / "sample_images"
    if not sample_dir.exists() or not list(sample_dir.glob("**/*.jpg")):
        create_sample_waste_dataset()

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Waste Classification CNN",
        "target_classes": CLASSES,
        "input_dimensions": "128x128x3"
    }

@app.post("/api/classify")
async def classify_image_endpoint(file: UploadFile = File(...)):
    """Classifies an uploaded image into one of the 5 waste categories."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG, PNG, WebP).")

    contents = await file.read()
    try:
        result = classify_waste_image(contents)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.get("/api/sample/{category}")
def get_sample_image(category: str):
    """Provides a sample image for the specified category for one-click testing."""
    cat_norm = category.capitalize()
    if cat_norm not in CLASSES:
        raise HTTPException(status_code=404, detail="Invalid waste category.")
    
    cat_dir = DATA_DIR / "sample_images" / cat_norm.lower()
    images = list(cat_dir.glob("*.jpg"))
    if not images:
        create_sample_waste_dataset()
        images = list(cat_dir.glob("*.jpg"))

    if images:
        return FileResponse(images[0], media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Sample image not found.")

@app.get("/api/metrics")
def get_metrics():
    """Returns classification report with precision, recall, F1, and overall accuracy."""
    return generate_classification_report()

@app.get("/api/results/curves")
def get_training_curves():
    if not CURVES_PLOT_PATH.exists():
        plot_training_curves()
    return FileResponse(CURVES_PLOT_PATH, media_type="image/png")

@app.get("/api/results/confusion-matrix")
def get_confusion_matrix():
    if not CONFUSION_MATRIX_PATH.exists():
        plot_confusion_matrix()
    return FileResponse(CONFUSION_MATRIX_PATH, media_type="image/png")

# Mount Static Files & Index
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Waste Classification CNN API is active. Check /docs for documentation."}
