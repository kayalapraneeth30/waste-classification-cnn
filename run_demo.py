import uvicorn
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    print("=" * 65)
    print("♻️ Starting Waste Classification CNN Web App")
    print("📍 Local Web App:  http://127.0.0.1:8001")
    print("📖 API Swagger UI: http://127.0.0.1:8001/docs")
    print("=" * 65)
    uvicorn.run("app.server:app", host="127.0.0.1", port=8001, reload=True)
