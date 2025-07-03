from fastapi import FastAPI, UploadFile
from pathlib import Path
from ultralytics import YOLO
from truefoundry.ml import get_client
import pandas as pd
import uvicorn
import os

app = FastAPI(
    title="Pest Detection API",
    description="Detect pests and recommend pesticides using YOLOv11s",
    version="1.0.0"
)

# Setup directories
MODEL_PATH_DIR = Path("./model")
CSV_PATH_DIR = Path("./mapping")
MODEL_PATH_DIR.mkdir(exist_ok=True)
CSV_PATH_DIR.mkdir(exist_ok=True)

# TrueFoundry client
client = get_client()

# Download and load YOLOv11s model
if not any(MODEL_PATH_DIR.glob("*.pt")):
    print("Downloading YOLOv11s model from Model Registry...")
    artifact_version = client.get_artifact_version_by_fqn("model:iit-ropar/yolo-best-model/yolov11s-best:1")
    model_download_path = artifact_version.download(path=str(MODEL_PATH_DIR))
    MODEL_PATH = next(Path(model_download_path).glob("*.pt"), None)
    if not MODEL_PATH:
        raise FileNotFoundError("No .pt file found in model download")
    MODEL_PATH = MODEL_PATH.name
else:
    MODEL_PATH = next(MODEL_PATH_DIR.glob("*.pt")).name

model = YOLO(Path(MODEL_PATH_DIR) / MODEL_PATH)  # Load YOLO model

# Download and load pesticide mapping CSV
if not any(CSV_PATH_DIR.glob("*.csv")):
    print("Downloading pesticide mapping CSV from Model Registry...")
    artifact_version = client.get_artifact_version_by_fqn("artifact:iit-ropar/yolo-best-model/pesticide_mapping:1")
    csv_download_path = artifact_version.download(path=str(CSV_PATH_DIR))
    CSV_PATH = next(Path(csv_download_path).glob("*.csv"), None)
    if not CSV_PATH:
        raise FileNotFoundError("No .csv file found in mapping download")
    CSV_PATH = CSV_PATH.name
else:
    CSV_PATH = next(CSV_PATH_DIR.glob("*.csv")).name

pesticide_map = pd.read_csv(Path(CSV_PATH_DIR) / CSV_PATH)

# Inference Endpoint with CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "capacitor://localhost", "https://your-web-domain.com"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile):
    temp_file = "temp.jpg"
    contents = await file.read()
    with open(temp_file, "wb") as f:
        f.write(contents)

    results = model(temp_file)
    predictions = results[0].boxes.data.cpu().numpy().tolist() if results[0].boxes else []

    response = []
    for box in results[0].boxes:
        pest_name = model.names[int(box.cls)]
        confidence = float(box.conf)
        if pest_name in pesticide_map["pest_name"].values:
            pesticide = pesticide_map[pesticide_map["pest_name"] == pest_name]["pesticide"].iloc[0]
            response.append({
                "pest_name": pest_name,
                "confidence": confidence,
                "pesticide_recommendation": pesticide
            })
        else:
            response.append({
                "pest_name": pest_name,
                "confidence": confidence,
                "pesticide_recommendation": "No recommendation available"
            })

    return {"predictions": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
