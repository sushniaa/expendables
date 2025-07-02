from fastapi import FastAPI, UploadFile
from pathlib import Path
from ultralytics import YOLO
from truefoundry.ml import get_client
import pandas as pd
import uvicorn

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
    artifact_version = client.get_artifact_version_by_fqn(
        "model:iit-ropar/yolo-best-model/yolov11s-best:1"
    )
    model_download_path = artifact_version.download(path=str(MODEL_PATH_DIR))
else:
    model_download_path = list(MODEL_PATH_DIR.glob("*.pt"))[0]

model = YOLO(str(model_download_path))  # Load YOLO model

# Download and load pesticide mapping CSV
if not any(CSV_PATH_DIR.glob("*.csv")):
    print("Downloading pesticide mapping CSV from Model Registry...")
    artifact_version = client.get_artifact_version_by_fqn(
        "artifact:iit-ropar/yolo-best-model/pesticide_mapping:1"
    )
    csv_download_path = artifact_version.download(path=str(CSV_PATH_DIR))
else:
    csv_download_path = list(CSV_PATH_DIR.glob("*.csv"))[0]

pesticide_map = pd.read_csv(str(csv_download_path))

# Inference Endpoint
@app.post("/predict")
async def predict(file: UploadFile):
    temp_file = "temp.jpg"
    contents = await file.read()
    with open(temp_file, "wb") as f:
        f.write(contents)

    results = model(temp_file)
    predictions = results[0].to("cpu").numpy().tolist() if results else []

    response = []

    for pred in results[0].names.values():
        pest_name = pred
        confidence = float(results[0].probs.get(pred_name, 0.0))

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
