# FaceID MLOps — Facial Recognition Web App
**Team:** Edwin, Glory, Justine, Mike | **ALU BSE**

---

## Quick Start

### Docker (Recommended for submission)
```bash
# 1. Copy model files into models/
cp facial_recognition_model.keras models/
cp class_names.json models/
cp training_config.json models/          # optional

# 2. Build & run
docker compose up --build

# 3. Open http://localhost:8000
```

### Local Python
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

---

## Project Structure
```
├── app.py                    # FastAPI — all API routes
├── src/
│   ├── prediction.py         # predict_face() — loads model, runs inference
│   ├── retrain.py            # retrain_model() — fine-tunes on uploaded data
│   └── database.py           # SQLite — persists uploads & stats
├── templates/
│   └── index.html            # Single-page UI (4 tabs)
├── models/                   # ← Place .keras + class_names.json here
├── uploads/                  # Auto-populated by /upload endpoint
├── data/                     # SQLite DB lives here
├── notebooks/
│   └── facial_recognition_training.ipynb
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## MLOps Pipeline

### 1 · Predict  (`/predict`)
Upload any face image → MobileNetV2 returns predicted identity + confidence bars

### 2 · Upload Data  (`/upload`)
Upload labeled face images → saved to SQLite DB + `uploads/<label>/` folder

### 3 · Retrain  (`/retrain`)
- Preprocesses uploaded images (normalize, augment, 80/20 split)  
- Loads saved `.keras` model as pretrained backbone  
- Fine-tunes with Adam + EarlyStopping + ReduceLROnPlateau  
- Saves updated model + class mapping  

### 4 · Insights  (`/insights`)
Dataset counts, image distribution chart, model config, DB stats

---

## Evaluation Metrics (notebook)
Accuracy · Loss · Precision · Recall · F1-Score · Confusion Matrix · ROC-AUC
