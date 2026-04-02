
# FaceID MLOps — Facial Recognition Web App
**Student:** Justine | **ALU BSE — Machine Learning Operations**

A Demo Video : https://www.loom.com/share/6e793e34688a490abcec2042f664325d
---

## Live Demo
Deployed via Cloudflare Tunnel on Google Colab (T4 GPU).  
Run `run_app_v3.ipynb` to get a live public URL.

---

## Quick Start

### Option A — Google Colab (Recommended)
1. Open `run_app_v3.ipynb` in Google Colab
2. Run all cells
3. Copy the `trycloudflare.com` URL printed in the last cell

### Option B — Docker
```bash
# Place model files in models/ first
cp facial_recognition_model.keras models/
cp class_names.json models/
cp training_config.json models/

docker compose up --build
# Open http://localhost:8000
```

### Option C — Local Python
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

---

## Project Structure
```
├── app.py                    # FastAPI — all API routes
├── src/
│   ├── prediction.py         # predict_face() — MobileNetV2 inference
│   ├── retrain.py            # retrain_model() — fine-tunes on new data
│   └── database.py           # SQLite — persists uploads & prediction logs
├── templates/
│   └── index.html            # Single-page UI (4 tabs)
├── models/                   # Place .keras + class_names.json here
├── uploads/                  # Auto-populated by /upload endpoint
├── data/                     # SQLite database lives here
├── notebooks/
│   └── facial_recognition_training_FINAL.ipynb
├── run_app_v3.ipynb          # Colab deployment notebook
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## MLOps Pipeline

### 1 · Predict (`/predict`)
Upload any face image → MobileNetV2 identifies the person + shows confidence bars for all classes

### 2 · Upload Data (`/upload`)
Upload labeled face images → saved to SQLite database + `uploads/<label>/` folder on disk

### 3 · Retrain (`/retrain`)
- Loads uploaded images directly (works with small datasets)
- Uses saved `.keras` model as pretrained backbone
- Fine-tunes with Adam optimizer + EarlyStopping
- Saves updated model weights + new class mapping

### 4 · Insights (`/insights`)
Dataset counts per class, uploaded image distribution chart, model architecture info, DB stats

---

## Model Architecture

| Component | Detail |
|---|---|
| Base Model | MobileNetV2 (ImageNet pretrained) |
| Fine-tuned layers | Layers 100+ unfrozen |
| Head | GAP → BatchNorm → Dense(128) + L2 → Dropout(0.5) → Dense(64) + L2 → Dropout(0.3) → Softmax |
| Optimizer | Adam (lr=0.0001) |
| Regularization | L2 (λ=0.01) + Dropout |
| Callbacks | EarlyStopping (patience=7) + ReduceLROnPlateau |
| Augmentation | Offline 35× per image + real-time (6 techniques) |

---

## Evaluation Metrics (see notebook)

| Metric | Result |
|---|---|
| Accuracy | 100% |
| Loss | ✅ |
| Precision | 1.0000 |
| Recall | 0.9032 |
| F1-Score | 1.0000 |
| Confusion Matrix | ✅ |
| ROC-AUC | ✅ |

---

## Dataset
- **Subjects:** Edwin, Glory, Justine, mike (4 team members)
- **Original images:** 1–3 per person
- **After augmentation:** ~36–108 per class
- **Augmentation techniques:** Rotation, flip, brightness, contrast, color jitter, blur, zoom, sharpness
