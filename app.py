"""
Facial Recognition MLOps — FastAPI Backend
Team: Edwin, Glory, Justine, Mike — ALU BSE
"""

import os, json, shutil, sqlite3, io, base64, zipfile, time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from src.prediction import predict_face
from src.database import init_db, save_upload, get_all_uploads, get_upload_stats

app = FastAPI(title="Facial Recognition MLOps", version="1.0.0")

# Mount static files (create dir if missing)
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Paths
MODEL_PATH     = "models/facial_recognition_model.keras"
CLASS_MAP_PATH = "models/class_names.json"
UPLOAD_DIR     = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Ensure data dir for DB
Path("data").mkdir(exist_ok=True)


# ── Init DB on startup ──────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()


# ── UI ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── HEALTH ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


# ── CLASSES ──────────────────────────────────────────────────────────────────
@app.get("/classes")
async def get_classes():
    if not os.path.exists(CLASS_MAP_PATH):
        return {"classes": []}
    with open(CLASS_MAP_PATH) as f:
        mapping = json.load(f)
    return {"classes": list(mapping.values())}


# ── PREDICT ─────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an image, return the predicted class + confidence."""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(503, "Model not found. Place facial_recognition_model.keras in /models/")

    tmp_path = f"/tmp/pred_{int(time.time())}_{file.filename}"
    with open(tmp_path, "wb") as f_out:
        f_out.write(await file.read())

    try:
        result = predict_face(tmp_path, MODEL_PATH, CLASS_MAP_PATH)
        with open(tmp_path, "rb") as f_in:
            img_b64 = base64.b64encode(f_in.read()).decode()
        result["image_b64"] = img_b64
        result["filename"]  = file.filename
        return JSONResponse(result)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── UPLOAD (save for retraining) ─────────────────────────────────────────────
@app.post("/upload")
async def upload_for_retraining(
    file: UploadFile = File(...),
    label: str = Form(...)
):
    """Upload a labeled image into the DB + disk for future retraining."""
    label = label.strip().lower().replace(" ", "_")
    dest_dir = UPLOAD_DIR / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename  = f"{int(time.time()*1000)}_{file.filename}"
    dest_path = dest_dir / filename

    content = await file.read()
    with open(dest_path, "wb") as f_out:
        f_out.write(content)

    save_upload(label=label, filename=filename, filepath=str(dest_path))

    return {"status": "saved", "label": label, "file": filename, "path": str(dest_path)}


# ── RETRAIN ─────────────────────────────────────────────────────────────────
@app.post("/retrain")
async def retrain_endpoint(
    epochs: int = Form(15),
    lr: float = Form(0.00005)
):
    """Retrain the model on all uploaded images."""
    from src.retrain import retrain_model

    classes = [d for d in UPLOAD_DIR.iterdir() if d.is_dir()]
    if len(classes) < 2:
        raise HTTPException(400, "Need images for at least 2 classes in uploads/ to retrain.")

    counts = {
        c.name: len(list(c.glob("*.jpg")) + list(c.glob("*.jpeg")) + list(c.glob("*.png")))
        for c in classes
    }
    if any(v < 2 for v in counts.values()):
        raise HTTPException(400, f"Need ≥2 images per class. Current counts: {counts}")

    result = retrain_model(
        dataset_dir=str(UPLOAD_DIR),
        model_path=MODEL_PATH,
        class_map_path=CLASS_MAP_PATH,
        epochs=epochs,
        learning_rate=lr
    )
    return {"status": "retrained", "metrics": result, "class_counts": counts}


# ── INSIGHTS ─────────────────────────────────────────────────────────────────
@app.get("/insights")
async def insights():
    """Return dataset statistics from DB + disk."""
    stats = get_upload_stats()

    config = {}
    if os.path.exists("models/training_config.json"):
        with open("models/training_config.json") as f:
            config = json.load(f)

    class_names = []
    if os.path.exists(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH) as f:
            mapping = json.load(f)
        class_names = list(mapping.values())

    upload_counts = {}
    if UPLOAD_DIR.exists():
        for d in UPLOAD_DIR.iterdir():
            if d.is_dir():
                upload_counts[d.name] = len(
                    list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
                )

    return {
        "db_stats":        stats,
        "training_config": config,
        "class_names":     class_names,
        "upload_counts":   upload_counts,
        "model_exists":    os.path.exists(MODEL_PATH)
    }
