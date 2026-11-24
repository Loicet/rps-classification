
import time
import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile

# Initialize the FastAPI application
app = FastAPI(
    title="Rock Paper Scissors Classifier API",
    description="An image classification API powered by TensorFlow",
    version="1.0.0"
)

# Model configuration
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "rps_baseline_model.keras"
)

CLASS_NAMES = ["rock", "paper", "scissors"]
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Prediction metrics
metrics = {
    "prediction_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "total_prediction_time": 0.0,
    "predictions_by_class": {name: 0 for name in ["rock", "paper", "scissors"]}
}

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model_status = "loaded"
except Exception as error:
    model = None
    model_status = f"failed: {str(error)}"


# 1. Welcome endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to the Rock Paper Scissors Classifier API!",
        "status": "running"
    }


# 2. Health check endpoint
@app.get("/health")
def health_check():
    return {
        "api_status": "running",
        "model_status": model_status,
        "model_loaded": model is not None
    }


# 3. Metrics endpoint
@app.get("/metrics")
def get_metrics():
    total = metrics["prediction_requests"]
    avg_time = (
        metrics["total_prediction_time"] / total if total > 0 else 0.0
    )
    return {
        **metrics,
        "average_prediction_time_seconds": round(avg_time, 4)
    }


# 4. Image prediction endpoint

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    start_time = time.perf_counter()
    metrics["prediction_requests"] += 1

    if model is None:
        metrics["failed_predictions"] += 1
        metrics["total_prediction_time"] += (
            time.perf_counter() - start_time
        )

        raise HTTPException(
            status_code=503,
            detail="The classification model is not available."
        )

    if file.content_type not in ["image/jpeg", "image/png"]:
        metrics["failed_predictions"] += 1
        metrics["total_prediction_time"] += (
            time.perf_counter() - start_time
        )

        raise HTTPException(
            status_code=400,
            detail="Please upload a JPEG or PNG image."
        )

    try:
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))

        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[0][predicted_index])

        processing_time = time.perf_counter() - start_time

        # Update successful prediction metrics
        metrics["successful_predictions"] += 1
        metrics["predictions_by_class"][predicted_class] += 1
        metrics["total_prediction_time"] += processing_time

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "processing_time_seconds": round(processing_time, 4),
            "probabilities": {
                CLASS_NAMES[i]: round(float(predictions[0][i]), 4)
                for i in range(len(CLASS_NAMES))
            }
        }

    except Exception as error:
        metrics["failed_predictions"] += 1
        metrics["total_prediction_time"] += (
            time.perf_counter() - start_time
        )

        raise HTTPException(
            status_code=400,
            detail=f"Could not process the image: {str(error)}"
        )

