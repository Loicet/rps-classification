
import time
import io
import os
import uuid

import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from typing import List

from training.retrain import retrain_model


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

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


# Prediction metrics
metrics = {
    "prediction_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "total_prediction_time": 0.0,
    "predictions_by_class": {
        name: 0 for name in CLASS_NAMES
    }
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
        metrics["total_prediction_time"] / total
        if total > 0 else 0.0
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

        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds the 5MB limit."
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

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

    except HTTPException:
        raise

    except Exception as error:
        metrics["failed_predictions"] += 1
        metrics["total_prediction_time"] += (
            time.perf_counter() - start_time
        )

        raise HTTPException(
            status_code=400,
            detail=f"Could not process the image: {str(error)}"
        )


# 5. Bulk image upload endpoint
@app.post("/upload-data/{class_name}")
async def upload_data(
    class_name: str,
    files: List[UploadFile] = File(...)
):

    # Only accept our three known classes
    if class_name not in CLASS_NAMES:
        raise HTTPException(
            status_code=400,
            detail="Class must be rock, paper, or scissors."
        )

    # Create the class folder if it doesn't exist
    project_folder = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    upload_folder = os.path.join(
        project_folder,
        "uploaded_data",
        class_name
    )

    os.makedirs(upload_folder, exist_ok=True)

    uploaded_files = []
    rejected_files = []

    for file in files:
        try:
            extension = os.path.splitext(
                file.filename or ""
            )[1].lower()

            if extension not in [".jpg", ".jpeg", ".png"]:
                rejected_files.append({
                    "filename": file.filename,
                    "reason": "Only JPG, JPEG, and PNG files are accepted."
                })
                continue

            contents = await file.read()

            if len(contents) > 10 * 1024 * 1024:
                rejected_files.append({
                    "filename": file.filename,
                    "reason": "File exceeds the 10 MB limit."
                })
                continue

            image = Image.open(io.BytesIO(contents))
            image.verify()

            new_filename = f"{uuid.uuid4().hex}{extension}"

            file_path = os.path.join(
                upload_folder,
                new_filename
            )

            with open(file_path, "wb") as image_file:
                image_file.write(contents)

            uploaded_files.append(new_filename)

        except Exception:
            rejected_files.append({
                "filename": file.filename,
                "reason": "The file could not be processed as a valid image."
            })

    return {
        "class_name": class_name,
        "uploaded_count": len(uploaded_files),
        "rejected_count": len(rejected_files),
        "uploaded_files": uploaded_files,
        "rejected_files": rejected_files
    }


# 6. Model retraining endpoint
@app.post("/retrain")
def retrain():

    global model, model_status

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The current model is not available."
        )

    try:
        result = retrain_model(MODEL_PATH)

        if result["promoted"]:
            # Load the newly promoted model into the API
            model = tf.keras.models.load_model(MODEL_PATH)
            model_status = "loaded"

        return {
            "message": (
                "New model successfully promoted."
                if result["promoted"]
                else "Candidate did not improve. Original model retained."
            ),
            "retraining_status": (
                "promoted" if result["promoted"] else "not_promoted"
            ),
            **result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Retraining failed: {str(error)}"
        )
