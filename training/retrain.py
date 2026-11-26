
import os
import random
import shutil
import numpy as np
import tensorflow as tf

from PIL import Image


# Keep this order consistent with the original model
CLASS_NAMES = ["rock", "paper", "scissors"]

IMAGE_SIZE = (150, 150)
UPLOAD_FOLDER = "uploaded_data"


def load_uploaded_images():
    images = []
    labels = []

    validation_images = []
    validation_labels = []

    rng = random.Random(42)

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(UPLOAD_FOLDER, class_name)

        if not os.path.isdir(class_folder):
            raise ValueError(f"Missing folder: {class_folder}")

        image_files = [
            filename for filename in os.listdir(class_folder)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        rng.shuffle(image_files)

        if len(image_files) < 3:
            raise ValueError(
                f"At least 3 images are required for {class_name}. "
                f"Found {len(image_files)}."
            )

        # Reserve approximately 25% for validation,
        # while keeping at least two training images.
        validation_count = max(1, round(len(image_files) * 0.25))

        training_files = image_files[validation_count:]
        validation_files = image_files[:validation_count]

        for filename in training_files:
            image_path = os.path.join(class_folder, filename)

            try:
                image = Image.open(image_path).convert("RGB")
                image = image.resize(IMAGE_SIZE)

                image_array = np.array(image, dtype=np.float32) / 255.0

                images.append(image_array)
                labels.append(class_index)

            except Exception:
                print(f"Skipping invalid image: {image_path}")

        for filename in validation_files:
            image_path = os.path.join(class_folder, filename)

            try:
                image = Image.open(image_path).convert("RGB")
                image = image.resize(IMAGE_SIZE)

                image_array = np.array(image, dtype=np.float32) / 255.0

                validation_images.append(image_array)
                validation_labels.append(class_index)

            except Exception:
                print(f"Skipping invalid image: {image_path}")

    if not images or not validation_images:
        raise ValueError("Not enough valid images to retrain the model.")

    return (
        np.array(images),
        np.array(labels),
        np.array(validation_images),
        np.array(validation_labels)
    )


def retrain_model(model_path):

    print("Loading uploaded images...")

    X_train, y_train, X_val, y_val = load_uploaded_images()

    print(f"Training images: {len(X_train)}")
    print(f"Validation images: {len(X_val)}")

    print("\nLoading existing model...")

    original_model = tf.keras.models.load_model(model_path)

    # Evaluate the existing model first
    original_loss, original_accuracy = original_model.evaluate(
        X_val, y_val, verbose=0
    )

    print(f"Original validation accuracy: {original_accuracy:.4f}")

    # Create a separate candidate model
    candidate_model = tf.keras.models.clone_model(original_model)
    candidate_model.set_weights(original_model.get_weights())

    candidate_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nFine-tuning candidate model...")

    candidate_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=4,
        verbose=1
    )

    candidate_loss, candidate_accuracy = candidate_model.evaluate(
        X_val, y_val, verbose=0
    )

    print(f"\nCandidate validation accuracy: {candidate_accuracy:.4f}")

    promoted = False

    if candidate_accuracy > original_accuracy:

        print("\nCandidate performed better. Promoting new model...")

        backup_path = model_path.replace(
            ".keras", "_backup.keras"
        )

        candidate_path = model_path.replace(
            ".keras", "_candidate.keras"
        )

        # Preserve the previous model
        shutil.copy2(model_path, backup_path)

        # Save candidate before replacing the existing model
        candidate_model.save(candidate_path)

        os.replace(candidate_path, model_path)

        promoted = True

        print("New model successfully promoted.")

    else:
        print("\nCandidate did not improve validation accuracy.")
        print("Keeping the original model.")

    return {
        "original_validation_accuracy": float(original_accuracy),
        "candidate_validation_accuracy": float(candidate_accuracy),
        "promoted": promoted,
        "training_images": len(X_train),
        "validation_images": len(X_val)
    }
