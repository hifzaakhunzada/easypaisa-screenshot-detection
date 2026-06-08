"""
modules/ml_model.py
Machine learning module — CNN classifier for real vs fake payment screenshots.
Uses MobileNetV2 transfer learning for efficiency on small datasets.
"""

import os
import numpy as np
from pathlib import Path

# Lazy TensorFlow import to avoid slow startup when not needed
_tf = None
_keras = None

def _get_tf():
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        from tensorflow import keras
        _tf = tf
        _keras = keras
    return _tf, _keras


_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(_MODEL_DIR, "payment_detector.keras")
META_PATH = os.path.join(_MODEL_DIR, "payment_detector_meta.json")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model() -> "keras.Model":
    """
    Build a transfer-learning model using MobileNetV2 as the backbone.
    Binary sigmoid output is P(class index 1) from flow_from_directory
    (with classes=['fake','real'] that is P(real); see payment_detector_meta.json).
    """
    tf, keras = _get_tf()

    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # Freeze pretrained weights initially

    model = keras.Sequential([
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),  # Binary output
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(data_dir: str, epochs: int = EPOCHS, save: bool = True):
    """
    Train the model on a directory with structure:
        data_dir/
            real/   ← genuine screenshots
            fake/   ← tampered screenshots

    Returns training history.
    """
    tf, keras = _get_tf()

    # Data augmentation for training
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,  # Don't flip payment UIs — text would break
        zoom_range=0.1,
    )

    # Lock class order so label 0 = fake, 1 = real (alphabetical default matches this).
    class_subset = ["fake", "real"]

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=class_subset,
        subset="training",
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=class_subset,
        subset="validation",
    )

    model = build_model()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]

    import math
    total = train_gen.samples
    fake_count = train_gen.classes.tolist().count(1)
    real_count = total - fake_count

    class_weight = {
        0: total / (2 * real_count),   # Real
        1: total / (2 * fake_count),   # Fake — gets higher weight if underrepresented
    }
    print(f"Class weights: Real={class_weight[0]:.2f}, Fake={class_weight[1]:.2f}")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    if save:
        os.makedirs(_MODEL_DIR, exist_ok=True)
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        # Persist how Keras maps folders → labels so inference matches training.
        idx_to_class = {int(v): k for k, v in train_gen.class_indices.items()}
        meta = {
            "class_indices": train_gen.class_indices,
            "index_to_class": {str(k): v for k, v in idx_to_class.items()},
            "sigmoid_is_probability_of_class_index_1": True,
        }
        import json
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Training metadata saved to {META_PATH}")

    return model, history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_loaded_model = None
_meta_cache = None


def _load_training_meta() -> dict:
    """How class 0/1 map to folder names; used to turn sigmoid into P(fake)."""
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache
    import json
    if os.path.isfile(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            _meta_cache = json.load(f)
        return _meta_cache
    # Older checkpoints trained with default flow_from_directory: fake=0, real=1.
    _meta_cache = {
        "class_indices": {"fake": 0, "real": 1},
        "index_to_class": {"0": "fake", "1": "real"},
        "sigmoid_is_probability_of_class_index_1": True,
    }
    return _meta_cache


def load_model():
    """Load the saved model (cached after first call)."""
    global _loaded_model
    if _loaded_model is None:
        tf, keras = _get_tf()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. "
                "Run train_model() first."
            )
        _loaded_model = keras.models.load_model(MODEL_PATH)
    return _loaded_model


def predict(image_path: str) -> dict:
    """
    Predict whether a payment screenshot is real or fake.
    Returns a dict with 'label' ('Real'/'Fake') and 'confidence' (0–1).

    Keras binary cross-entropy treats the sigmoid as P(y=1) where y is the
    integer class label for the *second* class (index 1). With folders fake/real,
    that is P(real), not P(fake). ``ml_score`` is always P(fake) in [0, 1].
    """
    tf, keras = _get_tf()
    model = load_model()
    meta = _load_training_meta()

    img = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    arr = keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    proba_class_1 = float(model.predict(arr, verbose=0)[0][0])
    idx_to_class = {int(k): v for k, v in meta.get("index_to_class", {"0": "fake", "1": "real"}).items()}
    name_1 = idx_to_class.get(1, "real")

    if name_1.lower() == "real":
        proba_fake = 1.0 - proba_class_1
    elif name_1.lower() == "fake":
        proba_fake = proba_class_1
    else:
        # Unknown layout: assume alphabetical fake=0, real=1.
        proba_fake = 1.0 - proba_class_1

    label = "Fake" if proba_fake >= 0.5 else "Real"
    confidence = proba_fake if label == "Fake" else 1.0 - proba_fake

    return {
        "label": label,
        "raw_score": round(proba_class_1, 4),
        "confidence": round(confidence, 4),
        "ml_score": round(proba_fake, 4),
    }


def model_is_trained() -> bool:
    return os.path.exists(MODEL_PATH)
